"""
CASP synthesizability probability proxy for highly-novel molecules.

ASKCOS itself requires a Docker-deployed server (not available here:
no Docker/WSL; askcos-demo.mit.edu returns 401).

Closest offline proxy used in the literature:
  RAscore (Reymond) — P(AiZynthFinder finds a route), trained on ChEMBL
  CASP labels. Papers compare AiZynthFinder solve rates to ASKCOS.

Also reports existing SCScore (Coley / ASKCOS synthetic complexity utility).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

CURATED = Path(__file__).resolve().parents[1]
OUT = CURATED / "generator_check" / "synthesizability"
RASCORE_ROOT = Path(__file__).resolve().parent / "vendor" / "RAscore"
sys.path.insert(0, str(RASCORE_ROOT))

import xgboost as xgb  # noqa: E402
from RAscore.RAscore_XGB import RAScorerXGB  # noqa: E402


def rascore_predict(scorer: RAScorerXGB, smiles: str) -> float:
    """Compatible predict across xgboost versions (old sklearn pickle → booster)."""
    arr = scorer.ecfp(smiles).reshape(1, -1)
    model = scorer.xgb_model
    try:
        return float(model.predict_proba(arr)[0][1])
    except Exception:
        dmat = xgb.DMatrix(arr)
        pred = model.get_booster().predict(dmat)
        return float(np.asarray(pred).ravel()[0])


def main() -> None:
    nov = pd.read_csv(OUT / "tables" / "scscore_pass_novelty.csv")
    hn = nov[nov["novelty_tier_local"] == "highly_novel"].sort_values("max_tc_any_ref").reset_index(drop=True)

    scorer = RAScorerXGB()
    rows = []
    for i, r in hn.iterrows():
        smi = r["canon"]
        try:
            ra = rascore_predict(scorer, smi)
        except Exception as e:
            ra = float("nan")
            print(f"[WARN] RAscore failed for #{i+1}: {e}")
        # interpret
        if ra >= 0.75:
            label = "likely_accessible"
        elif ra >= 0.40:
            label = "uncertain"
        else:
            label = "likely_hard"
        rows.append(
            {
                "id": i + 1,
                "SMILES": smi,
                "score": r["score"],
                "sa": r["sa"],
                "scscore": r["scscore"],
                "max_tc_any_ref": r["max_tc_any_ref"],
                "lambda_nm": r["lambda_nm"],
                "RAscore": ra,
                "casp_access_label": label,
                "method_note": "RAscore_XGB = P(AiZynthFinder route); ASKCOS server unavailable",
            }
        )
    res = pd.DataFrame(rows)
    tab = OUT / "tables"
    res.to_csv(tab / "highly_novel_askcos_proxy_rascore.csv", index=False)
    summary = {
        "askcos_status": "unavailable_locally",
        "askcos_reason": [
            "Docker not installed",
            "WSL not installed",
            "askcos-demo.mit.edu API returns 401 (auth required)",
        ],
        "proxy_method": "RAscore_XGB (Reymond) — ML estimate of CASP route-finding probability",
        "proxy_relation_to_askcos": (
            "RAscore is trained on AiZynthFinder solved/unsolved labels; "
            "AiZynthFinder and ASKCOS are both CASP tools used for synthesizability filtering "
            "(see Thakkar et al., Chem. Sci. 2021). Not identical to ASKCOS Tree Builder."
        ),
        "n": int(len(res)),
        "mean_RAscore": float(res["RAscore"].mean()),
        "label_counts": {k: int(v) for k, v in res["casp_access_label"].value_counts().items()},
        "molecules": res.to_dict(orient="records"),
    }
    (tab / "highly_novel_askcos_proxy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # figure: structures + RAScore bars
    mols, legs = [], []
    for _, r in res.iterrows():
        m = Chem.MolFromSmiles(r["SMILES"])
        AllChem.Compute2DCoords(m)
        mols.append(m)
        legs.append(
            f"#{int(r['id'])}  RAscore={r['RAscore']:.3f}  ({r['casp_access_label']})\n"
            f"SCScore={r['scscore']:.2f}  SA={r['sa']:.2f}  Score={r['score']:.3f}"
        )
    grid = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(460, 380), legends=legs, returnPNG=False)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = []
    for v in res["RAscore"]:
        if v >= 0.75:
            colors.append("#00c853")
        elif v >= 0.40:
            colors.append("#ffab00")
        else:
            colors.append("#d50000")
    ax.bar([f"#{i}" for i in res["id"]], res["RAscore"], color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0.75, color="#00c853", ls="--", lw=1, label="likely (≥0.75)")
    ax.axhline(0.40, color="#ffab00", ls="--", lw=1, label="uncertain (≥0.40)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("RAscore ≈ P(CASP finds a route)")
    ax.set_title("Synthesizability probability proxy (ASKCOS unavailable → RAscore)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    bar_path = OUT / "_tmp_rascore_bars.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # combine grid + bars
    bars = Image.open(bar_path)
    gw, gh = grid.size
    bw, bh = bars.size
    # scale bars to grid width
    new_bh = int(bh * gw / bw)
    bars = bars.resize((gw, new_bh), Image.Resampling.LANCZOS)
    pad = 64
    canvas = Image.new("RGB", (gw, pad + gh + new_bh + 16), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_s = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_s = font
    draw.text((16, 10), "Highly novel molecules — CASP synthesizability proxy", fill=(20, 20, 20), font=font)
    draw.text(
        (16, 36),
        "ASKCOS Tree Builder not runnable here · RAscore = P(AiZynthFinder finds route)",
        fill=(90, 90, 90),
        font=font_s,
    )
    canvas.paste(grid, (0, pad))
    canvas.paste(bars, (0, pad + gh + 8))
    out_png = OUT / "09_highly_novel_casp_proxy_rascore.png"
    canvas.save(out_png)
    bar_path.unlink(missing_ok=True)

    print(json.dumps(summary, indent=2))
    print(f"[DONE] {out_png}")


if __name__ == "__main__":
    main()
