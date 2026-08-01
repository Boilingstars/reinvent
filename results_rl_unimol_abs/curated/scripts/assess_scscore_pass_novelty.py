"""Novelty assessment for SCScore-pass molecules vs train / ChEMBL / molecules.csv (+ optional PubChem)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")

CURATED = Path(__file__).resolve().parents[1]
OUT = CURATED / "generator_check" / "synthesizability"
PASS = OUT / "tables" / "scscore_pass_sorted.csv"


def canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True) if m else None


def fps_from_smiles(smiles_list: list[str]) -> tuple[list[str], list]:
    valid, fps = [], []
    seen = set()
    for s in smiles_list:
        c = canon(s)
        if not c or c in seen:
            continue
        seen.add(c)
        m = Chem.MolFromSmiles(c)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        valid.append(c)
        fps.append(fp)
    return valid, fps


def max_tc(query_fps: list, ref_fps: list) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(len(query_fps))
    nearest_idx = np.full(len(query_fps), -1, dtype=int)
    if not ref_fps:
        return out, nearest_idx
    for i, q in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        j = int(np.argmax(sims))
        out[i] = float(sims[j])
        nearest_idx[i] = j
    return out, nearest_idx


def tier(row: pd.Series) -> str:
    if row["exact_in_train"] or row["exact_in_chembl"] or row["exact_in_molecules_csv"]:
        return "exact_known"
    mx = float(row["max_tc_any_ref"])
    if mx >= 0.85:
        return "near_duplicate"
    if mx >= 0.70:
        return "close_analog"
    if mx >= 0.40:
        return "moderate_similarity"
    return "highly_novel"


def pubchem_cid(smiles: str, timeout: float = 8.0) -> int | None:
    """Exact structure identity search in PubChem (optional network)."""
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            + urllib.parse.quote(smiles, safe="")
            + "/cids/JSON"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "reinvent-novelty-check/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cids = data.get("IdentifierList", {}).get("CID", [])
        # PubChem returns CID 0 for unknown / non-matching structures
        if not cids:
            return None
        cid = int(cids[0])
        return cid if cid > 0 else None
    except Exception:
        return None


def main() -> None:
    pass_df = pd.read_csv(PASS).copy()
    pass_df["canon"] = pass_df["SMILES"].map(canon)
    q_fps = []
    for c in pass_df["canon"]:
        m = Chem.MolFromSmiles(c)
        q_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))

    train = []
    with open(CURATED / "data" / "refs" / "train.smi", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s:
                train.append(s)
    chembl = []
    chembl_path = CURATED / "data" / "refs" / "chembl_drugs.smi"
    if chembl_path.is_file():
        with open(chembl_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                if s:
                    chembl.append(s)
    mol_smiles = []
    mols_csv = CURATED / "data" / "molecules.csv"
    if mols_csv.is_file():
        md = pd.read_csv(mols_csv)
        col = "smiles" if "smiles" in md.columns else "SMILES"
        mol_smiles = md[col].dropna().astype(str).tolist()

    print(f"[INFO] refs: train={len(train)} chembl={len(chembl)} molecules.csv={len(mol_smiles)}")
    train_s, train_fp = fps_from_smiles(train)
    chem_s, chem_fp = fps_from_smiles(chembl)
    mol_s, mol_fp = fps_from_smiles(mol_smiles)
    train_set, chem_set, mol_set = set(train_s), set(chem_s), set(mol_s)

    tc_train, ni_train = max_tc(q_fps, train_fp)
    tc_chem, ni_chem = max_tc(q_fps, chem_fp)
    tc_mol, ni_mol = max_tc(q_fps, mol_fp)

    rows = []
    for i, r in pass_df.reset_index(drop=True).iterrows():
        c = r["canon"]
        rows.append(
            {
                "SMILES": r["SMILES"],
                "canon": c,
                "score": r["score"],
                "sa": r["sa"],
                "scscore": r["scscore"],
                "lambda_nm": r["lambda_nm"],
                "exact_in_train": c in train_set,
                "exact_in_chembl": c in chem_set,
                "exact_in_molecules_csv": c in mol_set,
                "max_tc_train": tc_train[i],
                "nearest_train": train_s[ni_train[i]] if ni_train[i] >= 0 else None,
                "max_tc_chembl": tc_chem[i] if len(chem_fp) else np.nan,
                "nearest_chembl": chem_s[ni_chem[i]] if len(chem_fp) and ni_chem[i] >= 0 else None,
                "max_tc_molecules_csv": tc_mol[i] if len(mol_fp) else np.nan,
                "nearest_molecules_csv": mol_s[ni_mol[i]] if len(mol_fp) and ni_mol[i] >= 0 else None,
            }
        )
    res = pd.DataFrame(rows)
    res["max_tc_any_ref"] = res[["max_tc_train", "max_tc_chembl", "max_tc_molecules_csv"]].max(axis=1)
    res["novelty_tier"] = res.apply(tier, axis=1)

    print("[INFO] PubChem identity lookup…")
    cids = []
    for i, smi in enumerate(res["canon"]):
        cid = pubchem_cid(smi)
        cids.append(cid)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(res)}")
    res["pubchem_cid"] = cids
    res["exact_in_pubchem"] = res["pubchem_cid"].notna()
    # Local-ref novelty is primary; PubChem is an extra knownness flag
    res["novelty_tier_local"] = res["novelty_tier"]

    def tier_combined(row: pd.Series) -> str:
        t = row["novelty_tier_local"]
        if row.get("exact_in_pubchem") and t != "exact_known":
            return "exact_known_pubchem"
        return t

    res["novelty_tier"] = res.apply(tier_combined, axis=1)
    res = res.sort_values(["novelty_tier_local", "max_tc_any_ref", "scscore"])

    out_tab = OUT / "tables"
    res.to_csv(out_tab / "scscore_pass_novelty.csv", index=False)

    summary = {
        "n": int(len(res)),
        "exact_in_train": int(res["exact_in_train"].sum()),
        "exact_in_chembl": int(res["exact_in_chembl"].sum()),
        "exact_in_molecules_csv": int(res["exact_in_molecules_csv"].sum()),
        "exact_in_pubchem": int(res["exact_in_pubchem"].sum()),
        "tier_counts_local": {k: int(v) for k, v in res["novelty_tier_local"].value_counts().items()},
        "tier_counts_combined": {k: int(v) for k, v in res["novelty_tier"].value_counts().items()},
        "mean_max_tc_train": float(res["max_tc_train"].mean()),
        "mean_max_tc_any_ref": float(res["max_tc_any_ref"].mean()),
        "definitions": {
            "exact_known": "canonical SMILES match in train / ChEMBL / molecules.csv",
            "exact_known_pubchem": "exact structure in PubChem (CID>0), not in local refs",
            "near_duplicate": "max Tc to any local ref ≥ 0.85",
            "close_analog": "0.70 ≤ max Tc < 0.85",
            "moderate_similarity": "0.40 ≤ max Tc < 0.70",
            "highly_novel": "max Tc to local refs < 0.40",
        },
    }
    (out_tab / "scscore_pass_novelty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = [
        "highly_novel",
        "moderate_similarity",
        "close_analog",
        "near_duplicate",
        "exact_known",
    ]
    counts = res["novelty_tier_local"].value_counts()
    labels = [o for o in order if o in counts.index]
    vals = [int(counts[o]) for o in labels]
    colors = {
        "highly_novel": "#00c853",
        "moderate_similarity": "#2962ff",
        "close_analog": "#ff6d00",
        "near_duplicate": "#c51162",
        "exact_known": "#212121",
    }
    axes[0].barh(labels[::-1], vals[::-1], color=[colors[l] for l in labels[::-1]], edgecolor="k", lw=0.3)
    axes[0].set_xlabel("count")
    n_pub = int(res["exact_in_pubchem"].sum())
    axes[0].set_title(f"Local novelty (n=25); PubChem exact={n_pub}")
    for y, v in enumerate(vals[::-1]):
        axes[0].text(v + 0.1, y, str(v), va="center", fontsize=9)

    sc = axes[1].scatter(
        res["max_tc_any_ref"],
        res["score"],
        c=res["scscore"],
        cmap="viridis_r",
        s=55,
        edgecolors="k",
        linewidths=0.4,
    )
    axes[1].axvline(0.4, color="#00c853", ls="--", lw=1, label="Tc=0.4")
    axes[1].axvline(0.7, color="#ff6d00", ls="--", lw=1, label="Tc=0.7")
    axes[1].axvline(0.85, color="#c51162", ls="--", lw=1, label="Tc=0.85")
    fig.colorbar(sc, ax=axes[1], label="SCScore")
    axes[1].set_xlabel("Max Tanimoto to any local ref")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Novelty vs Score")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "06_scscore_pass_novelty.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # structure grid colored by tier in legend text
    mols, legs = [], []
    for _, r in res.sort_values("max_tc_any_ref").iterrows():
        m = Chem.MolFromSmiles(r["canon"])
        if m is None:
            continue
        mols.append(m)
        pub = f"CID {int(r['pubchem_cid'])}" if pd.notna(r["pubchem_cid"]) else "no PubChem"
        legs.append(
            f"{r['novelty_tier_local']}\nTc={r['max_tc_any_ref']:.2f}  {pub}\n"
            f"S={r['score']:.2f} SC={r['scscore']:.2f}"
        )
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280, 250), legends=legs)
        img.save(str(OUT / "07_scscore_pass_novelty_structures.png"))
        print("[OK] 07_scscore_pass_novelty_structures.png")

    # highly novel only (Tc<0.40 local, no PubChem preferred in legend)
    hn = res[res["novelty_tier_local"] == "highly_novel"].sort_values("max_tc_any_ref").reset_index(drop=True)
    hn_dir = OUT / "highly_novel_mols"
    hn_dir.mkdir(parents=True, exist_ok=True)
    for old in hn_dir.glob("novel_*.png"):
        old.unlink()
    if not hn.empty:
        mols_hn, legs_hn = [], []
        for i, r in hn.iterrows():
            m = Chem.MolFromSmiles(r["canon"])
            if m is None:
                continue
            AllChem.Compute2DCoords(m)
            mols_hn.append(m)
            pub = f"CID {int(r['pubchem_cid'])}" if pd.notna(r["pubchem_cid"]) else "no PubChem"
            legs_hn.append(
                f"#{i+1}  Tc={r['max_tc_any_ref']:.2f}  {pub}\n"
                f"Score={r['score']:.3f}  SA={r['sa']:.2f}  SC={r['scscore']:.2f}"
            )
            Draw.MolToFile(m, str(hn_dir / f"novel_{i+1:02d}.png"), size=(400, 320))
        if mols_hn:
            img = Draw.MolsToGridImage(mols_hn, molsPerRow=2, subImgSize=(360, 320), legends=legs_hn)
            img.save(str(OUT / "08_highly_novel_structures.png"))
            print(f"[OK] 08_highly_novel_structures.png (n={len(mols_hn)})")
    else:
        print("[WARN] no highly_novel molecules for 08_*.png")

    print(json.dumps(summary, indent=2))
    cols = [
        "novelty_tier_local",
        "max_tc_any_ref",
        "exact_in_pubchem",
        "pubchem_cid",
        "score",
        "scscore",
    ]
    print(res[cols].to_string(index=False))
    print(f"[DONE] → {OUT}")


if __name__ == "__main__":
    main()
