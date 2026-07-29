import csv
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


# Определяем Pydantic-модель для одной записи
class AbsorptionRecord(BaseModel):
    key: str
    smiles: str = Field(..., min_length=1, description="SMILES хромофора")
    solvent: str = Field(..., min_length=1, description="SMILES растворителя")
    peakwavs_max: Optional[float] = Field(None, description="Максимум поглощения (нм)")


def load_and_prepare_data(csv_path: Path, smiles_out: Path, solvents_out: Path):
    """
    Читает CSV, валидирует строки, сохраняет уникальные SMILES и растворители.
    """
    unique_chromophores = set()
    unique_solvents = set()
    total_rows = 0
    error_rows = 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Проверяем наличие нужных колонок
        required_fields = {'key', 'smiles', 'solvent', 'peakwavs_max'}
        if not required_fields.issubset(reader.fieldnames):
            missing = required_fields - set(reader.fieldnames)
            raise ValueError(f"CSV не содержит обязательные колонки: {missing}")

        for row_num, row in enumerate(reader, start=2):  # start=2, т.к. 1-я строка — заголовки
            total_rows += 1
            try:
                # Валидируем и приводим типы
                record = AbsorptionRecord(
                    key=row.get('key', '').strip(),
                    smiles=row.get('smiles', '').strip(),
                    solvent=row.get('solvent', '').strip(),
                    peakwavs_max=float(row['peakwavs_max']) if row.get('peakwavs_max') else None
                )
            except (ValidationError, ValueError) as e:
                print(f"Ошибка в строке {row_num}: {e}")
                error_rows += 1
                continue

            # Добавляем в множества (автоматически уникализируется)
            unique_chromophores.add(record.smiles)
            unique_solvents.add(record.solvent)

    # Сохраняем уникальные SMILES хромофоров
    with open(smiles_out, 'w', encoding='utf-8') as f:
        for smi in sorted(unique_chromophores):
            f.write(smi + '\n')

    # Сохраняем уникальные SMILES растворителей
    with open(solvents_out, 'w', encoding='utf-8') as f:
        for sol in sorted(unique_solvents):
            f.write(sol + '\n')

    print(f"Всего записей: {total_rows}")
    print(f"С ошибками: {error_rows}")
    print(f"Уникальных хромофоров: {len(unique_chromophores)} -> сохранены в {smiles_out}")
    print(f"Уникальных растворителей: {len(unique_solvents)} -> сохранены в {solvents_out}")


if __name__ == "__main__":
    # Укажите пути к файлам
    INPUT_CSV = Path(r"C:\\Users\\Evgen\\Desktop\\reinvent\\transfer_learning\\absorption_val.csv")
    OUTPUT_SMILES = Path("data/val.smi")
    OUTPUT_SOLVENTS = Path("data/solvents_val.smi")

    if not INPUT_CSV.exists():
        print(f"Файл {INPUT_CSV} не найден!")
    else:
        load_and_prepare_data(INPUT_CSV, OUTPUT_SMILES, OUTPUT_SOLVENTS)