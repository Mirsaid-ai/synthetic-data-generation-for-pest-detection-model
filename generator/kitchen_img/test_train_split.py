"""Create a deterministic 75/25 kitchen train/test split CSV."""

import csv
import random
from pathlib import Path


SEED = 42
TRAIN_FRAC = 0.75
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
CURATED_DIR = SCRIPT_DIR / "curated_img"
CSV_PATH = SCRIPT_DIR / "test_train_split.csv"


def curated_filenames():
    return sorted(
        p.name
        for p in CURATED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def build_split_rows():
    kitchen_ids = curated_filenames()
    rng = random.Random(SEED)
    shuffled = kitchen_ids[:]
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * TRAIN_FRAC)
    train_ids = set(shuffled[:train_count])
    return [
        {"id": kitchen_id, "train": 1 if kitchen_id in train_ids else 0}
        for kitchen_id in kitchen_ids
    ]


def write_csv(rows):
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "train"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_split_rows()
    write_csv(rows)
    train_count = sum(row["train"] for row in rows)
    test_count = len(rows) - train_count
    print(f"Wrote {CSV_PATH}")
    print(f"Train: {train_count}")
    print(f"Test: {test_count}")


if __name__ == "__main__":
    main()
