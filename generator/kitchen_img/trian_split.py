"""Create a deterministic 70/15/15 kitchen train/val/test split CSV."""

import csv
import random
from pathlib import Path


SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
CURATED_DIR = SCRIPT_DIR / "curated_img"
CSV_PATH = SCRIPT_DIR / "tain_split.csv"


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

    total = len(shuffled)
    train_count = int(total * TRAIN_FRAC)
    val_count = int(total * VAL_FRAC)

    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count : train_count + val_count])
    test_ids = set(shuffled[train_count + val_count :])

    return [
        {
            "id": kitchen_id,
            "train": 1 if kitchen_id in train_ids else 0,
            "val": 1 if kitchen_id in val_ids else 0,
            "test": 1 if kitchen_id in test_ids else 0,
        }
        for kitchen_id in kitchen_ids
    ]


def write_csv(rows):
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "train", "val", "test"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_split_rows()
    write_csv(rows)

    train_count = sum(row["train"] for row in rows)
    val_count = sum(row["val"] for row in rows)
    test_count = sum(row["test"] for row in rows)

    print(f"Wrote {CSV_PATH}")
    print(f"Train: {train_count}")
    print(f"Val: {val_count}")
    print(f"Test: {test_count}")


if __name__ == "__main__":
    main()
