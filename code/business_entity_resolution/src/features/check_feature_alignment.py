from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_FILE = PROJECT_ROOT / "data" / "features" / "training_pairs_hard_negative.tsv"

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "validation_pairs_features_sample.tsv"
)


def get_columns(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.reader(
            f,
            delimiter="\t",
        )

        return next(reader)


def main():

    print("=" * 70)
    print("FEATURE ALIGNMENT CHECK")
    print("=" * 70)

    print("\nReading training schema...")

    train_columns = get_columns(
        TRAIN_FILE
    )

    print(
        f"Training columns   : "
        f"{len(train_columns)}"
    )

    print("\nReading validation schema...")

    val_columns = get_columns(
        VAL_FILE
    )

    print(
        f"Validation columns : "
        f"{len(val_columns)}"
    )

    train_set = set(
        train_columns
    )

    val_set = set(
        val_columns
    )

    only_train = (
        train_set - val_set
    )

    only_val = (
        val_set - train_set
    )

    common = (
        train_set & val_set
    )

    print("\n" + "-" * 70)

    print(
        f"Common columns     : "
        f"{len(common)}"
    )

    print(
        f"Only training      : "
        f"{len(only_train)}"
    )

    print(
        f"Only validation    : "
        f"{len(only_val)}"
    )

    if only_train:

        print("\nOnly in training:")

        for column in sorted(
            only_train
        ):
            print(
                f"  - {column}"
            )

    if only_val:

        print("\nOnly in validation:")

        for column in sorted(
            only_val
        ):
            print(
                f"  - {column}"
            )

    print("\n" + "-" * 70)

    print("TRAINING COLUMNS")

    for i, column in enumerate(
        train_columns,
        start=1,
    ):
        print(
            f"{i:2d}. {column}"
        )

    print("\n" + "-" * 70)

    print("VALIDATION COLUMNS")

    for i, column in enumerate(
        val_columns,
        start=1,
    ):
        print(
            f"{i:2d}. {column}"
        )

    print("\n" + "=" * 70)

    if (
        train_set == val_set
    ):
        print(
            "PASS: Feature schemas match."
        )
    else:
        print(
            "WARNING: Feature schemas do NOT match."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()