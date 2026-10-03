import argparse
from pathlib import Path

import pandas as pd

from src import config
from src import pipeline as pipe


OUTPUT_DIR = Path("results")
MODELS_FILE = OUTPUT_DIR / "purged_chronological_54_models.csv"
AUDIT_FILE = OUTPUT_DIR / "purged_chronological_54_splits.csv"

X = 7
Y = 7


def save_csv(rows, path):
    """Save accumulated rows after every completed category."""
    pd.DataFrame(rows).to_csv(path, index=False)


def main(limit):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(
        config.DAILY_CLEAN,
        parse_dates=["Date"],
    )

    daily = daily.loc[
        ~daily["Category"].isin(config.EXCLUDED_CATEGORIES)
    ].copy()

    categories = sorted(daily["Category"].dropna().unique())
    print(f"Categories disponibles : {len(categories)}")

    # Resume from categories whose complete split audit was saved.
    if AUDIT_FILE.exists():
        audit_rows = pd.read_csv(AUDIT_FILE).to_dict("records")
    else:
        audit_rows = []

    if MODELS_FILE.exists():
        model_rows = pd.read_csv(MODELS_FILE).to_dict("records")
    else:
        model_rows = []

    completed = {
        row["Category"]
        for row in audit_rows
    }

    remaining = [
        category
        for category in categories
        if category not in completed
    ]

    if limit is not None:
        remaining = remaining[:limit]

    print(f"Categories a executer : {len(remaining)}")

    for i, category in enumerate(remaining, start=1):
        print(f"\n[{i}/{len(remaining)}] {category}")

        result = pipe.run_pipeline(
            daily,
            category,
            X=X,
            Y=Y,
            split_strategy="purged_chronological",
            verbose=False,
        )

        if result is None:
            raise RuntimeError(
                f"Pipeline sans resultat pour {category!r}"
            )

        overlap = result["split_overlap_report"]

        if not (overlap["Overlapping rows (%)"] == 0).all():
            raise RuntimeError(
                f"Chevauchement detecte pour {category!r}"
            )

        splits = result["splits"]
        audit = {
            "Category": category,
            "X": X,
            "Y": Y,
            "split_strategy": result["split_strategy"],
            "best_model_on_validation": result["best_model"],
            "maximum_target_overlap_pct": (
                overlap["Overlapping rows (%)"].max()
            ),
        }

        for name in ("train", "validation", "test"):
            part = splits[name]

            if part.empty:
                raise RuntimeError(
                    f"Partition {name} vide pour {category!r}"
                )

            first = pd.to_datetime(part["Date"].min())
            last = pd.to_datetime(part["Date"].max())

            audit[f"{name}_rows"] = len(part)
            audit[f"{name}_first_reference_date"] = first.date()
            audit[f"{name}_last_reference_date"] = last.date()
            audit[f"{name}_last_target_date"] = (
                last + pd.Timedelta(days=Y + X - 1)
            ).date()

        results = result["results"].copy()
        results["split_strategy"] = "purged_chronological"
        results["selected_on_validation"] = (
            results["Model"] == result["best_model"]
        )

        # Avoid duplicate rows if a previous run was interrupted
        # between saving the models file and the audit file.
        model_rows = [
            row
            for row in model_rows
            if row["Category"] != category
        ]
        model_rows.extend(results.to_dict("records"))
        save_csv(model_rows, MODELS_FILE)

        audit_rows.append(audit)
        save_csv(audit_rows, AUDIT_FILE)

        selected = results.loc[
            results["selected_on_validation"]
        ].iloc[0]

        print(
            f"Selection : {result['best_model']} | "
            f"Validation WAPE : "
            f"{selected['Validation WAPE (%)']:.2f}% | "
            f"Test WAPE : "
            f"{selected['Test WAPE (%)']:.2f}%"
        )

    print("\nExecution terminee.")
    print(f"Metriques : {MODELS_FILE}")
    print(f"Partitions : {AUDIT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Executer toutes les categories restantes.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Nombre de categories a executer (defaut : 3).",
    )

    args = parser.parse_args()

    main(limit=None if args.all else args.limit)