"""Audit cancellation matching without modifying the source data."""

from collections import defaultdict

import pandas as pd

from src import config


def audit_matching(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the current matching rule and record each decision."""
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    cancellations = df.loc[is_cancel]
    originals = df.loc[~is_cancel]

    original_index = defaultdict(list)

    for idx, customer, stock, quantity, date in zip(
        originals.index,
        originals["CustomerID"].to_numpy(),
        originals["StockCode"].astype(str).to_numpy(),
        originals["Quantity"].to_numpy(),
        originals["InvoiceDate"].to_numpy(),
    ):
        if pd.isna(customer):
            continue

        original_index[(customer, stock, quantity)].append(
            (date, idx)
        )

    for candidates in original_index.values():
        candidates.sort()

    used_originals = set()
    audit_rows = []

    for cancel_idx, invoice, customer, stock, quantity, cancel_date in zip(
        cancellations.index,
        cancellations["InvoiceNo"].astype(str),
        cancellations["CustomerID"].to_numpy(),
        cancellations["StockCode"].astype(str).to_numpy(),
        cancellations["Quantity"].to_numpy(),
        cancellations["InvoiceDate"].to_numpy(),
    ):
        available = []

        if not pd.isna(customer):
            available = [
                (date, idx)
                for date, idx in original_index.get(
                    (customer, stock, -quantity), []
                )
                if idx not in used_originals and date <= cancel_date
            ]

        selected_idx = available[-1][1] if available else None

        if selected_idx is not None:
            used_originals.add(selected_idx)

        audit_rows.append({
            "cancellation_row": cancel_idx,
            "invoice": invoice,
            "available_candidates": len(available),
            "selected_original_row": selected_idx,
            "status": (
                "unmatched" if not available
                else "unique_available_candidate"
                if len(available) == 1
                else "multiple_available_candidates"
            ),
        })

    return pd.DataFrame(audit_rows)


def main() -> None:
    print(f"Reading: {config.RAW_EXCEL}")
    raw = pd.read_excel(config.RAW_EXCEL)

    audit = audit_matching(raw)

    print("\nMatching decisions:")
    print(audit["status"].value_counts().to_string())

    matched = audit["selected_original_row"].dropna()

    assert matched.is_unique, (
        "One original transaction was assigned more than once."
    )

    print(f"\nTotal matched cancellation rows: {len(matched)}")

    output_path = config.RESULTS / "cancellation_matching_audit.csv"
    audit.to_csv(output_path, index=False)

    print(f"Audit saved to: {output_path}")


if __name__ == "__main__":
    main()