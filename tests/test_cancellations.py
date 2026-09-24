import pandas as pd

from src.data import (
    cancellation_matching_diagnostic,
    match_cancellations,
)


def make_transactions():
    return pd.DataFrame({
        "InvoiceNo": [
            "100", "C101",     # Exact cancellation
            "200", "201", "C202",  # Two possible originals
            "C300",             # No matching original
        ],
        "CustomerID": [1, 1, 2, 2, 2, 3],
        "StockCode": ["A", "A", "B", "B", "B", "C"],
        "Quantity": [5, -5, 10, 10, -10, -7],
        "InvoiceDate": pd.to_datetime([
            "2026-01-01 10:00",
            "2026-01-01 10:15",
            "2026-01-02 09:00",
            "2026-01-02 10:00",
            "2026-01-02 11:00",
            "2026-01-03 12:00",
        ]),
    })


def test_exact_cancellation_removes_both_lines():
    df = make_transactions().iloc[:2].copy()

    cleaned = match_cancellations(df, verbose=False)

    assert cleaned.empty


def test_ambiguous_cancellation_selects_most_recent_original():
    df = make_transactions().iloc[2:5].copy()

    cleaned = match_cancellations(df, verbose=False)

    # The most recent original (invoice 201) is removed.
    assert cleaned["InvoiceNo"].tolist() == ["200"]


def test_unmatched_cancellation_is_removed():
    df = make_transactions().iloc[5:].copy()

    cleaned = match_cancellations(df, verbose=False)

    assert cleaned.empty


def test_matching_diagnostic_does_not_modify_input():
    df = make_transactions()
    original = df.copy(deep=True)

    report = cancellation_matching_diagnostic(df)

    pd.testing.assert_frame_equal(df, original)

    assert len(report) == 3

    statuses = report.set_index("invoice")["matching_status"]

    assert statuses["C101"] == "unique_candidate"
    assert statuses["C202"] == "multiple_candidates"
    assert statuses["C300"] == "unmatched"


def test_one_original_cannot_be_matched_twice():
    df = pd.DataFrame({
        "InvoiceNo": ["100", "C101", "C102"],
        "CustomerID": [1, 1, 1],
        "StockCode": ["A", "A", "A"],
        "Quantity": [5, -5, -5],
        "InvoiceDate": pd.to_datetime([
            "2026-01-01 10:00",
            "2026-01-01 10:15",
            "2026-01-01 10:30",
        ]),
    })

    cleaned = match_cancellations(df, verbose=False)

    # The original is removed once; both cancellation lines are removed.
    assert cleaned.empty