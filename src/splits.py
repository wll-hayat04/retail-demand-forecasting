"""Train / validation / test splitting strategies.

The block-shuffled split is the chosen methodology; see the split-strategy
notebook for the comparison against random, chronological, and TimeSeriesCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def block_shuffled_split(
    data: pd.DataFrame,
    block_size: int = config.BLOCK_SIZE,
    train_size: float = config.TRAIN_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Split into contiguous blocks of ``block_size`` days, then shuffle blocks.

    Compromise between a random split (severe leakage — adjacent rows share
    overlapping target windows) and a chronological split (no leakage, but
    the 1-year dataset means the test period is a different season from the
    training period, so models fail on distribution shift).

    Returns a dict keyed "train" / "validation" / "test" so downstream code
    is explicit about which split it is using.
    """
    data = data.sort_values("Date").reset_index(drop=True)

    n_blocks = int(np.ceil(len(data) / block_size))
    block_ids = np.repeat(np.arange(n_blocks), block_size)[: len(data)]

    data = data.copy()
    data["block_id"] = block_ids

    unique_blocks = data["block_id"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_blocks)

    n_train = int(len(unique_blocks) * train_size)
    n_val = int(len(unique_blocks) * val_size)

    assignment = {
        "train": unique_blocks[:n_train],
        "validation": unique_blocks[n_train : n_train + n_val],
        "test": unique_blocks[n_train + n_val :],
    }

    return {
        name: (
            data[data["block_id"].isin(blocks)]
            .drop(columns="block_id")
            .reset_index(drop=True)
        )
        for name, blocks in assignment.items()
    }


def leakage_ratio(splits: dict[str, pd.DataFrame], X: int) -> float:
    """Fraction of test rows with a training row within X-1 days.

    Two rows leak into each other when their target windows overlap, which
    happens when they are fewer than X days apart. This quantifies the
    residual leakage the block-shuffled split accepts, so the number in the
    report is measured rather than asserted.
    """
    train_dates = splits["train"]["Date"].to_numpy()
    test_dates = splits["test"]["Date"].to_numpy()

    if len(train_dates) == 0 or len(test_dates) == 0:
        return np.nan

    window = np.timedelta64(X - 1, "D")
    leaked = sum(
        np.any(np.abs(train_dates - d) <= window) for d in test_dates
    )
    return leaked / len(test_dates)


def stratified_block_split(
    data,
    target_col,
    block_size=config.BLOCK_SIZE,
    train_size=config.TRAIN_SIZE,
    val_size=config.VAL_SIZE,
    random_state=config.RANDOM_STATE,
    shuffle_window=3,
):
    """Block split stratified by block-level demand.

    Plain block shuffling leaves the seasonal composition of each split to
    chance. With only ~24 blocks in a one-year series, some draws put the
    whole Christmas ramp in one split and none of it in another, which makes
    the resulting scores incomparable across seeds.

    This reduces the VARIANCE of the estimate, not the difficulty of the
    problem. One year of data still contains exactly one Christmas.
    """
    data = data.sort_values("Date").reset_index(drop=True)

    n_blocks = int(np.ceil(len(data) / block_size))
    data = data.copy()
    data["block_id"] = np.repeat(np.arange(n_blocks), block_size)[: len(data)]

    rng = np.random.default_rng(random_state)

    # Order blocks by mean demand, then shuffle within short windows. The
    # window matters: without it the assignment is fully determined by demand
    # order, every seed returns the same split, and there is no way to
    # estimate uncertainty at all.
    ordered = (data.groupby("block_id")[target_col].mean()
               .sort_values().index.to_numpy().copy())
    for i in range(0, len(ordered), shuffle_window):
        window = ordered[i : i + shuffle_window].copy()
        rng.shuffle(window)
        ordered[i : i + shuffle_window] = window

    n = len(ordered)
    n_val = max(int(round(n * val_size)), 1)
    n_test = max(n - int(round(n * train_size)) - n_val, 1)
    n_train = n - n_val - n_test

    # Largest-remainder interleaving: each block goes to whichever split is
    # furthest behind its quota, so all three spread across the demand range
    # instead of clustering at one end.
    quotas = {"train": n_train, "validation": n_val, "test": n_test}
    assigned = {k: [] for k in quotas}

    for j, block in enumerate(ordered, start=1):
        deficits = {
            k: quotas[k] * j / n - len(assigned[k])
            for k in quotas
            if len(assigned[k]) < quotas[k]
        }
        assigned[max(deficits, key=deficits.get)].append(block)

    return {
        name: (data[data["block_id"].isin(blocks)]
               .drop(columns="block_id")
               .reset_index(drop=True))
        for name, blocks in assigned.items()
    }
def target_window_overlap_ratio(
    left: pd.DataFrame,
    right: pd.DataFrame,
    X: int,
) -> float:
    """Fraction of rows in `left` whose target window overlaps a
    target window belonging to `right`.

    The target for a reference date t covers X consecutive days,
    starting Y days after t. When Y is identical across the two
    datasets, two target windows overlap if their reference dates
    differ by fewer than X days.

    The ratio is directional: overlap(left, right) is not necessarily
    equal to overlap(right, left).
    """
    if not isinstance(X, (int, np.integer)) or X < 1:
        raise ValueError("X must be a positive integer.")

    if left.empty:
        return np.nan

    if right.empty:
        return 0.0

    left_dates = pd.to_datetime(left["Date"]).to_numpy(
        dtype="datetime64[D]"
    )
    right_dates = np.sort(
        pd.to_datetime(right["Date"]).to_numpy(dtype="datetime64[D]")
    )

    # Find the closest reference date in `right` for each date
    # in `left`, without creating an expensive pairwise matrix.
    positions = np.searchsorted(right_dates, left_dates)

    overlaps = np.zeros(len(left_dates), dtype=bool)
    threshold = np.timedelta64(X, "D")

    has_next = positions < len(right_dates)
    next_indices = positions[has_next]
    overlaps[has_next] |= (
        np.abs(right_dates[next_indices] - left_dates[has_next])
        < threshold
    )

    has_previous = positions > 0
    previous_indices = positions[has_previous] - 1
    overlaps[has_previous] |= (
        np.abs(
            right_dates[previous_indices]
            - left_dates[has_previous]
        )
        < threshold
    )

    return float(overlaps.mean())


def split_overlap_report(
    splits: dict[str, pd.DataFrame],
    X: int,
) -> pd.DataFrame:
    """Report target-window overlap between all split pairs.

    Each result gives the proportion of rows in the first split
    overlapping at least one target window in the second split.
    """
    pairs = [
        ("train", "validation"),
        ("validation", "train"),
        ("train", "test"),
        ("test", "train"),
        ("validation", "test"),
        ("test", "validation"),
    ]

    rows = []

    for left_name, right_name in pairs:
        left = splits[left_name]
        right = splits[right_name]

        rows.append({
            "Split": left_name,
            "Compared with": right_name,
            "Rows": len(left),
            "Overlapping rows (%)": (
                100 * target_window_overlap_ratio(left, right, X)
            ),
        })

    return pd.DataFrame(rows)

def purged_chronological_split(
    data: pd.DataFrame,
    X: int,
    Y: int,
    train_size: float = config.TRAIN_SIZE,
    val_size: float = config.VAL_SIZE,
) -> dict[str, pd.DataFrame]:
    """Chronological train/validation/test split with label-availability purge.

    For a reference date t, the target covers dates
    [t + Y, t + Y + X - 1], inclusive.

    Training rows are retained only if their target is fully observable
    before the first validation reference date.

    Validation rows are retained only if their target is fully observable
    before the first test reference date.

    The three periods are defined before purging. Purged rows are not
    reassigned to another split.
    """
    if not isinstance(X, (int, np.integer)) or X < 1:
        raise ValueError("X must be a positive integer.")

    if not isinstance(Y, (int, np.integer)) or Y < 1:
        raise ValueError("Y must be a positive integer.")

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if train_size + val_size >= 1:
        raise ValueError(
            "train_size + val_size must be strictly less than 1."
        )

    if "Date" not in data.columns:
        raise KeyError("The data must contain a 'Date' column.")

    ordered = data.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"])

    if ordered["Date"].isna().any():
        raise ValueError("The Date column contains missing values.")

    if ordered["Date"].duplicated().any():
        raise ValueError(
            "Reference dates must be unique within one category."
        )

    ordered = ordered.sort_values("Date").reset_index(drop=True)

    n = len(ordered)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise ValueError(
            "Not enough rows for non-empty train, validation "
            "and test periods."
        )

    # Define the chronological evaluation boundaries before purging.
    validation_start = ordered.loc[n_train, "Date"]
    test_start = ordered.loc[n_train + n_val, "Date"]

    train = ordered.iloc[:n_train].copy()
    validation = ordered.iloc[n_train:n_train + n_val].copy()
    test = ordered.iloc[n_train + n_val:].copy()

    # A target for reference date t becomes fully observable at
    # t + Y + X - 1. It must be observable before the next period starts.
    target_end_offset = pd.Timedelta(days=Y + X - 1)

    train = train.loc[
        train["Date"] + target_end_offset < validation_start
    ].copy()

    validation = validation.loc[
        validation["Date"] + target_end_offset < test_start
    ].copy()

    result = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    if any(part.empty for part in result.values()):
        raise ValueError(
            "At least one split is empty after purging. "
            "Use more observations, shorter horizons or "
            "different split proportions."
        )

    return result