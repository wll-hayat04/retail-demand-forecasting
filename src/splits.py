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
