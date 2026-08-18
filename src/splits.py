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
