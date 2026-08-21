"""Raw data preparation: Excel -> cleaned daily category sales."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config

ADMIN_CODES = ["POST", "D", "M", "C2", "DOT", "BANK CHARGES", "CRUK", "S", "PADS"]


def match_cancellations(df, verbose=True):
    """Remove cancellation rows AND the orders they reverse.

    A "C" invoice carries a negative quantity reversing an earlier order.
    Dropping only the negative line leaves the original counted as a real sale.
    Two cases dominate this dataset:

        541431 / C541433   74,215 units  2011-01-18, cancelled after 16 min
        581483 / C581484   80,995 units  2011-12-09, cancelled after 12 min

    Left in, the second gave Children's Toys & Playsets a test R2 of -2211:
    the spike sits in training, inflates every lag and rolling feature around
    it, and the model then predicts tens of thousands against actuals of ~1400.
    """
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
    cancels = df[is_cancel]
    originals = df[~is_cancel]

    index = defaultdict(list)
    for idx, cust, stock, qty, date in zip(
        originals.index,
        originals["CustomerID"].to_numpy(),
        originals["StockCode"].astype(str).to_numpy(),
        originals["Quantity"].to_numpy(),
        originals["InvoiceDate"].to_numpy(),
    ):
        if pd.isna(cust):
            continue
        index[(cust, stock, qty)].append((date, idx))
    for key in index:
        index[key].sort()

    drop_idx = set(cancels.index)
    used = set()
    matched = 0

    for cust, stock, qty, cdate in zip(
        cancels["CustomerID"].to_numpy(),
        cancels["StockCode"].astype(str).to_numpy(),
        cancels["Quantity"].to_numpy(),
        cancels["InvoiceDate"].to_numpy(),
    ):
        if pd.isna(cust):
            continue
        candidates = index.get((cust, stock, -qty))
        if not candidates:
            continue
        for date, idx in reversed(candidates):
            if idx not in used and date <= cdate:
                used.add(idx)
                drop_idx.add(idx)
                matched += 1
                break

    if verbose:
        pct = matched / len(cancels) * 100 if len(cancels) else 0
        print(f"  cancellation rows           : {len(cancels)}")
        print(f"  matched to an original order: {matched} ({pct:.0f}%)")
        print(f"  rows removed in total       : {len(drop_idx)}")

    out = df.drop(index=list(drop_idx))
    if verbose:
        print(f"  max quantity after          : {out['Quantity'].max()}")
    return out


def clean_transactions(df, verbose=True):
    n0 = len(df)
    if verbose:
        print(f"raw rows: {n0}")
    df = match_cancellations(df, verbose=verbose)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    df = df[~df["StockCode"].astype(str).str.upper().isin(ADMIN_CODES)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    if verbose:
        print(f"rows after cleaning: {len(df)} ({len(df)/n0*100:.1f}% kept)")
    return df


def map_categories(df, mapping_path=None, verbose=True):
    mapping_path = mapping_path or config.CATEGORY_MAP
    with open(mapping_path, encoding="utf-8") as f:
        categories = json.load(f)
    df = df.copy()
    desc = df["Description"].astype(str).str.strip().str.lower()
    df["Category"] = desc.map(categories).fillna("Unclassified")
    if verbose:
        print(f"unclassified rows: {(df['Category']=='Unclassified').mean()*100:.2f}%")
        print(f"categories: {df['Category'].nunique()}")
    return df


def aggregate_daily(df, verbose=True):
    """One row per (date, category), with explicit zero days.

    The full grid matters: a category with no sales on a given day must appear
    as zero. Otherwise lag_7 stops meaning "one week ago".
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    daily = (
        df.groupby(["Date", "Category"])
        .agg(Quantity=("Quantity", "sum"),
             Revenue=("Revenue", "sum"),
             Orders=("InvoiceNo", "nunique"))
        .reset_index()
    )

    all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    all_cats = sorted(daily["Category"].unique())
    grid = pd.MultiIndex.from_product([all_dates, all_cats], names=["Date", "Category"])

    daily = (
        daily.set_index(["Date", "Category"])
        .reindex(grid, fill_value=0)
        .reset_index()
        .sort_values(["Category", "Date"])
        .reset_index(drop=True)
    )

    if verbose:
        print(f"daily rows: {len(daily)} ({len(all_dates)} days x {len(all_cats)} categories)")
        gaps = pd.Series(all_dates).diff().dropna()
        assert (gaps == pd.Timedelta(days=1)).all(), "date index is not contiguous"
    return daily


def build_daily_dataset(raw_path=None, out_path=None, verbose=True):
    raw_path = raw_path or config.RAW_EXCEL
    out_path = out_path or config.DAILY_CLEAN
    if verbose:
        print(f"reading {raw_path}")
    df = pd.read_excel(raw_path)
    df = clean_transactions(df, verbose=verbose)
    df = map_categories(df, verbose=verbose)
    daily = aggregate_daily(df, verbose=verbose)
    daily.to_csv(out_path, index=False)
    if verbose:
        print(f"written to {out_path}")
    return daily
