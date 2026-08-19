"""Model registry.

Every model is a function with the same signature:

    fit_predict(train, val, test, features, target) -> (train_pred, val_pred, test_pred)

Adding a model means writing one function and decorating it. Nothing in the
training loop, the notebooks, or the pipeline changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from . import config

MODELS: dict = {}


def register(name: str):
    def decorator(fn):
        MODELS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

@register("7-Day Rolling Sum Baseline")
def _baseline(train, val, test, features, target):
    """Predict the next X days as the sum of the previous 7.

    Uses no features and requires no fitting — the reference every model
    has to beat to justify its complexity.
    """
    return (
        train["rolling_sum_7"].to_numpy(),
        val["rolling_sum_7"].to_numpy(),
        test["rolling_sum_7"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

def _sklearn_fit_predict(model, train, val, test, features, target):
    model.fit(train[features], train[target])
    return (
        model.predict(train[features]),
        model.predict(val[features]),
        model.predict(test[features]),
    )


@register("Linear Regression")
def _linreg(train, val, test, features, target):
    return _sklearn_fit_predict(
        LinearRegression(), train, val, test, features, target
    )


@register("Random Forest")
def _rf(train, val, test, features, target):
    return _sklearn_fit_predict(
        RandomForestRegressor(
            n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        train, val, test, features, target,
    )


@register("XGBoost")
def _xgb(train, val, test, features, target):
    return _sklearn_fit_predict(
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=config.RANDOM_STATE,
            verbosity=0,
        ),
        train, val, test, features, target,
    )


# ---------------------------------------------------------------------------
# TabFM — zero-shot tabular foundation model (Google Research)
#
# Registered only if the library imports and the weights load, so the rest
# of the project runs unchanged on a machine without it.
# ---------------------------------------------------------------------------

TABFM_MODEL = None
TABFM_AVAILABLE = False


def enable_tabfm(backend: str = "pytorch", max_context_rows=None,
                 n_estimators: int = 32) -> bool:
    """Load TabFM weights once and register it as a model.

    Call this explicitly from a notebook — loading downloads weights and is
    slow, so it should not happen on import.
    """
    global TABFM_MODEL, TABFM_AVAILABLE

    try:
        from tabfm import TabFMRegressor

        if backend == "pytorch":
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1
        else:
            from tabfm import tabfm_v1_0_0_jax as tabfm_v1

        import os as _os, torch as _torch
        _torch.set_num_threads(_os.cpu_count())
        # float32: bfloat16 is designed for GPU and is emulated (slowly) on
        # CPUs without native support. Measured ~2x faster here.
        # bfloat16 is native on GPU; on CPU it is emulated and slow, but
        # float32 doubles memory and can exhaust the Windows paging file.
        _device = "cuda" if _torch.cuda.is_available() else None
        print(f"TabFM device: {_device or 'cpu'}")
        TABFM_MODEL = tabfm_v1.load(model_type="regression", device=_device)
    except Exception as exc:
        print(f"TabFM unavailable ({type(exc).__name__}: {exc})")
        return False

    def _prep(df, features):
        Xd = df[features].copy()
        for c in config.CATEGORICAL_FEATURES:
            if c in Xd.columns:
                Xd[c] = Xd[c].astype(str)
        return Xd

    @register("TabFM")
    def _tabfm(train, val, test, features, target):
        # TabFM does not train weights: .fit() stores the training rows as
        # context and .predict() runs one forward pass. The whole training
        # set must therefore fit inside the context window — the library
        # default of 100 rows would silently sample away most of it.
        reg = TabFMRegressor(
            model=TABFM_MODEL,
            max_num_rows=max_context_rows,
            n_estimators=n_estimators,
            random_state=config.RANDOM_STATE,
        )
        reg.fit(_prep(train, features), train[target])
        # Train predictions skipped: TabFM predicts rows sitting inside its
        # own context, so a train score measures recall, not fit. It was also
        # ~70% of the inference cost.
        import numpy as _np
        return (
            _np.full(len(train), _np.nan),
            reg.predict(_prep(val, features)),
            reg.predict(_prep(test, features)),
        )

    TABFM_AVAILABLE = True
    print("TabFM registered.")
    return True


# ---------------------------------------------------------------------------

def model_names() -> list[str]:
    return list(MODELS)


# Models whose train-set metrics are not comparable to the others.
# TabFM predicts training rows that are literally inside its context, so its
# Train WAPE measures recall, not fit. Judge it on validation and test only.
NO_MEANINGFUL_TRAIN_METRIC = {"7-Day Rolling Sum Baseline", "TabFM"}
