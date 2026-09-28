"""Feature engineering. Every feature at date t uses prices up to and including t only."""
import numpy as np
import pandas as pd

FEATURES = ["ret_5d", "ret_20d", "ret_60d", "vol_20d", "px_ma20", "rsi_14"]


def rsi(prices, window=14):
    """Simple (rolling-mean) Relative Strength Index, rescaled to [-0.5, 0.5]."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)) / 100 - 0.5


def compute_features(prices):
    """Daily wide feature frames {name: DataFrame(date x asset)}.

    All operations are backward-looking (pct_change, rolling windows ending at t),
    so the value at t never depends on prices after t.
    """
    daily_ret = prices.pct_change()
    return {
        "ret_5d": prices.pct_change(5),
        "ret_20d": prices.pct_change(20),
        "ret_60d": prices.pct_change(60),
        "vol_20d": daily_ret.rolling(20).std() * np.sqrt(252),
        "px_ma20": prices / prices.rolling(20).mean() - 1,
        "rsi_14": rsi(prices, 14),  # simple momentum / overbought-oversold indicator
    }


def _to_long(wide, name):
    long = wide.rename_axis(index="date", columns="asset").reset_index()
    long = long.melt(id_vars="date", var_name="asset", value_name=name)
    return long.set_index(["date", "asset"])[name]


def build_panel(prices, rebalance_dates):
    """Monthly (date, asset) panel of features plus the next-month return target.

    target at date t = P(next month-end) / P(t) - 1. It is only *known* at the
    next rebalance date, which the model's training filter respects.
    """
    feats = compute_features(prices)
    cols = [_to_long(feats[f].loc[rebalance_dates], f) for f in FEATURES]
    monthly = prices.loc[rebalance_dates]
    target = monthly.shift(-1) / monthly - 1
    cols.append(_to_long(target, "target"))
    panel = pd.concat(cols, axis=1).sort_index()
    return panel.dropna(subset=FEATURES)
