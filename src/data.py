"""Market data: download once with yfinance, then serve from a local CSV cache."""
import os

import numpy as np
import pandas as pd


def clean_prices(prices, max_ffill=5):
    """Align assets on a common calendar without using future information.

    Short gaps (holidays on one exchange, a missing print) are forward-filled
    for at most `max_ffill` days; forward fill only uses past prices, so it
    cannot leak the future. Any date still missing an asset is dropped, so every
    row is a date on which all assets have a usable price.
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.ffill(limit=max_ffill).dropna(how="any")
    return prices.astype(float)


def load_prices(tickers, start, end, path="data/prices.csv", refresh=False):
    """Return adjusted close prices, using the cache unless `refresh` or it is missing."""
    if os.path.exists(path) and not refresh:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            print(f"Loaded cached prices from {path}")
            return clean_prices(cached[tickers])
        print("Cache is missing some tickers; downloading again.")

    import yfinance as yf  # imported lazily so cached runs need no network

    print(f"Downloading {len(tickers)} tickers from yfinance ({start} to {end})")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers]
    if prices.dropna(how="all").empty:
        raise RuntimeError("yfinance returned no data")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prices.to_csv(path, float_format="%.6f")
    print(f"Cached prices to {path}")
    return clean_prices(prices)


def synthetic_prices(tickers, start, end, seed=0):
    """Offline smoke-test data: one-factor correlated random walks (NOT market data).

    Used only with `python run.py --synthetic` to exercise the pipeline when
    yfinance is unreachable. It is never written to the price cache.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, inclusive="left")
    n, k = len(dates), len(tickers)
    beta = rng.uniform(0.3, 1.3, k)
    vol = rng.uniform(0.15, 0.35, k) / np.sqrt(252)
    market = rng.normal(0.0003, 0.011, n)
    idio = rng.normal(0, 1, (n, k)) * vol
    rets = 0.0001 + market[:, None] * beta + idio
    return pd.DataFrame(100 * np.exp(np.cumsum(np.log1p(rets), axis=0)),
                        index=dates, columns=tickers)


def month_end_dates(index):
    """Last trading day of each calendar month in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)
