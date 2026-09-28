"""Return forecasts: a historical-mean baseline and one walk-forward Random Forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import FEATURES


def historical_mean_forecast(daily_returns, window=252, horizon=21):
    """Baseline: trailing mean daily return scaled to a `horizon`-day forecast."""
    return daily_returns.iloc[-window:].mean() * horizon


class WalkForwardRF:
    """Pooled (all assets) Random Forest retrained on an expanding window.

    At rebalance date t the model is fit only on panel rows dated strictly before
    t. A row dated s has target P(next month-end after s)/P(s) - 1, whose end date
    is <= t, so every training label was fully realized by t. The configuration is
    fixed in advance and deliberately not tuned.
    """

    def __init__(self, n_estimators=100, max_depth=4, min_samples_leaf=50,
                 retrain_every=12, random_state=42):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, max_features=0.5,
                           n_jobs=-1, random_state=random_state)
        self.retrain_every = retrain_every
        self.model = None
        self._since_fit = 0
        self.importances = []  # one vector per fit
        self.fit_log = []      # (rebalance date, last training date, n rows)

    def predict(self, panel, t):
        """Forecast next-month return for each asset at date t (fit first if due)."""
        if self.model is None or self._since_fit >= self.retrain_every:
            dates = panel.index.get_level_values("date")
            train = panel.loc[dates < t].dropna(subset=["target"])
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(train[FEATURES].values, train["target"].values)
            self.importances.append(self.model.feature_importances_)
            self.fit_log.append((t, train.index.get_level_values("date").max(), len(train)))
            self._since_fit = 0
        self._since_fit += 1
        x = panel.xs(t, level="date")[FEATURES]
        return pd.Series(self.model.predict(x.values), index=x.index)

    def mean_importance(self):
        return pd.Series(np.mean(self.importances, axis=0), index=FEATURES).sort_values()
