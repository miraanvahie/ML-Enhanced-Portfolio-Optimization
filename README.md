# ML-Enhanced Portfolio Optimization

An out-of-sample research pipeline that tests whether a machine-learning return
forecast, blended with market equilibrium through **Black-Litterman**, improves a
mean-variance portfolio **after transaction costs**. It is compared against simple,
hard-to-beat baselines.

```
historical prices (yfinance, cached) → features → Random Forest forecast
→ Black-Litterman posterior → covariance → long-only optimizer
→ turnover + transaction costs → monthly out-of-sample backtest
```

The whole pipeline runs in well under a minute on a laptop: 10 assets, monthly
rebalancing, one Random Forest configuration (100 shallow trees) refit once a year.

## Quick start

```bash
pip install -r requirements.txt
python run.py                 # downloads once, then reads data/prices.csv
python run.py --refresh       # force a new download
python run.py --cost-bps 25   # different proportional cost rate
python run.py --synthetic --results-dir /tmp/smoke   # offline smoke test (fake data)
```

`--synthetic` runs the pipeline on simulated random-walk prices. It only checks
that the code works when Yahoo Finance is unreachable. Those numbers are not
market results and should not be reported as such.

## Universe and data

| Group | Tickers |
|---|---|
| US equity ETFs | SPY, QQQ, IWM |
| International equity | EFA |
| Bonds / gold | TLT, GLD |
| Large-cap stocks | AAPL, MSFT, JPM, XOM |

Daily split- and dividend-adjusted closes from 2012 to mid-2026. The first run saves
them to `data/prices.csv`, and later runs read that file without touching the
network. Short gaps are forward-filled (past data only, at most 5 days), and any
date that still lacks a price is dropped so that all assets share one calendar.

## Methodology

### 1. Features (all backward-looking)
For each asset on each month-end *t*: 5-, 20- and 60-day returns, 20-day annualized
volatility, price relative to its 20-day moving average, and a 14-day RSI (a
momentum indicator). Every feature is a trailing window that ends at *t*.
`run.py` checks this directly: features computed on prices truncated at *t* must
equal the full-sample features at *t*.

**Target:** the asset's return from month-end *t* to the next month-end.

### 2. Return forecasts
* **Baseline:** trailing 252-day mean daily return × 21 (next-month forecast).
* **ML:** one `RandomForestRegressor` (100 trees, `max_depth=4`,
  `min_samples_leaf=50`, `max_features=0.5`). It is pooled across assets so it
  learns one mapping from features to next-month return. The configuration was
  fixed in advance and never tuned.

**Time-series validation.** There is no random `train_test_split`. The model is
trained walk-forward on an **expanding window** and refit every 12 months. At a
rebalance date *t* it only sees rows dated before *t*. Those rows' labels end on
or before *t*, so every label it trains on was already realized. The first
forecast needs 36 months of labelled history. Every prediction is therefore made
on data the model has never seen.

### 3. Black-Litterman view blending
Black-Litterman gives a principled way to blend noisy views with a stable prior:

1. **Equilibrium (prior) returns:** π = δ Σ w_mkt, with δ = 2.5. There are no
   market caps for a mixed ETF/stock universe, so the equal-weight portfolio is
   used as the reference "market" portfolio. This choice is documented.
2. **ML views:** one absolute view per asset (P = I). Q is the annualized RF forecast.
3. **View uncertainty:** Ω = diag(P τΣ Pᵀ) / c, where τ = 0.05 and the confidence
   c = 0.5. At c = 0.5 an ML view is trusted half as much as the prior.
4. **Posterior:** μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q].

μ_BL is then passed to the same max-Sharpe optimizer as the other strategies. The
prior, views and posterior at the last rebalance are saved to
`results/black_litterman_last.csv`.

### 4. Risk estimation
Sample covariance of the trailing 252 daily returns, annualized × 252 and
re-estimated at every rebalance.

### 5. Portfolio optimization
`scipy.optimize.minimize` (SLSQP) with w ≥ 0, Σw = 1 and a 40% cap per asset. The
cap keeps any single forecast from concentrating the whole portfolio in one name.
`src/portfolio.py` implements `portfolio_return`, `portfolio_variance`,
`portfolio_volatility` and `sharpe_ratio`.

| Strategy | Expected returns | Objective |
|---|---|---|
| Equal Weight | none | 1/N |
| Min Volatility | none | min wᵀΣw |
| Historical Mean MaxSharpe | trailing mean | max Sharpe |
| ML Forecast MaxSharpe | RF forecast | max Sharpe |
| Black-Litterman + ML | BL posterior | max Sharpe |

If no asset has a positive expected return, max-Sharpe falls back to min-volatility.

### 6. Transaction costs and turnover
At each rebalance the previous weights have drifted with prices. Then:

```
turnover_t = Σ_i |w_target,i − w_drifted,i|
cost_t     = turnover_t × cost_rate        (default 10 bps)
```

The cost is taken out of portfolio value on the day of the trade, so **every
reported performance number is net of costs**. The first rebalance builds the
portfolio from cash (turnover = 1) for every strategy.

### 7. Backtest
Month-end rebalancing. Weights are computed from data up to the close of *t*,
traded at that close, and then earn the daily returns of (t, t_next]. All
strategies share the same out-of-sample dates. `run.py` asserts this, along with
the weight constraints and that all training data predates each forecast.

## Results

Running `python run.py` writes:

| File | Contents |
|---|---|
| `results/performance.csv` | Cumulative, annualized return, volatility, Sharpe, max drawdown, turnover, costs, gross vs net |
| `results/ml_metrics.csv` | MAE, RMSE, directional accuracy, correlation, rank IC: RF vs baseline |
| `results/robustness_costs_sharpe.csv` | Net Sharpe at 0 / 10 / 25 / 50 bps |
| `results/black_litterman_last.csv` | π, Q and μ_BL at the last rebalance |
| `results/*.png` | Net cumulative performance, drawdowns, predicted vs realized, average allocations, feature importance |

> **Status:** the market-data results have not been generated yet. The environment
> this was built in could not reach Yahoo Finance, so the pipeline was only
> validated on synthetic prices. Run `python run.py` with network access to
> produce the cache and the results above. Draw conclusions only from those files.

### How to read them
* **Judge strategies on net performance.** A strategy that trades a lot can win
  before costs and lose after them. The cost-robustness table shows how quickly
  each strategy's Sharpe falls as costs rise.
* **Prediction accuracy ≠ investment performance.** A model can have a lower
  RMSE than the baseline and still build worse portfolios. RMSE rewards forecasts
  close to zero, but the optimizer only cares about how assets *rank* relative to
  each other, and about the extra trading that forecast changes cause. That is why
  a cross-sectional rank IC is reported alongside MAE/RMSE.
* **Monthly single-stock and ETF returns are mostly noise.** An out-of-sample
  correlation of a few percent would already be typical. It is a legitimate
  result if the Random Forest does **not** beat the historical-mean baseline or
  equal weighting. The project does not assume ML adds value; it tests whether it
  does.

## Concepts

**Portfolio optimization / mean-variance theory.** Markowitz: pick weights to trade
off expected return μᵀw against variance wᵀΣw. The tangency (max-Sharpe) portfolio
has the best return per unit of risk.

**Covariance and diversification.** Portfolio risk depends on how assets move
together, not only on each asset's volatility. Combining imperfectly correlated
assets (equities, Treasuries, gold) lowers risk below the weighted average of the
individual volatilities.

**Sharpe ratio.** (annualized return − r_f) / annualized volatility. Here r_f = 0,
so the reported numbers are return-to-risk ratios.

**ML return forecasting.** A Random Forest can pick up non-linear effects such as
short-term reversal or volatility regimes. With few noisy observations it can also
fit patterns that do not persist.

**Time-series validation and look-ahead bias.** Shuffled cross-validation lets a
model train on the future and test on the past, which inflates results. This
project uses only past data at every step: expanding-window training, trailing
features and trailing covariance.

**Black-Litterman.** Raw return estimates make mean-variance weights extreme and
unstable. Black-Litterman starts from equilibrium returns implied by a reference
portfolio and tilts toward the views in proportion to how much they are trusted.

**Transaction costs and turnover.** Turnover is how much of the portfolio is
traded at a rebalance. Forecast-driven strategies usually trade much more than
equal weight or min-vol, and costs can wipe out a thin edge.

**Out-of-sample backtesting.** Every decision uses only the information available
on the day it is made, and all strategies are compared over the same period.

## Limitations

* **Financial ML is vulnerable to overfitting and estimation error.** There are
  only ~130 monthly decisions, and a small edge cannot be told apart from luck.
  None of the results should be read as evidence of a profitable strategy.
* **Survivorship and selection bias.** The universe is today's well-known, liquid
  names, which are known to have survived and done well.
* The cost model is simple and proportional. There is no spread, market impact,
  borrowing or tax modelling. Trades fill at the rebalance-day close.
* r_f = 0, no cash asset, long-only, and one fixed parameter set (no
  sensitivity sweep beyond cost rates).
* Equal weights stand in for market-cap weights in the Black-Litterman prior.
* Sample covariance is noisy with 10 assets and 252 observations. Shrinkage would
  be a natural extension.

## Project structure

```
run.py                 pipeline, configuration, sanity checks, reporting
src/data.py            yfinance download + CSV cache, cleaning, month-ends
src/features.py        backward-looking features and next-month target
src/models.py          historical-mean baseline, walk-forward Random Forest
src/black_litterman.py equilibrium returns and posterior
src/portfolio.py       portfolio maths and SLSQP optimizers
src/backtest.py        signal generation, cost-aware simulator
src/evaluation.py      performance and forecast metrics, plots
NOTES.md               mathematical notes and design decisions
```
