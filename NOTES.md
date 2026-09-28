# Notes: mathematics and implementation decisions

Notation: N assets, weights w ∈ ℝᴺ, expected returns μ, covariance Σ (both
annualized), risk-free rate r_f (= 0 here).

## Core formulas

| Quantity | Formula | Code |
|---|---|---|
| Expected return | μ_p = wᵀμ | `portfolio_return` |
| Variance | σ²_p = wᵀΣw = Σᵢ Σⱼ wᵢwⱼσᵢⱼ | `portfolio_variance` |
| Volatility | σ_p = √(wᵀΣw) | `portfolio_volatility` |
| Sharpe ratio | S = (μ_p − r_f) / σ_p | `sharpe_ratio` |
| Turnover at rebalance t | τ_t = Σᵢ \|w_{t,i} − w̃_{t,i}\| | `simulate` |
| Drifted weight | w̃_{t,i} = w_{t−1,i}(1+R_i) / Σⱼ w_{t−1,j}(1+R_j) | `simulate` |
| Transaction cost | C_t = τ_t × κ (κ = cost rate, default 0.001) | `simulate` |
| Net return, rebalance day | r_net = (1 − C_t)(1 + r_gross) − 1 | `simulate` |
| Equilibrium returns | π = δ Σ w_mkt | `implied_equilibrium_returns` |
| View uncertainty | Ω = diag(P τΣ Pᵀ) / c | `black_litterman` |
| BL posterior mean | μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q] | `black_litterman` |

Annualization: daily mean × 252, daily covariance × 252, monthly forecast × 12.
Reported Sharpe = mean(daily net) / std(daily net) × √252. Annualized return is
geometric: (1 + cumulative)^(252 / days) − 1.

## Why covariance matters and diversification

σ²_p = Σᵢ wᵢ²σᵢ² + Σ_{i≠j} wᵢwⱼρᵢⱼσᵢσⱼ. When ρ < 1 the cross terms are smaller
than they would be for perfectly correlated assets, so σ_p < Σᵢ wᵢσᵢ. That gap is
the diversification benefit. The optimizer sees assets only through μ and Σ.
Getting Σ wrong mis-sizes risk, and near-singular Σ matrices let the optimizer
place large bets on spurious "hedges".

## Estimation error

With T observations the standard error of an annualized mean return is about
σ/√T_years. For σ = 20% over 1 year that is ±20%, which is larger than any
plausible true premium. Covariances are estimated much more precisely than means.
This is why min-volatility and equal-weight portfolios, which ignore μ entirely,
are so hard to beat. Mean-variance optimization amplifies the error in μ. It
over-weights assets whose returns were overestimated, which Michaud called
"error maximization".

## Why Black-Litterman stabilizes return estimates

The BL posterior is a precision-weighted average of the prior π and the views Q.
With P = I it simplifies to

    μ_BL = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ [(τΣ)⁻¹π + Ω⁻¹Q]

With the He-Litterman choice Ω = τ diag(Σ)/c, and ignoring cross-correlations,
each asset's posterior is roughly (π + cQ)/(1 + c). With c = 0.5 that is ⅔ prior
and ⅓ view. The prior π = δΣw_mkt is consistent with Σ by construction: its
max-Sharpe portfolio is w_mkt itself. Starting from π means that without strong
views the optimizer holds a diversified portfolio, and each view tilts it only by
an amount proportional to its confidence. This shrinks noisy ML forecasts toward
a sensible anchor instead of taking them at face value, which should mean less
concentration and less turnover than raw ML max-Sharpe.

Choices made:
* **Absolute views, P = I.** This is the simplest mapping from one forecast per asset.
* **τ = 0.05.** A conventional value (τ ≈ 1/T). With He-Litterman Ω the posterior
  depends on c much more than on τ.
* **c = 0.5.** A prior-dominated setting chosen in advance because monthly ML
  forecasts are noisy. It was not tuned on backtest results.
* **w_mkt = equal weight.** The universe mixes ETFs and single stocks, so a
  market-cap vector is not meaningful. Equal weight is a neutral, look-ahead-free
  reference.
* **δ = 2.5.** A common risk-aversion value. It scales π and so sets how
  "optimistic" the prior is compared with the ML views.

## Why ML forecasts may be noisy, and may legitimately fail

* Monthly returns have a very low signal-to-noise ratio. The predictable part of
  returns is usually a few percent of the variance.
* The training set is small. There are ~10 assets × ~36–170 months, and pooled
  rows are cross-sectionally correlated, so the effective sample size is even
  smaller than the row count.
* Markets are non-stationary. Relationships learned in 2013–2018 (a low-rate bull
  market) need not hold in 2022 (rate shock).
* The target is noisy, so a shallow forest mostly predicts values near the mean.
  That can give a slightly lower RMSE than a historical mean while carrying almost
  no cross-sectional ranking information.
* Every forecast change creates turnover and therefore cost.

The model is shallow with large leaves (`max_depth=4`, `min_samples_leaf=50`) and
uses a fixed configuration. **It was deliberately not tuned until it beat the
baseline.** Iterating on the backtest would have made the out-of-sample period part
of the training data. A negative result here is informative, not a bug.

## Look-ahead bias (and how it is prevented)

Using information at time t that was not available until later.

* Features use trailing windows ending at t. `check_no_lookahead` in `run.py`
  recomputes them on prices truncated at t and asserts equality.
* The label for month s is P(s⁺)/P(s) − 1, where s⁺ is the next month-end. At
  rebalance t the model trains only on rows with s < t, so s⁺ ≤ t and each label
  is known. `run.py` asserts that the last training date comes before t.
* Covariance and historical mean use `daily.loc[:t]`.
* Weights set at t's close earn returns from t+1 onward.
* Forward-filling missing prices uses the past only.
* On synthetic random-walk data the RF's out-of-sample rank IC is about 0, as it
  should be. Any leak would show up as spurious skill on pure noise.

## Survivorship bias

The tickers were chosen in 2026 and are all names that survived and prospered
(AAPL, MSFT, QQQ). Their historical returns are biased upward, which flatters
every strategy, most of all return-chasing ones. A rigorous study would use
point-in-time index constituents including delisted firms.

## Transaction costs and turnover

Turnover is measured against drifted weights, not last month's targets. Even
equal weight has non-zero turnover because it must rebalance back to 1/N. Costs
are charged as a fraction of portfolio value on the trade date, so they compound
through the equity curve. Strategy decisions do not depend on the cost rate, so
the cost-robustness check reuses the same weights and re-simulates at 0 / 10 / 25
/ 50 bps. This is cheap and isolates the effect of costs.

## Overfitting

Researcher degrees of freedom (features, windows, universe, model settings,
BL parameters) are just as dangerous as model complexity. Mitigations here:
parameters fixed before the run, one model, no search, simple baselines, a
single out-of-sample backtest, and a report of whatever comes out. With ~130
monthly observations a Sharpe difference of 0.3 is well within noise. The
standard error of an annual Sharpe estimate over 11 years is ≈ √((1 + S²/2)/11) ≈ 0.3.

## Other implementation decisions

* **Pooled RF with no asset identifier.** The model learns feature → return
  relationships shared across assets instead of memorizing each asset's mean.
* **Annual refits.** Keeps compute tiny (about 12 fits) while still adapting to
  new data.
* **40% weight cap.** An extra constraint besides w ≥ 0 and Σw = 1. Without it a
  single optimistic forecast can put almost everything into one asset.
* **Max-Sharpe fallback.** If all expected returns are ≤ r_f the tangency
  portfolio is not defined, so the optimizer uses min-volatility instead.
* **SLSQP failure.** Falls back to equal weights. This has not happened in
  practice, but it prevents an invalid portfolio.
