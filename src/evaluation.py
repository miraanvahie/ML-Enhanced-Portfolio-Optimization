"""Performance metrics, forecast metrics and the (few) plots."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order: colour follows the strategy, never its rank.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def drawdown(returns):
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def performance_metrics(gross, net, log):
    """Net-of-cost performance summary for one strategy (rf = 0)."""
    years = len(net) / 252
    cum_net = (1 + net).prod() - 1
    return {
        "Gross Cumulative Return": (1 + gross).prod() - 1,
        "Net Cumulative Return": cum_net,
        "Annualized Return (net)": (1 + cum_net) ** (1 / years) - 1,
        "Annualized Volatility": net.std() * np.sqrt(252),
        "Sharpe Ratio (net)": net.mean() / net.std() * np.sqrt(252),
        "Max Drawdown": drawdown(net).min(),
        "Total Turnover": log["turnover"].sum(),
        "Avg Turnover per Rebalance": log["turnover"].iloc[1:].mean(),
        "Total Transaction Costs": log["cost"].sum(),
    }


def forecast_metrics(pred, realized):
    err = pred - realized
    return {
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "Directional Accuracy": (np.sign(pred) == np.sign(realized)).mean(),
        "Correlation": np.corrcoef(pred, realized)[0, 1],
    }


def ml_evaluation(forecasts):
    """RF vs historical-mean baseline on identical (date, asset) out-of-sample pairs."""
    f = forecasts.dropna()
    out = {name: forecast_metrics(f[name], f["realized"]) for name in ["ml", "baseline"]}
    # Cross-sectional rank IC: what a long-only allocator actually uses.
    for name in ["ml", "baseline"]:
        ic = f.groupby("date").apply(
            lambda g: g[name].rank().corr(g["realized"].rank()), include_groups=False)
        out[name]["Mean Rank IC"] = ic.mean()
    table = pd.DataFrame(out).T
    table.index = ["Random Forest", "Historical Mean (baseline)"]
    return table


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_plots(nets, weights, forecasts, importance, outdir):
    os.makedirs(outdir, exist_ok=True)
    names = list(nets)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, s in zip(COLORS, names):
        wealth = (1 + nets[s]).cumprod()
        ax.plot(wealth, color=c, label=s)
        ax.annotate(f"{wealth.iloc[-1]:.2f}", (wealth.index[-1], wealth.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", color=MUTED, va="center")
    ax.set_title("Net cumulative growth of $1 (after transaction costs)", loc="left")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    _save(fig, f"{outdir}/cumulative_net_performance.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    for c, s in zip(COLORS, names):
        ax.plot(drawdown(nets[s]) * 100, color=c, label=s, linewidth=1.2)
    ax.set_title("Drawdown from running peak (net)", loc="left")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    _save(fig, f"{outdir}/drawdowns.png")

    f = forecasts.dropna()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(f["ml"] * 100, f["realized"] * 100, s=10, alpha=0.35, color=COLORS[0],
               edgecolors="none")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    corr = np.corrcoef(f["ml"], f["realized"])[0, 1]
    ax.set_title(f"RF predicted vs realized next-month return (corr = {corr:.3f})", loc="left")
    ax.set_xlabel("Predicted (%)")
    ax.set_ylabel("Realized (%)")
    _save(fig, f"{outdir}/ml_predicted_vs_realized.png")

    avg = pd.DataFrame({s: w.mean() for s, w in weights.items()}).T * 100
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(avg.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(avg.shape[1]), avg.columns)
    ax.set_yticks(range(avg.shape[0]), avg.index)
    ax.grid(False)
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            v = avg.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 0.6 * avg.values.max() else INK, fontsize=8)
    ax.set_title("Average portfolio weight over the out-of-sample period (%)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025)
    _save(fig, f"{outdir}/average_allocations.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(importance.index, importance.values, color=COLORS[0], height=0.6)
    ax.set_title("Random Forest feature importance (mean over refits)", loc="left")
    ax.set_xlabel("Impurity-based importance")
    ax.grid(axis="y", visible=False)
    _save(fig, f"{outdir}/feature_importance.png")
