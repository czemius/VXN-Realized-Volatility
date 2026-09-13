import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DAILY_DIR = PROJECT_ROOT / "data" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "visualization" / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Load daily data
# ------------------------------------------------------------

files = sorted(DAILY_DIR.glob("*.parquet"))

daily = pd.concat(
    [pd.read_parquet(file) for file in files],
    ignore_index=True
)

daily = (
    daily
    .sort_values("trading_date")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Calculate future realized volatility
# ------------------------------------------------------------

horizon = 30

cumulative_variance = daily["realized_variance"].cumsum()

future_variance = (
    cumulative_variance.shift(-horizon)
    - cumulative_variance
)

daily["future_RV_30D"] = (
    np.sqrt(
        future_variance * 252 / horizon
    ) * 100
)

daily["VRP_30D"] = (
    daily["VXN"]
    - daily["future_RV_30D"]
)

daily = daily.dropna(
    subset=["VXN", "future_RV_30D"]
)


# ------------------------------------------------------------
# Regression statistics
# ------------------------------------------------------------

x = daily["VXN"].to_numpy()
y = daily["future_RV_30D"].to_numpy()

beta, alpha = np.polyfit(x, y, 1)

predicted = alpha + beta * x

ss_res = np.sum((y - predicted) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)

r_squared = 1 - (ss_res / ss_tot)


# ============================================================
# CHART 1
# VXN vs Future 30D Realized Volatility
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    alpha=0.35,
    s=12
)

x_line = np.linspace(
    x.min(),
    x.max(),
    100
)

y_line = alpha + beta * x_line

plt.plot(
    x_line,
    y_line,
    linewidth=2
)

plt.xlabel("VXN (%)")
plt.ylabel("Future 30D Realized Volatility (%)")
plt.title("VXN vs Future 30D NQ Realized Volatility")

plt.text(
    0.05,
    0.95,
    f"Regression: RV = {alpha:.2f} + {beta:.2f} × VXN\n"
    f"R² = {r_squared:.2f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vxn_vs_future_rv_30d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 2
# VXN Quintiles vs Future Realized Volatility
# ============================================================

daily["VXN_quintile"] = pd.qcut(
    daily["VXN"],
    5,
    labels=[
        "Lowest 20%",
        "20–40%",
        "40–60%",
        "60–80%",
        "Highest 20%"
    ]
)

quintile_stats = (
    daily
    .groupby("VXN_quintile", observed=False)
    .agg(
        VXN=("VXN", "mean"),
        future_RV=("future_RV_30D", "mean")
    )
)

fig, ax = plt.subplots(figsize=(10, 6))

positions = np.arange(len(quintile_stats))
width = 0.35

ax.bar(
    positions - width / 2,
    quintile_stats["VXN"],
    width,
    label="VXN"
)

ax.bar(
    positions + width / 2,
    quintile_stats["future_RV"],
    width,
    label="Future 30D RV"
)

ax.set_xticks(positions)
ax.set_xticklabels(quintile_stats.index)

ax.set_xlabel("VXN Regime")
ax.set_ylabel("Volatility (%)")
ax.set_title("VXN vs Future 30D Realized Volatility by VXN Regime")

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vxn_quintiles_vs_future_rv.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 3
# VRP Distribution
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    daily["VRP_30D"],
    bins=50,
    alpha=0.75
)

plt.axvline(
    daily["VRP_30D"].mean(),
    linestyle="--",
    linewidth=2,
    label=f"Mean VRP = {daily['VRP_30D'].mean():.2f}"
)

plt.axvline(
    0,
    linestyle="-",
    linewidth=1
)

plt.xlabel("VXN − Future 30D Realized Volatility (percentage points)")
plt.ylabel("Frequency")
plt.title("Distribution of 30D Volatility Risk Premium")

plt.legend()
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vrp_distribution_30d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 4
# VRP by VXN Regime
# ============================================================

vrp_by_quintile = (
    daily
    .groupby("VXN_quintile", observed=False)["VRP_30D"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.bar(
    vrp_by_quintile.index,
    vrp_by_quintile.values
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("VXN Regime")
plt.ylabel("Mean VRP (percentage points)")
plt.title("30D Volatility Risk Premium by VXN Regime")

plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vrp_by_vxn_regime.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CHART 5
# VXN vs Future Realized Volatility Through Time
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily["trading_date"],
    daily["VXN"],
    linewidth=1,
    label="VXN"
)

plt.plot(
    daily["trading_date"],
    daily["future_RV_30D"],
    linewidth=1,
    label="Future 30D RV"
)

plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.title("VXN vs Future 30D NQ Realized Volatility Through Time")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "vxn_vs_future_rv_30d_time_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("Visualization complete.")
print()
print(f"Observations: {len(daily):,}")
print(f"Mean VXN: {daily['VXN'].mean():.2f}%")
print(f"Mean Future 30D RV: {daily['future_RV_30D'].mean():.2f}%")
print(f"Mean VRP: {daily['VRP_30D'].mean():.2f}")
print(f"Regression: RV = {alpha:.2f} + {beta:.2f} × VXN")
print(f"R²: {r_squared:.2f}")
print()
print(f"Charts saved to: {OUTPUT_DIR}")