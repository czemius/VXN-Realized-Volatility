import pandas as pd
import numpy as np
from pathlib import Path


# Folders

daily_folder = Path(
    # Path to your daily NQ and VXN folder
)


# Load daily data

files = sorted(daily_folder.glob("*.parquet"))

if not files:
    raise FileNotFoundError("No daily parquet files found.")


daily = pd.concat(
    [pd.read_parquet(file) for file in files],
    ignore_index=True
)


# Prepare the dataset

daily["trading_date"] = pd.to_datetime(
    daily["trading_date"]
)

daily = daily.sort_values(
    "trading_date"
).reset_index(drop=True)


# Remove rows where either side of the comparison is unavailable.

daily = daily.dropna(
    subset=["VXN", "realized_variance"]
).copy()


print("=" * 60)
print("REALIZED VS IMPLIED VOLATILITY")
print("=" * 60)

print(f"Observations: {len(daily):,}")
print(
    f"Date range: "
    f"{daily['trading_date'].min().date()} → "
    f"{daily['trading_date'].max().date()}"
)


# Build forward realized-volatility targets

# We use variance rather than averaging daily volatility.
# Variances add across time, which lets us construct a
# mathematically consistent multi-day realized volatility.

cumulative_variance = (
    daily["realized_variance"].cumsum()
)


horizons = [5, 10, 20, 30]


for horizon in horizons:

    # Sum variance from t+1 through t+horizon.
    #
    # cumulative_variance[t+h] - cumulative_variance[t]
    # gives exactly the variance accumulated after day t.

    future_variance = (
        cumulative_variance.shift(-horizon)
        - cumulative_variance
    )

    # Convert multi-day variance into annualized volatility.

    future_rv = np.sqrt(
        future_variance * 252 / horizon
    )

    daily[f"future_RV_{horizon}D"] = (
        future_rv * 100
    )


# Basic VXN statistics

print("\n" + "=" * 60)
print("VXN")
print("=" * 60)

print(
    f"Mean:   {daily['VXN'].mean():.2f}"
)

print(
    f"Median: {daily['VXN'].median():.2f}"
)

print(
    f"Min:    {daily['VXN'].min():.2f}"
)

print(
    f"Max:    {daily['VXN'].max():.2f}"
)


# Compare VXN with future realized volatility

results = []


for horizon in horizons:

    target = f"future_RV_{horizon}D"

    sample = daily[
        ["VXN", target]
    ].dropna()

    vxn = sample["VXN"]
    rv = sample[target]


    # Forecast error.
    #
    # Positive = VXN was higher than future RV.
    # Negative = VXN was lower than future RV.

    error = vxn - rv


    # Relative error tells us the size of the
    # forecast error relative to realized volatility.

    relative_error = (
        error / rv
    )


    # VXN / RV ratio.
    #
    # 1.00 = perfect level
    # >1.00 = VXN overestimated RV
    # <1.00 = VXN underestimated RV.

    ratio = (
        vxn / rv
    )


    results.append(
        {
            "Horizon": f"{horizon}D",
            "Observations": len(sample),
            "Mean VXN": vxn.mean(),
            "Mean Future RV": rv.mean(),
            "Mean Bias": error.mean(),
            "MAE": error.abs().mean(),
            "RMSE": np.sqrt(
                np.mean(error ** 2)
            ),
            "Mean Relative Error": relative_error.mean(),
            "Mean VXN/RV": ratio.mean(),
            "Correlation": vxn.corr(rv),
        }
    )


# Results table

results_df = pd.DataFrame(results)


print("\n" + "=" * 60)
print("VXN VS FUTURE REALIZED VOLATILITY")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Regression

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

for horizon in horizons:

    target = f"future_RV_{horizon}D"

    sample = daily[
        ["VXN", target]
    ].dropna()

    x = sample["VXN"].to_numpy()
    y = sample[target].to_numpy()


    # y = alpha + beta * VXN

    beta, alpha = np.polyfit(
        x,
        y,
        1
    )


    predicted = alpha + beta * x

    ss_res = np.sum(
        (y - predicted) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    r_squared = (
        1 - ss_res / ss_tot
    )


    print(
        f"{horizon:>2}D | "
        f"alpha = {alpha:8.4f} | "
        f"beta = {beta:8.4f} | "
        f"R² = {r_squared:.4f}"
    )


# VXN calibration by percentile

print("\n" + "=" * 60)
print("VXN CALIBRATION")
print("=" * 60)

daily["VXN_percentile"] = (
    daily["VXN"]
    .rank(pct=True)
)


daily["VXN_bucket"] = pd.qcut(
    daily["VXN_percentile"],
    5,
    labels=[
        "Lowest 20%",
        "20-40%",
        "40-60%",
        "60-80%",
        "Highest 20%"
    ]
)


for horizon in horizons:

    target = f"future_RV_{horizon}D"

    calibration = (
        daily
        .groupby("VXN_bucket", observed=True)
        .agg(
            mean_VXN=("VXN", "mean"),
            mean_future_RV=(target, "mean"),
            observations=(target, "count")
        )
    )

    print(f"\n{horizon}D")

    print(
        calibration.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

## Volatility Risk Premium

# VRP measures how much higher implied volatility was
# compared with the volatility that actually occurred in the future.

print("\n" + "=" * 60)
print("VOLATILITY RISK PREMIUM")
print("=" * 60)

vrp_results = []

for horizon in horizons:

    target = f"future_RV_{horizon}D"

    sample = daily[
        ["VXN", target]
    ].dropna().copy()


    # Positive VRP means VXN was higher than
    # subsequently realized volatility.

    sample["VRP"] = (
        sample["VXN"]
        - sample[target]
    )

    vrp = sample["VRP"]


    vrp_results.append(
        {
            "Horizon": f"{horizon}D",
            "Observations": len(sample),
            "Mean VRP": vrp.mean(),
            "Median VRP": vrp.median(),
            "Std Dev": vrp.std(),
            "25th Percentile": vrp.quantile(0.25),
            "75th Percentile": vrp.quantile(0.75),
            "Minimum": vrp.min(),
            "Maximum": vrp.max(),
            "VXN > RV %": (
                (vrp > 0).mean() * 100
            )
        }
    )


vrp_df = pd.DataFrame(
    vrp_results
)


print(
    vrp_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# VRP by VXN regime

print("\n" + "=" * 60)
print("VRP BY VXN REGIME")
print("=" * 60)

for horizon in horizons:

    target = f"future_RV_{horizon}D"

    sample = daily[
        ["VXN", target, "VXN_bucket"]
    ].dropna().copy()


    sample["VRP"] = (
        sample["VXN"]
        - sample[target]
    )


    regime_vrp = (
        sample
        .groupby(
            "VXN_bucket",
            observed=True
        )
        .agg(
            mean_VXN=("VXN", "mean"),
            mean_RV=(target, "mean"),
            mean_VRP=("VRP", "mean"),
            median_VRP=("VRP", "median"),
            observations=("VRP", "count")
        )
    )


    print(f"\n{horizon}D")

    print(
        regime_vrp.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )


print("\n" + "=" * 60)
print("VRP ANALYSIS COMPLETE")
print("=" * 60)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)