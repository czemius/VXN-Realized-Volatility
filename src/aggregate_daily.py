import pandas as pd
from pathlib import Path
import numpy as np


# Folders

input_folder = Path(
    # Path to the folder containing the NQ parquets attached with VXN
)

output_folder = Path(
    # Path to the folder where the daily aggregated files will be saved
)

output_folder.mkdir(parents=True, exist_ok=True)


# Process each yearly file

for file in sorted(input_folder.glob("*.parquet")):

    print(f"Processing {file.name}...")

    df = pd.read_parquet(file)

    # My parquets are in UTC. I want to convert them into New York time.
    # If your parquets are already in New York time, you can skip this step.

    # Convert timestamps and put them in New York time.
    # The CME session is defined in New York time.

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    timestamp_ny = (
        df["timestamp"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/New_York")
    )


    # NQ's trading session runs from 18:00 to 17:00 the next day.
    # We label the evening part as the following day's session.
    # Monday 18:00 -> Tuesday session
    # Tuesday 10:00 -> Tuesday session

    calendar_date = timestamp_ny.dt.normalize()

    evening_session = timestamp_ny.dt.hour >= 18

    trading_date = (
        calendar_date
        + pd.to_timedelta(
            evening_session.astype(int),
            unit="D"
        )
    )

    df["trading_date"] = trading_date.dt.date


    # Make sure the minutes are in chronological order.

    df = df.sort_values("timestamp")

    # 1-minute log returns.
    # Don't let the first minute of a session inherit the previous session's return.

    df["log_return"] = np.log(
     df["close"]
     / df.groupby("trading_date")["close"].shift(1)
    )

    # Collapse the 1-minute data into one row per trading session.

    daily = (
        df.groupby("trading_date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),

            # Realized variance = sum of squared intraday returns.
            realized_variance=(
                "log_return",
                lambda x: np.sum(x.dropna() ** 2)
            ),

            VXN=("VXNCLS", "first")
        )
        .reset_index()
    )


    # Take the square root to get realized volatility.

    daily["RV"] = np.sqrt(
        daily["realized_variance"]
    )


    # Annualize RV so it's on the same scale as VXN.

    daily["RV_annualized"] = (
        daily["RV"] * np.sqrt(252)
    )

    # Save the daily dataset.

    output_file = output_folder / file.name

    daily.to_parquet(
        output_file,
        index=False
    )

    # Quick check

    print(f"Saved: {output_file}")
    print(f"Trading days: {len(daily):,}")

    print(
        daily[
            [
                "trading_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "VXN",
                "realized_variance",
                "RV",
                "RV_annualized"
            ]
        ].head()
    )

    print()

print("Daily aggregation complete.")