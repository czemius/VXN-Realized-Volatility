import pandas as pd
from pathlib import Path


# Load

nq_folder = Path(
    # Path to the folder containing the NQ parquets
)

output_folder = Path(
    # Path to the folder where the merged files will be saved
)

vxn_file = Path(
    # Path to the VXN CSV file
)

output_folder.mkdir(parents=True, exist_ok=True)

# VXN data

vxn_df = pd.read_csv(vxn_file)

vxn_df["observation_date"] = pd.to_datetime(
    vxn_df["observation_date"]
)

vxn_df["observation_date"] = (
    vxn_df["observation_date"].dt.date
)

print("VXN data loaded.")
print(vxn_df.head())
print(vxn_df.dtypes)

# NQ Parquets

for file in nq_folder.glob("*.parquet"):

    print("\n" + "=" * 60)
    print(f"Processing: {file.name}")

    df = pd.read_parquet(file)

    # The NQ timestamps are stored in UTC.
    # For this merge we're only matching the calendar date.
    
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["date"] = df["timestamp"].dt.date

    # Merge

    # Attach the daily VXN observation to each NQ minute
    # that falls on the same calendar date.

    df = df.merge(
        vxn_df[["observation_date", "VXNCLS"]],
        left_on="date",
        right_on="observation_date",
        how="left"
    )

    df.drop(
        columns=["date", "observation_date"],
        inplace=True
    )

    # Save

    output_file = output_folder / file.name

    df.to_parquet(
        output_file,
        index=False
    )

    # Verify

    # Check how much of the NQ data actually received a VXN value.

    total_rows = len(df)

    matched_rows = df["VXNCLS"].notna().sum()

    missing_rows = df["VXNCLS"].isna().sum()

    match_percentage = (
        matched_rows / total_rows * 100
    )

    print(f"Saved: {output_file}")
    print(f"Total rows:   {total_rows:,}")
    print(f"VXN matched:  {matched_rows:,}")
    print(f"VXN missing:  {missing_rows:,}")
    print(f"Match rate:   {match_percentage:.2f}%")

    print("\nExample matched rows:")

    print(
        df[df["VXNCLS"].notna()][
            ["timestamp", "close", "VXNCLS"]
        ].head(5)
    )

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)