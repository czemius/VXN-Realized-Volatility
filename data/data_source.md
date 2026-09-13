# Data Sources



## VXN



Source: https://fred.stlouisfed.org/series/VXNCLS



Daily VXN observations are sourced from the Federal Reserve Bank of St. Louis (FRED).



## NQ



Source: https://huggingface.co/datasets/mdelcristo/NQ-F\_1min\_OHLCV\_Parquet



NQ 1-minute OHLCV data is sourced from the mdelcristo NQ-F\_1min\_OHLCV\_Parquet dataset on Hugging Face.



The datasets are not included in this repository. Download the source data and place it in the appropriate `data/` directories before running the pipeline.



## Data Notes



- The original NQ 1-minute Parquet files use UTC timestamps.

- Timestamps are converted to `America/New\_York` when assigning observations to CME trading sessions.

- The 2025 NQ dataset is partial and currently ends on 2025-07-25.

- VXN is daily data, while NQ is 1-minute data.

