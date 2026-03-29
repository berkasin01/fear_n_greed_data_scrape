# CNN Fear and Greed Index Data PreProcess, Anomaly Machine Learning Models

Historical CNN Fear and Greed Index data from 2011 to present, with a Python scraper to keep it updated.

## What It Does

The CNN Fear and Greed Index measures market sentiment on a scale of 0 (Extreme Fear) to 100 (Extreme Greed) based on seven indicators: market momentum, stock price strength, stock price breadth, put/call options, market volatility, safe haven demand, and junk bond demand.

This repo provides:

1. A CSV file with daily Fear and Greed scores going back to January 2011
2. A Python script that scrapes the latest data from CNN and merges it with the existing historical file

## Data

`cnn_fear_and_greed_index.csv` contains two columns:

- `date` — trading day (YYYY-MM-DD)
- `combined_value` — Fear and Greed score (0-100, integer)

Missing dates are market holidays or weekends when no data is published.

## Usage

Run the scraper to update the CSV with the latest data:

```
python get_cnn_greed_index.py
```

The script fetches the most recent data from CNN's public API, merges it with the existing CSV, deduplicates, and saves. Run it whenever you want to bring the data up to date.

## Requirements

```
pip install requests pandas
```

## Why This Exists

Most financial sentiment data is either locked behind paid APIs or only available as a live snapshot with no history. This repo provides 15+ years of daily Fear and Greed data for free, ready to use in ML projects, backtesting, or market analysis.
