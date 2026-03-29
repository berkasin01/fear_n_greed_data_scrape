# CNN Fear and Greed Index - Anomaly Detection

Historical CNN Fear and Greed Index data from 2011 to present with anomaly detection using statistical and ML methods.

## What It Does

Takes 15+ years of daily CNN Fear and Greed Index data (3968 rows) and runs four different anomaly detection methods to find unusual market sentiment days. The project covers data scraping, feature engineering, statistical analysis, and unsupervised ML.

## Data

`cnn_fear_and_greed_index.csv` contains two columns:
- `date` - trading day (YYYY-MM-DD)
- `combined_value` - Fear and Greed score (0-100, integer)

Missing dates are market holidays or weekends.

## Feature Engineering

From one column I built: daily change, percentage change, 7-day rolling mean, 7-day rolling standard deviation, distance from rolling mean, and day of week. These features feed into both the statistical and ML methods.

## Methods and Results

Four anomaly detection approaches on the same dataset:

| Method | Anomalies Found | Type |
|--------|----------------|------|
| Z-Score | ~92 | Statistical |
| IQR | ~1000+ | Statistical |
| Isolation Forest | ~200 | Unsupervised ML |
| DBSCAN | 29 | Unsupervised ML |

**Z-Score** - the strictest statistical method. Uses mean and std which get dragged by extreme values, so it misses more anomalies than you would expect.

**IQR** - catches the most because it uses quartiles instead of mean/std so extreme values do not throw it off. The 1.5x fence is wider than Z-score's threshold.

**Isolation Forest** - uses random splits to isolate unusual points. The contamination parameter controls how many anomalies it flags (set to 5%). Catches multivariate patterns the statistical methods miss.

**DBSCAN** - the most selective. Only flags points that are completely alone in feature space with no nearby cluster. Found just 29 true outliers.

## Key Findings

- IQR found way more anomalies than Z-score. Makes sense because Z-score's mean and std get dragged by the extreme values it is trying to catch
- Z-score flags Monday and Tuesday more for daily change anomalies, possibly from weekend sentiment building up. IQR shows it more evenly spread so the pattern might not be as strong as it looks
- No clear day-of-week pattern for extreme fear/greed scores. Sustained extreme sentiment comes from broader market events not weekly cycles
- DBSCAN was the most conservative, only flagging the most extreme outliers in multi dimensional feature space
- Isolation Forest and DBSCAN both show Monday and Friday leading, suggesting bookend days of the trading week see more unusual activity

## Usage

Update the CSV:
```
python get_cnn_greed_index.py
```

Run preprocessing and analysis:
```
python data_preprocessing.py
```

Run ML models:
```
python ml_models.py
```

## Requirements
```
pip install pandas numpy matplotlib scikit-learn requests
```

## Why This Exists

Most financial sentiment data is locked behind paid APIs or only available as a live snapshot. This repo gives you 15+ years of daily data for free plus shows how different anomaly detection methods compare on real financial data.