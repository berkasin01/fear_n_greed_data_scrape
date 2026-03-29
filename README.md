# CNN Fear and Greed Index - Anomaly Detection and Analysis
Historical CNN Fear and Greed Index data from 2011 to present, with anomaly detection using statistical and ML methods.
What It Does
The CNN Fear and Greed Index measures market sentiment on a scale of 0 (Extreme Fear) to 100 (Extreme Greed). This repo takes 15+ years of daily data and runs anomaly detection to find unusual market sentiment days.
The project covers:

Data scraping from CNN's public API to keep the dataset updated
Feature engineering from a single column (daily change, pct change, rolling stats, z-scores)
Anomaly detection using Z-score, IQR, Isolation Forest, and DBSCAN
Visualisation and analysis of results

Data
cnn_fear_and_greed_index.csv contains two columns:

date - trading day (YYYY-MM-DD)
combined_value - Fear and Greed score (0-100, integer)

3968 rows from January 2011 to present. Missing dates are market holidays or weekends.
Methods
Statistical: Z-score and IQR on both raw scores and daily changes. IQR found significantly more anomalies than Z-score because Z-score's mean and std get dragged by the extreme values it is trying to detect.
ML (coming soon): Isolation Forest and DBSCAN for comparison against the statistical methods.
Key Findings

IQR catches more anomalies than Z-score across both features, which is expected given how extreme values skew mean and std
Z-score flags Monday and Tuesday more for daily change anomalies, possibly from weekend sentiment building up. IQR shows a more even spread so the pattern may not be as strong as it looks
No clear day-of-week pattern for extreme scores with either method, meaning sustained fear or greed is driven by broader market events not weekly cycles

Usage
Update the CSV with latest data:
python get_cnn_greed_index.py
Run the analysis:
python data_preprocessing.py
Requirements
pip install pandas numpy matplotlib scikit-learn requests
```
