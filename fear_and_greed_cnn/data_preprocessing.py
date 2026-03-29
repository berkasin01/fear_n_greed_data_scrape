import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("cnn_fear_and_greed_index.csv")
print(df.shape)

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()

df["daily_change"] = df["combined_value"].diff()
df["pct_change"] = df["combined_value"].pct_change() * 100
df["rolling_mean_7"] = df["combined_value"].rolling(7).mean()
df["rolling_std_7"] = df["combined_value"].rolling(7).std()
df["distance_from_mean"] = df["combined_value"] - df["rolling_mean_7"]


df["z_score_combined_val"] = (df["combined_value"] - df["combined_value"].mean()) / df["combined_value"].std()
df["z_score_daily_change"] = (df["daily_change"] - df["daily_change"].mean()) / df["daily_change"].std()

df = df.dropna()

print("Combined value anomalies (|z| > 2):")
print(df[df["z_score_combined_val"].abs() > 2][["date","day_of_week", "combined_value", "z_score_combined_val"]])

print("\nDaily change anomalies (|z| > 2):")
print(df[df["z_score_daily_change"].abs() > 2][["date", "day_of_week", "daily_change", "z_score_daily_change"]])

