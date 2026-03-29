import pandas as pd
import numpy as np

df = pd.read_csv("cnn_fear_and_greed_index.csv")

df["daily_change"] = df["combined_value"].diff()
df["pct_change"] = df["combined_value"].pct_change() * 100
df["rolling_mean_7"] = df["combined_value"].rolling(7).mean()
df["rolling_std_7"] = df["combined_value"].rolling(7).std()
df["distance_from_mean"] = df["combined_value"] - df["rolling_mean_7"]

