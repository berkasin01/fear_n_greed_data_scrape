import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("cnn_fear_and_greed_index.csv")
print(df.shape)

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()

#feature engineer
df["daily_change"] = df["combined_value"].diff()
df["pct_change"] = df["combined_value"].pct_change() * 100
df["rolling_mean_7"] = df["combined_value"].rolling(7).mean()
df["rolling_std_7"] = df["combined_value"].rolling(7).std()
df["distance_from_mean"] = df["combined_value"] - df["rolling_mean_7"]

#z score
df["z_score_combined_val"] = (df["combined_value"] - df["combined_value"].mean()) / df["combined_value"].std()
df["z_score_daily_change"] = (df["daily_change"] - df["daily_change"].mean()) / df["daily_change"].std()

#clean values na values change columns create
df = df.dropna()

df_combined_z_score = df[df["z_score_combined_val"].abs() > 2][
    ["date", "day_of_week", "combined_value", "z_score_combined_val"]]
c_val_counts = df_combined_z_score["day_of_week"].value_counts()
# plt.bar(c_val_counts.index, c_val_counts.values)
# plt.title("Z Score of Fear & Greed Per Score")
# plt.xlabel("Day")
# plt.ylabel("Count")
# plt.show()

df_daily_change_z_score = df[df["z_score_daily_change"].abs() > 2][
    ["date", "day_of_week", "daily_change", "z_score_daily_change"]]
print(df_daily_change_z_score)
dc_val_counts = df_daily_change_z_score["day_of_week"].value_counts()
# plt.bar(dc_val_counts.index, dc_val_counts.values)
# plt.title("Z Score of Fear & Greed Per Change of Score")
# plt.xlabel("Day")
# plt.ylabel("Count")
# plt.show()


#IQR calculation
combined_q1 = df["combined_value"].quantile(0.25)
combined_q2 = df["combined_value"].quantile(0.50)
combined_q3 = df["combined_value"].quantile(0.75)

combined_iqr = (combined_q3 - combined_q1)
combined_fence = combined_iqr * 1.5
combined_upper = combined_iqr + combined_fence
combined_lower = combined_iqr - combined_fence

daily_q1 = df["daily_change"].quantile(0.25)
daily_q2 = df["daily_change"].quantile(0.50)
daily_q3 = df["daily_change"].quantile(0.75)

daily_iqr = (daily_q3 - daily_q1)
daily_fence = daily_iqr * 1.5
daily_upper = daily_iqr + daily_fence
daily_lower = daily_iqr - daily_fence

df["iqr_anomaly_combined"] = (df["combined_value"] < combined_lower) | (df["combined_value"] > combined_upper)
iqr_combined_anomalies = df[df["iqr_anomaly_combined"] == True]
# iqr_c_val_counts = iqr_combined_anomalies["day_of_week"].value_counts()
# plt.bar(iqr_c_val_counts.index, iqr_c_val_counts.values)
# plt.title("IQR Combined Value Anomalies by Day of Week")
# plt.xlabel("Day")
# plt.ylabel("Count")
# plt.show()

df["iqr_anomaly_daily"] = (df["daily_change"] < daily_lower) | (df["daily_change"] > daily_upper)
iqr_daily_anomalies = df[df["iqr_anomaly_daily"] == True]
iqr_d_val_counts = iqr_daily_anomalies["day_of_week"].value_counts()
# plt.bar(iqr_d_val_counts.index, iqr_d_val_counts.values)
# plt.title("IQR Daily Change Anomalies by Day of Week")
# plt.xlabel("Day")
# plt.ylabel("Count")
# plt.show()

