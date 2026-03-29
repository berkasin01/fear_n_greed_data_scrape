import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("cnn_fear_and_greed_index.csv")
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()

#feature engineer
df["daily_change"] = df["combined_value"].diff()
df["pct_change"] = df["combined_value"].pct_change() * 100
df["rolling_mean_7"] = df["combined_value"].rolling(7).mean()
df["rolling_std_7"] = df["combined_value"].rolling(7).std()
df["distance_from_mean"] = df["combined_value"] - df["rolling_mean_7"]

df.dropna(inplace=True)

features = df[["daily_change", "pct_change", "rolling_std_7", "distance_from_mean", "rolling_mean_7","combined_value"]].dropna()

#IsolationForest Model
iso = IsolationForest(contamination=0.05, random_state=42)
df["anom"] = iso.fit_predict(features)


anom_df = df[df["anom"] == -1]
iso_count_v = anom_df["day_of_week"].value_counts()
# plt.bar(iso_count_v.index, iso_count_v.values)
# plt.title("IsolationForest anomalies count")
# plt.xlabel("Weekday")
# plt.ylabel("Anomaly Count")
# plt.show()



scaler = StandardScaler()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
features = df[["daily_change", "pct_change", "rolling_std_7", "distance_from_mean", "rolling_mean_7","combined_value"]]
scaled = scaler.fit_transform(features)
db = DBSCAN(eps=1.5, min_samples=5)
features["dbscan_label"] = db.fit_predict(scaled)
anom_db_df = df.loc[features[features["dbscan_label"] == -1].index]
db_df_count_v = anom_db_df["day_of_week"].value_counts()
plt.bar(db_df_count_v.index, db_df_count_v.values)
plt.title("DB Scan anomalies count")
plt.xlabel("Weekday")
plt.ylabel("Anomaly Count")
plt.show()

