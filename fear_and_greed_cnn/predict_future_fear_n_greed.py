import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("cnn_fear_and_greed_index.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

df["daily_change"] = df["combined_value"].diff()
df["pct_change"] = df["combined_value"].pct_change() * 100
df["rolling_mean_7"] = df["combined_value"].rolling(7).mean()
df["rolling_std_7"] = df["combined_value"].rolling(7).std()
df["distance_from_mean"] = df["combined_value"] - df["rolling_mean_7"]


df["target"] = (df["daily_change"].shift(-1) > 0).astype(int)
df = df.dropna()

features = ["daily_change", "pct_change", "rolling_mean_7", "rolling_std_7", "distance_from_mean", "combined_value"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")
print(f"Target balance: {y.value_counts().to_dict()}")