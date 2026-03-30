import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

df = pd.read_csv("cnn_fear_and_greed_index.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# create day of week first, then dummies
df["day_of_week"] = df["date"].dt.day_name()
day_dummies = pd.get_dummies(df["day_of_week"], prefix="day")
df = pd.concat([df, day_dummies], axis=1)

df["daily_change"] = df["combined_value"].diff()
df["pct_change"] = df["combined_value"].pct_change() * 100
df["rolling_mean_7"] = df["combined_value"].rolling(7).mean()
df["rolling_std_7"] = df["combined_value"].rolling(7).std()
df["distance_from_mean"] = df["combined_value"] - df["rolling_mean_7"]

df["target"] = (df["daily_change"].shift(-1) > 0).astype(int)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

features = ["daily_change", "pct_change", "rolling_mean_7", "rolling_std_7",
            "distance_from_mean", "combined_value",
            "day_Monday", "day_Tuesday", "day_Wednesday", "day_Thursday", "day_Friday"]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

models = {"Logistic Regression": (log_reg, log_pred),
          "Random Forest": (rf, rf_pred),
          "XGBoost": (xgb, xgb_pred)}

for name, model in [("Logistic Regression", log_reg), ("Random Forest", rf), ("XGBoost", xgb)]:
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"{name}: Mean AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

for name, (model, preds) in models.items():
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

plt.figure(figsize=(10, 6))

for name, (model, preds) in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC: {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (with Day of Week)")
plt.legend()
plt.show()

importance = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()