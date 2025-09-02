# LSTMeval_lightgbm.py
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ---------- Load & Prepare Data ----------
df = pd.read_csv("all_outlets_sales_1year.csv")

# Ensure datetime
df["created_at"] = pd.to_datetime(df["created_at"])
df = df.sort_values(["outlet_id", "product_id", "created_at"]).reset_index(drop=True)

# Encode categorical IDs
for col in ["product_id", "outlet_id"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ---------- Feature Engineering ----------
def create_features(data):
    data = data.copy()
    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        data[f"lag_{lag}"] = data.groupby(["outlet_id", "product_id"])["quantity"].shift(lag)
    # Rolling means
    for window in [7, 14, 30]:
        data[f"roll_mean_{window}"] = (
            data.groupby(["outlet_id", "product_id"])["quantity"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )
    return data

df = create_features(df)
df = df.dropna().reset_index(drop=True)  # drop rows where lag features not available

# ---------- Train/Test Split (time-based) ----------
cutoff_date = df["created_at"].max() - pd.DateOffset(months=1)
train_df = df[df["created_at"] <= cutoff_date]
test_df = df[df["created_at"] > cutoff_date]

FEATURES = [col for col in df.columns if col not in ["quantity", "created_at", 
                                                     "product_name", "product_attribute", 
                                                     "product_type", "outlet_name"]]
TARGET = "quantity"

# ---------- Train LightGBM ----------
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)

model.fit(
    train_df[FEATURES],
    train_df[TARGET],
    eval_set=[(test_df[FEATURES], test_df[TARGET])],
    eval_metric="rmse"
)

# ---------- Predictions ----------
test_df["forecast"] = model.predict(test_df[FEATURES])

# ---------- Evaluation ----------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # compatible with older sklearn
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    r2 = r2_score(y_true, y_pred)
    bias = (np.sum(y_pred - y_true) / np.sum(y_true)) * 100
    return mae, rmse, mape, wmape, r2, bias

# Top 3 products by total sales
top3_products = (
    df.groupby("product_id")["quantity"].sum().nlargest(3).index.tolist()
)

print("\n==== Top 3 Products ====")
print(top3_products)

top3_df = test_df[test_df["product_id"].isin(top3_products)]

mae, rmse, mape, wmape, r2, bias = calculate_metrics(top3_df[TARGET], top3_df["forecast"])

print("\n==== LightGBM Metrics (Top 3) ====")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE (%): {mape:.4f}")
print(f"wMAPE (%): {wmape:.4f}")
print(f"R²: {r2:.4f}")
print(f"Bias (%): {bias:.4f}")
