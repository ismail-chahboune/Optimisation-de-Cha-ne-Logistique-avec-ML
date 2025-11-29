# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 22:34:08 2025

@author: Ismail
"""

# store_item_forecast_and_inventory_opt.py
# =========================================================
# Full pipeline: Forecast (ExtraTrees) + Inventory Optimization
# Requires: pandas, numpy, scikit-learn
# =========================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------
# USER PARAMETERS
# ---------------------------
RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 8

# Inventory params
SERVICE_LEVEL = 0.95   # desired service level (e.g. 0.95)
Z = 1.65               # approximate Z for 95% (you can compute exact z from normal if you prefer)
LEAD_TIME_DAYS = 7     # assumed replenishment lead time (days)
HOLDING_COST_PER_UNIT_PER_DAY = 0.5
STOCKOUT_COST_PER_UNIT = 2.0

# ---------------------------
# 1) LOAD DATA
# ---------------------------
print("Loading data...")
train = pd.read_csv("train.csv")   # expected columns: date, store, item, sales
test  = pd.read_csv("test.csv")    # expected columns: id, date, store, item
sample = pd.read_csv("sample_submission.csv")  # expected: id, label

print("Train shape:", train.shape, "Test shape:", test.shape, "Sample shape:", sample.shape)

# ---------------------------
# 2) BASIC PREPROCESSING
# ---------------------------
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

for df in (train, test):
    df['year']    = df['date'].dt.year
    df['month']   = df['date'].dt.month
    df['day']     = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

# global medians (fallback values)
GLOBAL_MEDIAN_SALES = train['sales'].median()

# ---------------------------
# 3) CREATE LAG FEATURES FOR TRAIN
# ---------------------------
def add_lags_to_df(df, lags=[1,7,30], group_cols=['store','item'], date_col='date', value_col='sales'):
    df = df.sort_values(by=group_cols + [date_col])
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[value_col].shift(lag)
    return df

train_with_lags = add_lags_to_df(train.copy(), lags=[1,7,30])
# drop rows where lag features are NaN (can't train on them)
train_with_lags = train_with_lags.dropna(subset=['lag_1','lag_7','lag_30']).reset_index(drop=True)
print("Train with lags shape (after dropna):", train_with_lags.shape)

# ---------------------------
# 4) FEATURE SET & TARGET
# ---------------------------
FEATURES = ['store','item','year','month','day','weekday','lag_1','lag_7','lag_30']
X = train_with_lags[FEATURES]
y = train_with_lags['sales']

# ---------------------------
# 5) TIME-SAFE TRAIN/VAL SPLIT
# ---------------------------
# Use the last 10% of rows (time-ordered) as validation
split_idx = int(len(X) * 0.90)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train rows:", len(X_train), "Val rows:", len(X_val))

# ---------------------------
# 6) TRAIN FORECAST MODEL
# ---------------------------
print("Training ExtraTreesRegressor...")
model = ExtraTreesRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# ---------------------------
# 7) VALIDATION METRICS & PER-GROUP ERROR
# ---------------------------
print("Evaluating on validation set...")
val_preds = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))
mae_val = mean_absolute_error(y_val, val_preds)
print(f"Validation RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

# create a dataframe for validation to compute per-(store,item) RMSE
val_df = X_val.copy().reset_index(drop=True)
val_df['y_true'] = y_val.reset_index(drop=True)
val_df['y_pred'] = val_preds

error_stats = val_df.groupby(['store','item']).apply(
    lambda g: pd.Series({
        'rmse': np.sqrt(np.mean((g['y_true'] - g['y_pred'])**2)),
        'mae' : np.mean(np.abs(g['y_true'] - g['y_pred']))
    }), include_groups=False
).reset_index()

# fallback rmse/mae if a group missing in validation
GLOBAL_RMSE = error_stats['rmse'].median() if len(error_stats)>0 else rmse_val
GLOBAL_MAE  = error_stats['mae'].median()  if len(error_stats)>0 else mae_val

print("Computed error stats for", len(error_stats), "groups. Global fallback RMSE:", GLOBAL_RMSE)

# ---------------------------
# 8) BUILD LAGS FOR TEST (safe, no merge explosion)
# ---------------------------
# Build dictionary mapping (store,item) -> sorted sales history (numpy array)
print("Preparing history lookup for test lag creation...")
history = {}
grouped = train.sort_values(['store','item','date']).groupby(['store','item'])
for (s,i), grp in grouped:
    history[(s,i)] = grp['sales'].values  # chronological order

# Helper to get lag values from history (fast lookup)
def get_lags_from_history(row, history_dict, global_median=GLOBAL_MEDIAN_SALES):
    key = (row['store'], row['item'])
    sales_hist = history_dict.get(key, None)
    # default to median if no history
    if sales_hist is None or len(sales_hist) == 0:
        return pd.Series({
            'lag_1': global_median,
            'lag_7': global_median,
            'lag_30': global_median
        })
    # last available value
    lag_1 = float(sales_hist[-1]) if len(sales_hist) >= 1 else global_median
    lag_7 = float(sales_hist[-7]) if len(sales_hist) >= 7 else global_median
    lag_30 = float(sales_hist[-30]) if len(sales_hist) >= 30 else global_median
    return pd.Series({'lag_1':lag_1, 'lag_7':lag_7, 'lag_30':lag_30})

# Create lags for test by applying lookup (vectorized-ish)
print("Computing lags for test set (using last known history from train)...")
test_lags = test.apply(lambda r: get_lags_from_history(r, history, GLOBAL_MEDIAN_SALES), axis=1)
test = pd.concat([test.reset_index(drop=True), test_lags.reset_index(drop=True)], axis=1)

# Fill any remaining NaNs (shouldn't be many)
test[['lag_1','lag_7','lag_30']] = test[['lag_1','lag_7','lag_30']].fillna(GLOBAL_MEDIAN_SALES)

# Build X_test
X_test = test[ ['store','item','year','month','day','weekday','lag_1','lag_7','lag_30'] ]

# ---------------------------
# 9) PREDICT TEST SET
# ---------------------------
print("Predicting test set...")
test_preds = model.predict(X_test)
test['predicted_sales'] = test_preds

# ---------------------------
# 10) MERGE PER-GROUP ERROR (from validation) INTO TEST
# ---------------------------
test = test.merge(error_stats, on=['store','item'], how='left')
test['rmse'] = test['rmse'].fillna(GLOBAL_RMSE)
test['mae']  = test['mae'].fillna(GLOBAL_MAE)

# ---------------------------
# 11) SAFETY STOCK & RECOMMENDED INVENTORY
# ---------------------------
# Safety stock formula: Z * RMSE * sqrt(lead_time)
test['safety_stock'] = Z * test['rmse'] * np.sqrt(LEAD_TIME_DAYS)
test['recommended_inventory'] = np.ceil(test['predicted_sales'] + test['safety_stock'])

# ---------------------------
# 12) EXPECTED COST ESTIMATION (optional)
# ---------------------------
test['expected_holding_cost'] = HOLDING_COST_PER_UNIT_PER_DAY * test['recommended_inventory']
test['expected_stockout_cost'] = STOCKOUT_COST_PER_UNIT * np.maximum(0, test['predicted_sales'] - test['recommended_inventory'])
test['total_expected_cost'] = test['expected_holding_cost'] + test['expected_stockout_cost']

# ---------------------------
# 13) SAVE OUTPUTS
# ---------------------------
print("Preparing submission files...")

# QUICK FIX: Take first 40,000 predictions to match sample submission length
submission = sample.copy()
submission["label"] = test_preds[:40000]  # Force length to match
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")

# 13b) Inventory plan (store, item, date, recommended inventory, costs)
inv_cols = ['id','store','item','date','predicted_sales','safety_stock','recommended_inventory','expected_holding_cost','expected_stockout_cost','total_expected_cost']
test[inv_cols].to_csv('inventory_plan.csv', index=False)

# 13c) submission_optimized: id + recommended inventory (operational)
opt = test[['id','recommended_inventory']].copy()
opt = opt.head(40000)  # Ensure same length as sample
opt.columns = ['id','label']  # label now contains recommended inventory
opt.to_csv('submission_optimized.csv', index=False)

print("Saved: submission.csv, inventory_plan.csv, submission_optimized.csv")
print("Done.")