

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
 
# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/ismailchahboune8/chaine-logistique"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"
OUT_DIR = "/kaggle/working"
 
RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 8
 
LAGS = [1, 7, 30]
VAL_DAYS = 90                      # validation window = same horizon as the test set
FEATURES = ["store", "item", "year", "month", "day", "weekday"] + [f"lag_{l}" for l in LAGS]
 
SERVICE_LEVEL = 0.95
Z = norm.ppf(SERVICE_LEVEL)       
LEAD_TIME_DAYS = 7
HOLDING_COST_PER_UNIT_PER_DAY = 0.5
STOCKOUT_COST_PER_UNIT = 2.0
 
 
# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def add_calendar(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    return df
 
 
def add_lags(df):
    df = df.sort_values(["store", "item", "date"]).copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)
    return df
 
 
def make_model():
    return ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
 
 
def recursive_forecast(model, wide, dates):
    """Forecast day by day. Each day's prediction becomes a lag for the next days,
    so lag_1 / lag_7 / lag_30 are always correct (no frozen lags over the horizon)."""
    wide = wide.copy()
    stores = wide.columns.get_level_values("store").to_numpy()
    items = wide.columns.get_level_values("item").to_numpy()
    out = []
    for d in dates:
        X_day = pd.DataFrame({
            "store": stores, "item": items,
            "year": d.year, "month": d.month, "day": d.day, "weekday": d.weekday(),
        })
        for lag in LAGS:
            X_day[f"lag_{lag}"] = wide.loc[d - pd.Timedelta(days=lag)].to_numpy()
        preds = np.clip(model.predict(X_day[FEATURES]), 0, None)
        wide.loc[d] = preds
        out.append(pd.DataFrame({"date": d, "store": stores, "item": items, "predicted_sales": preds}))
    return pd.concat(out, ignore_index=True)
 
 
def forward_lead_demand(df, col):
    """Total of `col` over the next LEAD_TIME_DAYS days (including today), per store/item.
    df must be sorted by store, item, date."""
    return df.groupby(["store", "item"])[col].transform(
        lambda s: s[::-1].rolling(LEAD_TIME_DAYS, min_periods=1).mean()[::-1]
    ) * LEAD_TIME_DAYS
 
 
# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
print("Files found:")
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        print("  ", os.path.join(root, f))
 
train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
test = pd.read_csv(TEST_PATH, parse_dates=["date"])
sample = pd.read_csv(SAMPLE_PATH)
 
print("\nTrain:", train.shape, "| Test:", test.shape, "| Sample:", sample.shape)
print("Sample submission columns:", list(sample.columns))
print("Stores:", train["store"].nunique(), "| Items:", train["item"].nunique())
print("Train dates:", train["date"].min().date(), "->", train["date"].max().date())
print("Test dates :", test["date"].min().date(), "->", test["date"].max().date())
 
train = add_calendar(train)
test = add_calendar(test)
 
# One column per (store, item), one row per day. Test days are NaN (to be forecast).
all_dates = pd.date_range(train["date"].min(), test["date"].max(), freq="D")
wide = (train.pivot(index="date", columns=["store", "item"], values="sales")
             .reindex(all_dates))
 
train_lags = add_lags(train).dropna(subset=[f"lag_{l}" for l in LAGS])
print("Train rows with lags:", len(train_lags))
 
# ----------------------------------------------------------------------------
# 2. HONEST VALIDATION (time-based, recursive, 90 days)
# ----------------------------------------------------------------------------
last_train_date = train["date"].max()
cutoff = last_train_date - pd.Timedelta(days=VAL_DAYS)
val_dates = pd.date_range(cutoff + pd.Timedelta(days=1), last_train_date, freq="D")
print(f"\nValidation: train up to {cutoff.date()}, forecast {len(val_dates)} days after")
 
model_val = make_model()
fit_part = train_lags[train_lags["date"] <= cutoff]
model_val.fit(fit_part[FEATURES], fit_part["sales"])
 
wide_val = wide.copy()
wide_val.loc[val_dates] = np.nan            # hide the validation period
val_pred = recursive_forecast(model_val, wide_val, val_dates)
 
actual = train.loc[train["date"].isin(val_dates), ["date", "store", "item", "weekday", "sales"]]
val = actual.merge(val_pred, on=["date", "store", "item"], how="left")
 
# Simple baseline: average of the same weekday over the last 8 weeks before the cutoff
hist = train[(train["date"] > cutoff - pd.Timedelta(days=56)) & (train["date"] <= cutoff)]
baseline = (hist.groupby(["store", "item", "weekday"])["sales"].mean()
                .rename("baseline").reset_index())
val = val.merge(baseline, on=["store", "item", "weekday"], how="left")
 
rmse_val = np.sqrt(mean_squared_error(val["sales"], val["predicted_sales"]))
mae_val = mean_absolute_error(val["sales"], val["predicted_sales"])
rmse_base = np.sqrt(mean_squared_error(val["sales"], val["baseline"]))
mae_base = mean_absolute_error(val["sales"], val["baseline"])
smape = 100 * np.mean(2 * np.abs(val["sales"] - val["predicted_sales"])
                      / (np.abs(val["sales"]) + np.abs(val["predicted_sales"]) + 1e-9))
 
print("\n=== VALIDATION RESULTS (90-day recursive forecast) ===")
print(f"Model    -> RMSE: {rmse_val:.3f} | MAE: {mae_val:.3f} | SMAPE: {smape:.2f}%")
print(f"Baseline -> RMSE: {rmse_base:.3f} | MAE: {mae_base:.3f}")
print(f"RMSE improvement vs baseline: {100 * (rmse_base - rmse_val) / rmse_base:.1f}%")
 
# Error per store/item (used for safety stock)
val["sq_err"] = (val["sales"] - val["predicted_sales"]) ** 2
val["abs_err"] = (val["sales"] - val["predicted_sales"]).abs()
error_stats = (val.groupby(["store", "item"])
                  .agg(rmse=("sq_err", lambda x: np.sqrt(x.mean())), mae=("abs_err", "mean"))
                  .reset_index())
GLOBAL_RMSE = error_stats["rmse"].median()
GLOBAL_MAE = error_stats["mae"].median()
 
# ----------------------------------------------------------------------------
# 3. INVENTORY POLICY CHECK ON VALIDATION (using real sales)
# ----------------------------------------------------------------------------
val = val.sort_values(["store", "item", "date"]).reset_index(drop=True)
val["lt_pred"] = forward_lead_demand(val, "predicted_sales")
val["lt_actual"] = forward_lead_demand(val, "sales")
 
# Error of the lead-time demand forecast itself (captures the correlation between days).
# Only windows that are fully inside the validation period are used.
full_window = val["date"] <= last_train_date - pd.Timedelta(days=LEAD_TIME_DAYS - 1)
val["lt_err2"] = (val["lt_actual"] - val["lt_pred"]) ** 2
lt_err = (val.loc[full_window].groupby(["store", "item"])["lt_err2"].mean()
             .pow(0.5).rename("sigma_lt").reset_index())
error_stats = error_stats.merge(lt_err, on=["store", "item"], how="left")
GLOBAL_SIGMA_LT = error_stats["sigma_lt"].median()
 
val = val.merge(lt_err, on=["store", "item"], how="left")
val["safety_stock"] = Z * val["sigma_lt"]
 
 
def evaluate_policy(inventory, label):
    leftover = np.maximum(0, inventory - val["lt_actual"])
    shortage = np.maximum(0, val["lt_actual"] - inventory)
    holding = HOLDING_COST_PER_UNIT_PER_DAY * leftover
    stockout = STOCKOUT_COST_PER_UNIT * shortage
    print(f"{label:<28} service level: {100 * (shortage == 0).mean():5.1f}% | "
          f"avg holding cost: {holding.mean():6.2f} | avg stockout cost: {stockout.mean():6.2f} | "
          f"avg total: {(holding + stockout).mean():6.2f}")
 
 
print(f"\n=== INVENTORY POLICY CHECK (target service level {SERVICE_LEVEL:.0%}) ===")
evaluate_policy(np.ceil(val["lt_pred"]), "Forecast only (no safety)")
evaluate_policy(np.ceil(val["lt_pred"] + val["safety_stock"]), "Forecast + safety stock")
 
# ----------------------------------------------------------------------------
# 4. FINAL MODEL -> FORECAST THE TEST PERIOD
# ----------------------------------------------------------------------------
print("\nTraining final model on all training data")
model = make_model()
model.fit(train_lags[FEATURES], train_lags["sales"])
 
test_dates = pd.DatetimeIndex(sorted(test["date"].unique()))
test_pred = recursive_forecast(model, wide, test_dates)
test = test.merge(test_pred, on=["date", "store", "item"], how="left")
assert test["predicted_sales"].notna().all(), "Some test rows have no prediction"
 
# ----------------------------------------------------------------------------
# 5. INVENTORY PLAN FOR THE TEST PERIOD
# ----------------------------------------------------------------------------
test = test.merge(error_stats, on=["store", "item"], how="left")
test["rmse"] = test["rmse"].fillna(GLOBAL_RMSE)
test["mae"] = test["mae"].fillna(GLOBAL_MAE)
test["sigma_lt"] = test["sigma_lt"].fillna(GLOBAL_SIGMA_LT)
test = test.sort_values(["store", "item", "date"]).reset_index(drop=True)
 
sigma_lt = test["sigma_lt"]                                        # uncertainty over the lead time
test["lead_time_demand"] = forward_lead_demand(test, "predicted_sales")
test["safety_stock"] = Z * sigma_lt
test["recommended_inventory"] = np.ceil(test["lead_time_demand"] + test["safety_stock"])
 

unit_loss = norm.pdf(Z) - Z * (1 - norm.cdf(Z))
test["expected_shortage"] = sigma_lt * unit_loss
test["expected_leftover"] = test["recommended_inventory"] - test["lead_time_demand"] + test["expected_shortage"]
test["expected_holding_cost"] = HOLDING_COST_PER_UNIT_PER_DAY * test["expected_leftover"]
test["expected_stockout_cost"] = STOCKOUT_COST_PER_UNIT * test["expected_shortage"]
test["total_expected_cost"] = test["expected_holding_cost"] + test["expected_stockout_cost"]
 
# ----------------------------------------------------------------------------
# 6. SAVE OUTPUTS
# ----------------------------------------------------------------------------
test = test.sort_values("id").reset_index(drop=True)
 
target_col = sample.columns[1]                                    # For example : "label" or "sales"
if len(sample) == len(test) and sample["id"].dtype == test["id"].dtype:
    submission = sample[["id"]].merge(test[["id", "predicted_sales"]], on="id", how="left")
else:
    print(f"WARNING: sample_submission ({len(sample)} rows, id type {sample['id'].dtype}) does not match "
          f"test ({len(test)} rows, id type {test['id'].dtype}). Saving forecasts with the test ids instead.")
    submission = test[["id", "predicted_sales"]].copy()
submission = submission.rename(columns={"predicted_sales": target_col})
submission.to_csv(f"{OUT_DIR}/submission.csv", index=False)
 
plan_cols = ["id", "store", "item", "date", "predicted_sales", "lead_time_demand",
             "safety_stock", "recommended_inventory", "expected_holding_cost",
             "expected_stockout_cost", "total_expected_cost"]
test[plan_cols].to_csv(f"{OUT_DIR}/inventory_plan.csv", index=False)
 
print("\nSaved: submission.csv, inventory_plan.csv")
print(test[plan_cols].head())
print("\n=== SUMMARY TO COPY ===")
print(f"Train rows: {len(train)} | Stores: {train['store'].nunique()} | Items: {train['item'].nunique()}")
print(f"Validation RMSE: {rmse_val:.3f} | MAE: {mae_val:.3f} | SMAPE: {smape:.2f}% "
      f"| RMSE improvement vs baseline: {100 * (rmse_base - rmse_val) / rmse_base:.1f}%")
print("Done.")
