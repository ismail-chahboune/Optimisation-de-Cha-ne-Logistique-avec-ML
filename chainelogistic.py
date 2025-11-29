

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 8


SERVICE_LEVEL = 0.95   
Z = 1.65               
LEAD_TIME_DAYS = 7    
HOLDING_COST_PER_UNIT_PER_DAY = 0.5
STOCKOUT_COST_PER_UNIT = 2.0


print("Loading data...")
train = pd.read_csv("train.csv")  
test  = pd.read_csv("test.csv")    
sample = pd.read_csv("sample_submission.csv")  

print("Train shape:", train.shape, "Test shape:", test.shape, "Sample shape:", sample.shape)


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

for df in (train, test):
    df['year']    = df['date'].dt.year
    df['month']   = df['date'].dt.month
    df['day']     = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday


GLOBAL_MEDIAN_SALES = train['sales'].median()


def add_lags_to_df(df, lags=[1,7,30], group_cols=['store','item'], date_col='date', value_col='sales'):
    df = df.sort_values(by=group_cols + [date_col])
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[value_col].shift(lag)
    return df

train_with_lags = add_lags_to_df(train.copy(), lags=[1,7,30])

train_with_lags = train_with_lags.dropna(subset=['lag_1','lag_7','lag_30']).reset_index(drop=True)
print("Train with lags shape (after dropna):", train_with_lags.shape)


FEATURES = ['store','item','year','month','day','weekday','lag_1','lag_7','lag_30']
X = train_with_lags[FEATURES]
y = train_with_lags['sales']



split_idx = int(len(X) * 0.90)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train rows:", len(X_train), "Val rows:", len(X_val))


print("Training ExtraTreesRegressor...")
model = ExtraTreesRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)


print("Evaluating on validation set...")
val_preds = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))
mae_val = mean_absolute_error(y_val, val_preds)
print(f"Validation RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")


val_df = X_val.copy().reset_index(drop=True)
val_df['y_true'] = y_val.reset_index(drop=True)
val_df['y_pred'] = val_preds

error_stats = val_df.groupby(['store','item']).apply(
    lambda g: pd.Series({
        'rmse': np.sqrt(np.mean((g['y_true'] - g['y_pred'])**2)),
        'mae' : np.mean(np.abs(g['y_true'] - g['y_pred']))
    }), include_groups=False
).reset_index()


GLOBAL_RMSE = error_stats['rmse'].median() if len(error_stats)>0 else rmse_val
GLOBAL_MAE  = error_stats['mae'].median()  if len(error_stats)>0 else mae_val

print("Computed error stats for", len(error_stats), "groups. Global fallback RMSE:", GLOBAL_RMSE)



print("Preparing history lookup for test lag creation...")
history = {}
grouped = train.sort_values(['store','item','date']).groupby(['store','item'])
for (s,i), grp in grouped:
    history[(s,i)] = grp['sales'].values  


def get_lags_from_history(row, history_dict, global_median=GLOBAL_MEDIAN_SALES):
    key = (row['store'], row['item'])
    sales_hist = history_dict.get(key, None)
   
    if sales_hist is None or len(sales_hist) == 0:
        return pd.Series({
            'lag_1': global_median,
            'lag_7': global_median,
            'lag_30': global_median
        })
    
    lag_1 = float(sales_hist[-1]) if len(sales_hist) >= 1 else global_median
    lag_7 = float(sales_hist[-7]) if len(sales_hist) >= 7 else global_median
    lag_30 = float(sales_hist[-30]) if len(sales_hist) >= 30 else global_median
    return pd.Series({'lag_1':lag_1, 'lag_7':lag_7, 'lag_30':lag_30})


print("Computing lags for test set (using last known history from train)...")
test_lags = test.apply(lambda r: get_lags_from_history(r, history, GLOBAL_MEDIAN_SALES), axis=1)
test = pd.concat([test.reset_index(drop=True), test_lags.reset_index(drop=True)], axis=1)


test[['lag_1','lag_7','lag_30']] = test[['lag_1','lag_7','lag_30']].fillna(GLOBAL_MEDIAN_SALES)


X_test = test[ ['store','item','year','month','day','weekday','lag_1','lag_7','lag_30'] ]


print("Predicting test set...")
test_preds = model.predict(X_test)
test['predicted_sales'] = test_preds


test = test.merge(error_stats, on=['store','item'], how='left')
test['rmse'] = test['rmse'].fillna(GLOBAL_RMSE)
test['mae']  = test['mae'].fillna(GLOBAL_MAE)



test['safety_stock'] = Z * test['rmse'] * np.sqrt(LEAD_TIME_DAYS)
test['recommended_inventory'] = np.ceil(test['predicted_sales'] + test['safety_stock'])


test['expected_holding_cost'] = HOLDING_COST_PER_UNIT_PER_DAY * test['recommended_inventory']
test['expected_stockout_cost'] = STOCKOUT_COST_PER_UNIT * np.maximum(0, test['predicted_sales'] - test['recommended_inventory'])
test['total_expected_cost'] = test['expected_holding_cost'] + test['expected_stockout_cost']


print("Preparing submission files...")


submission = sample.copy()
submission["label"] = test_preds[:40000]  
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved as submission.csv")


inv_cols = ['id','store','item','date','predicted_sales','safety_stock','recommended_inventory','expected_holding_cost','expected_stockout_cost','total_expected_cost']
test[inv_cols].to_csv('inventory_plan.csv', index=False)


opt = test[['id','recommended_inventory']].copy()
opt = opt.head(40000)  
opt.columns = ['id','label']  
opt.to_csv('submission_optimized.csv', index=False)

print("Saved: submission.csv, inventory_plan.csv, submission_optimized.csv")
print("Done.")
