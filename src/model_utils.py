import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    'DayOfWeek', 'DOW_sin', 'DOW_cos', 'Month_sin', 'Month_cos',
    'DayOfMonth', 'WeekOfYear', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'Sales_lag1', 'Sales_lag7', 'Sales_roll7', 'Sales_roll30'
]


def add_features(data):
    """Add calendar + lag + rolling features. All lag/rolling features look
    only backwards (shift before rolling) so there is no data leakage."""
    data = data.copy()
    data['Month']      = data['Date'].dt.month
    data['DayOfMonth'] = data['Date'].dt.day
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)

    # Cyclic encoding so the model understands e.g. Dec -> Jan continuity
    data['DOW_sin']   = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DOW_cos']   = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    data['StateHoliday']  = (data['StateHoliday'] != '0').astype(int)
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(int)

    data['Sales_lag1']   = data['Sales'].shift(1)
    data['Sales_lag7']   = data['Sales'].shift(7)
    data['Sales_roll7']  = data['Sales'].shift(1).rolling(7).mean()
    data['Sales_roll30'] = data['Sales'].shift(1).rolling(30).mean()

    return data


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, MAPE. Closed days (Sales=0) are filtered upstream
    so MAPE never divides by zero."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}


def train_and_save_model(store_id=1):
    dtype = {'StateHoliday': str, 'SchoolHoliday': str}
    df       = pd.read_csv('src/data/train.csv', dtype=dtype, parse_dates=['Date'], low_memory=False)
    store_df = pd.read_csv('src/data/store.csv')
    df = pd.merge(df, store_df, on='Store', how='left')

    # Filter closed days — Sales=0 rows break MAPE and confuse the model
    data = df[(df['Store'] == store_id) & (df['Open'] == 1)].sort_values('Date').reset_index(drop=True)

    # Keep last 90 days (including closed days) for the agent to reference
    df[df['Store'] == store_id].sort_values('Date').tail(90).to_csv('src/data/recentdata.csv', index=False)

    data = add_features(data)
    model_df = data[FEATURE_COLUMNS + ['Sales']].dropna().reset_index(drop=True)

    X, y = model_df[FEATURE_COLUMNS], model_df['Sales']

    # Time-based split — never shuffle a time series
    split = int(len(model_df) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    preds   = model.predict(X_test)
    metrics = compute_metrics(y_test.values, preds)
    print(f"✅ Model trained! RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%")

    with open('src/salesmodel.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('src/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    train_and_save_model()
