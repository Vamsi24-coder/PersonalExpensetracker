import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import datetime

class XGBoostPredictor:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'xgboost_expense_model.pkl')
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        self.is_trained = False

    def prepare_features(self, daily_df):
        df = daily_df.copy()
        df['expense_date'] = pd.to_datetime(df['expense_date'])
        
        # Lag Features
        df['lag_1'] = df['amount'].shift(1)
        df['lag_7'] = df['amount'].shift(7)
        df['lag_30'] = df['amount'].shift(30)
        
        # Rolling Statistics
        df['rolling_mean_30'] = df['amount'].rolling(window=30).mean()
        df['rolling_std_30'] = df['amount'].rolling(window=30).std()
        
        # Drop NaN rows from shifts/rolling
        df = df.dropna()
        
        features = ['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_30', 'rolling_std_30']
        X = df[features]
        y = df['amount']
        return X, y, df

    def train(self, daily_df):
        if len(daily_df) < 60:
            print("Insufficient data for XGBoost training (need 60+ days).")
            return None
            
        X, y, _ = self.prepare_features(daily_df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"XGBoost Model saved. Metrics: {metrics}")
        return metrics

    def predict_future(self, last_df, num_days=30):
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
            else:
                return None

        current_df = last_df.tail(60).copy()
        predictions = []
        
        for i in range(num_days):
            last_date = current_df['expense_date'].max()
            next_date = last_date + datetime.timedelta(days=1)
            
            # Prepare feature for next day
            feat_dict = {
                'day_of_week': next_date.weekday(),
                'month': next_date.month,
                'is_weekend': 1 if next_date.weekday() >= 5 else 0,
                'lag_1': current_df.iloc[-1]['amount'],
                'lag_7': current_df.iloc[-7]['amount'],
                'lag_30': current_df.iloc[-30]['amount'],
                'rolling_mean_30': current_df['amount'].tail(30).mean(),
                'rolling_std_30': current_df['amount'].tail(30).std()
            }
            
            X_next = pd.DataFrame([feat_dict])
            pred_val = max(0, self.model.predict(X_next)[0])
            predictions.append({'date': next_date, 'predicted_amount': pred_val})
            
            # Update current_df
            new_row = pd.DataFrame([{
                'expense_date': next_date,
                'amount': pred_val,
                'day_of_week': next_date.weekday(),
                'month': next_date.month,
                'is_weekend': 1 if next_date.weekday() >= 5 else 0
            }])
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            
        return pd.DataFrame(predictions)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import load_data, aggregate_by_day
    
    data = load_data()
    if not data.empty:
        daily = aggregate_by_day(data)
        predictor = XGBoostPredictor()
        predictor.train(daily)
        res = predictor.predict_future(daily)
        if res is not None:
            print(res.head())
