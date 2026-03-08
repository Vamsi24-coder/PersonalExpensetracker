import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import datetime

class ExpensePredictor:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', f'expense_predictor_{model_type}.joblib')
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.is_trained = False

    def prepare_features(self, daily_df):
        df = daily_df.copy()
        # Ensure dates are datetime
        df['expense_date'] = pd.to_datetime(df['expense_date'])
        
        # Lag features
        df['lag_1'] = df['amount'].shift(1)
        df['lag_7'] = df['amount'].shift(7)
        df['lag_30'] = df['amount'].shift(30)
        
        # Drop rows with NaN from shifts
        df = df.dropna()
        
        features = ['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_30d']
        X = df[features]
        y = df['amount']
        return X, y

    def train(self, daily_df):
        if len(daily_df) < 40:
            print("Not enough data to train prediction model.")
            return None
            
        X, y = self.prepare_features(daily_df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluation
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_absolute_error(y_test, y_pred)) # Simplification
        }
        
        joblib.dump(self.model, self.model_path)
        print(f"Model trained. Metrics: {metrics}")
        return metrics

    def predict_next_days(self, last_df, num_days=30):
        """Predicts future expenses day by day."""
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
            else:
                return None

        # This is a recursive prediction approach
        predictions = []
        current_df = last_df.tail(40).copy() # Need at least 30 days for lags
        
        start_date = current_df['expense_date'].max()
        
        for i in range(num_days):
            next_date = start_date + datetime.timedelta(days=i+1)
            
            # Prepare feature for next day
            features = pd.DataFrame([{
                'day_of_week': next_date.weekday(),
                'month': next_date.month,
                'is_weekend': 1 if next_date.weekday() >= 5 else 0,
                'lag_1': current_df.iloc[-1]['amount'],
                'lag_7': current_df.iloc[-7]['amount'],
                'lag_30': current_df.iloc[-30]['amount'],
                'rolling_mean_30d': current_df['amount'].tail(30).mean()
            }])
            
            pred_amount = self.model.predict(features)[0]
            predictions.append({'date': next_date, 'predicted_amount': max(0, pred_amount)})
            
            # Update current_df for next iteration
            new_row = pd.DataFrame([{
                'expense_date': next_date,
                'amount': pred_amount,
                'day_of_week': next_date.weekday(),
                'is_weekend': 1 if next_date.weekday() >= 5 else 0
            }])
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            
        return pd.DataFrame(predictions)

if __name__ == "__main__":
    from preprocessing import load_data, aggregate_by_day
    data = load_data()
    if not data.empty:
        daily_data = aggregate_by_day(data)
        predictor = ExpensePredictor()
        predictor.train(daily_data)
        future = predictor.predict_next_days(daily_data)
        if future is not None:
            print("Future predictions:")
            print(future.head())
