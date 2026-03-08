import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import datetime

class LSTMPredictor:
    def __init__(self, sequence_length=7):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'lstm_expense_model.h5')
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False

    def prepare_data(self, daily_df):
        if len(daily_df) < self.sequence_length + 1:
            return None, None
            
        data = daily_df['amount'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, daily_df, epochs=50, batch_size=32):
        X, y = self.prepare_data(daily_df)
        if X is None:
            print("Insufficient data for LSTM training.")
            return None
            
        self.build_model()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.is_trained = True
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"LSTM Model saved to {self.model_path}")
        return True

    def predict_future(self, daily_df, num_days=30):
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                # Need to refit scaler on original data
                self.scaler.fit(daily_df['amount'].values.reshape(-1, 1))
                self.is_trained = True
            else:
                return None

        last_sequence = daily_df['amount'].values[-self.sequence_length:].reshape(-1, 1)
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        predictions = []
        current_seq = last_sequence_scaled.reshape(1, self.sequence_length, 1)
        
        start_date = daily_df['expense_date'].max()
        
        for i in range(num_days):
            pred_scaled = self.model.predict(current_seq, verbose=0)
            pred_val = self.scaler.inverse_transform(pred_scaled)[0][0]
            pred_val = max(0, float(pred_val))
            
            next_date = start_date + datetime.timedelta(days=i+1)
            predictions.append({'date': next_date, 'predicted_amount': pred_val})
            
            # Update sequence
            next_val_scaled = pred_scaled.reshape(1, 1, 1)
            current_seq = np.append(current_seq[:, 1:, :], next_val_scaled, axis=1)
            
        return pd.DataFrame(predictions)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import load_data, aggregate_by_day
    
    data = load_data()
    if not data.empty:
        daily = aggregate_by_day(data)
        predictor = LSTMPredictor()
        predictor.train(daily)
        res = predictor.predict_future(daily)
        if res is not None:
             print(res.head())
