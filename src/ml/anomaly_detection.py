import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

class AnomalyDetector:
    def __init__(self, contamination=0.03):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False

    def prepare_features(self, df):
        temp_df = df.copy()
        # Numerical features for anomaly detection
        # amount, day_of_week, rolling_mean_7d, etc.
        features = ['amount', 'day_of_week']
        if 'rolling_mean_7d' in temp_df.columns:
            features.append('rolling_mean_7d')
        if 'rolling_std_7d' in temp_df.columns:
            features.append('rolling_std_7d')
            
        X = temp_df[features].fillna(0)
        return X

    def train(self, df):
        X = self.prepare_features(df)
        if len(X) < 10:
             return
        self.model.fit(X)
        self.is_trained = True

    def detect(self, df):
        if not self.is_trained:
            self.train(df)
            
        X = self.prepare_features(df)
        preds = self.model.predict(X)
        # Isolation Forest returns -1 for anomalies
        anomalies = df[preds == -1]
        return anomalies

    def detect_zscore(self, df, threshold=3):
        # Basic Z-score for single transaction amount spikes
        mean = df['amount'].mean()
        std = df['amount'].std()
        anomalies = df[(df['amount'] - mean).abs() > threshold * std]
        return anomalies

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import load_data, feature_engineering
    
    data = load_data()
    if not data.empty:
        data = feature_engineering(data)
        detector = AnomalyDetector()
        anomalies = detector.detect(data)
        print(f"Detected {len(anomalies)} Isolation Forest anomalies.")
        print(anomalies.head())
