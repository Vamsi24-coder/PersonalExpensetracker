import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SpendingClustering:
    def __init__(self, n_clusters=3):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'spending_clusters.pkl')
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_names = {
            0: "Minimal Spenders",
            1: "Balanced Spenders",
            2: "Luxury Spenders"
        }
        self.is_trained = False

    def prepare_features(self, df):
        # Aggregate features per user_id
        # avg_monthly_expense, category_distribution, expense_variance
        user_groups = df.groupby('user_id').agg(
            avg_monthly_expense=('amount', 'mean'),
            expense_variance=('amount', 'std'),
            total_transactions=('expense_id', 'count')
        ).fillna(0)
        
        # Category distribution
        cat_dist = df.pivot_table(index='user_id', columns='category', values='amount', aggfunc='count', fill_value=0)
        # Normalize
        cat_dist = cat_dist.div(cat_dist.sum(axis=1), axis=0)
        
        features = pd.concat([user_groups, cat_dist], axis=1).fillna(0)
        return features

    def train(self, df):
        if df.empty: return
        X = self.prepare_features(df)
        if len(X) < 3:
             return
             
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'features': X.columns.tolist()}, self.model_path)
        print(f"K-Means Clustering model saved to {self.model_path}")

    def predict(self, user_df):
        if not self.is_trained:
            if os.path.exists(self.model_path):
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
            else:
                return "Unknown"

        X = self.prepare_features(user_df)
        X_scaled = self.scaler.transform(X)
        cluster_id = self.model.predict(X_scaled)[0]
        return self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import load_data
    
    data = load_data()
    if not data.empty:
        clustering = SpendingClustering()
        clustering.train(data)
        # For simplicity, just predicting on the entire set as one user group
        print(f"User classification: {clustering.predict(data)}")
