import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

class BehaviorClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def cluster_categories(self, df):
        """Clusters categories based on spending patterns."""
        if df.empty:
            return pd.DataFrame()
            
        # Feature Engineering for Categories
        cat_stats = df.groupby('category').agg({
            'amount': ['mean', 'std', 'count', 'sum']
        }).fillna(0)
        
        cat_stats.columns = ['avg_amount', 'std_amount', 'frequency', 'total_amount']
        
        # Normalize features
        features = ['avg_amount', 'std_amount', 'frequency']
        scaled_features = self.scaler.fit_transform(cat_stats[features])
        
        cat_stats['cluster'] = self.model.fit_predict(scaled_features)
        
        # Label clusters
        # (This is a bit heuristic, usually cluster 0, 1, 2)
        # We can sort clusters by avg_amount to give them meaningful names
        cluster_means = cat_stats.groupby('cluster')['avg_amount'].mean().sort_values()
        
        label_map = {
            cluster_means.index[0]: 'Minimal/Essential',
            cluster_means.index[1]: 'Balanced',
            cluster_means.index[-1]: 'Luxury/High-Spend'
        }
        
        cat_stats['cluster_label'] = cat_stats['cluster'].map(label_map)
        
        return cat_stats.reset_index()

if __name__ == "__main__":
    from preprocessing import load_data
    data = load_data()
    if not data.empty:
        clusterer = BehaviorClusterer()
        clusters = clusterer.cluster_categories(data)
        print("Category Clusters:")
        print(clusters[['category', 'cluster_label', 'avg_amount']])
