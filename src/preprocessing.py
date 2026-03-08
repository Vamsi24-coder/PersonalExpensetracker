import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

# Assuming database.py ENGINE_URL is accessible
from database import ENGINE_URL

def load_data():
    engine = create_engine(ENGINE_URL)
    query = "SELECT * FROM expenses"
    df = pd.read_sql(query, engine)
    df['expense_date'] = pd.to_datetime(df['expense_date'])
    return df

def feature_engineering(df):
    if df.empty:
        return df
    
    # Sort by date
    df = df.sort_values('expense_date')
    
    # Time features
    df['day_of_week'] = df['expense_date'].dt.dayofweek
    df['month'] = df['expense_date'].dt.month
    df['day'] = df['expense_date'].dt.day
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Category encoding (one-hot or label)
    # For many models, we might want to aggregate by day or category
    
    # Rolling averages by category
    df['rolling_mean_7d'] = df.groupby('category')['amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['rolling_std_7d'] = df.groupby('category')['amount'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)
    
    # Trend: current amount vs rolling mean
    df['expense_trend'] = df['amount'] / (df['rolling_mean_7d'] + 1) # avoid div by zero
    
    return df

def aggregate_by_day(df):
    """Aggregates expenses by day for time series forecasting."""
    daily_df = df.groupby('expense_date').agg({
        'amount': 'sum',
        'expense_id': 'count'
    }).rename(columns={'expense_id': 'num_transactions'}).reset_index()
    
    daily_df['day_of_week'] = daily_df['expense_date'].dt.dayofweek
    daily_df['month'] = daily_df['expense_date'].dt.month
    daily_df['is_weekend'] = daily_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    daily_df['rolling_mean_30d'] = daily_df['amount'].rolling(window=30, min_periods=1).mean()
    
    return daily_df

if __name__ == "__main__":
    data = load_data()
    if not data.empty:
        features = feature_engineering(data)
        print("Feature engineering complete. Sample data:")
        print(features.head())
    else:
        print("No data found in database.")
