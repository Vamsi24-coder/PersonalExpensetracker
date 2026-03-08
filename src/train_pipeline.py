import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import init_db, get_session, User, Expense
from preprocessing import load_data, aggregate_by_day, feature_engineering
from ml.xgboost_expense_prediction import XGBoostPredictor
from ml.lstm_forecasting import LSTMPredictor
from ml.expense_classifier import ExpenseClassifier
from ml.anomaly_detection import AnomalyDetector
from ml.clustering import SpendingClustering
from ml.budget_engine import AdvancedBudgetEngine

def run_training_pipeline():
    print("🚀 Starting Advanced ML Training Pipeline...")
    
    # 1. Load Data
    df = load_data()
    if df.empty:
        print("❌ No data found in database. Please generate synthetic data first.")
        return
    
    # 2. Daily Aggregation for Time-Series
    daily_df = aggregate_by_day(df)
    
    # 3. XGBoost Training
    print("\n--- Training XGBoost Predictor ---")
    xgb_predictor = XGBoostPredictor()
    xgb_predictor.train(daily_df)
    
    # 4. LSTM Training
    print("\n--- Training LSTM Forecasting ---")
    lstm_predictor = LSTMPredictor()
    lstm_predictor.train(daily_df, epochs=30)
    
    # 5. SVM Classification Training
    print("\n--- Training SVM Expense Classifier ---")
    classifier = ExpenseClassifier()
    classifier.train(df)
    
    # 6. Anomaly Detection (Isolation Forest)
    print("\n--- Initializing Anomaly Detector ---")
    feat_df = feature_engineering(df)
    detector = AnomalyDetector()
    detector.train(feat_df)
    
    # 7. User Behavior Clustering (K-Means)
    print("\n--- Training Spending Clustering ---")
    clustering = SpendingClustering()
    clustering.train(df)
    
    # 8. Budget Recommendations (ElasticNet)
    print("\n--- Generating Budget Recommendations ---")
    budget_engine = AdvancedBudgetEngine()
    recs = budget_engine.recommend_budgets(df, user_id=1) # Default user
    budget_engine.save_recommendations(recs)
    
    print("\n✅ Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_training_pipeline()
