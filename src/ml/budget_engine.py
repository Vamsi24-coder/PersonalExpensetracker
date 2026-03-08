import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import joblib
import os
from database import get_session, Budget, Expense

class AdvancedBudgetEngine:
    def __init__(self, alpha=0.1, l1_ratio=0.5):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'elasticnet_budget_model.pkl')
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        self.is_trained = False

    def prepare_features(self, df):
        # Features: monthly_income, avg_category_expense, expense_volatility, etc.
        # This is a simplification; in a real scenario, we'd need a labeled dataset of 'ideal' budgets.
        # Since we don't have labeled 'ideal' budgets, we use current spending as a proxy for target.
        category_stats = df.groupby('category').agg(
            avg_expense=('amount', 'mean'),
            expense_volatility=('amount', 'std')
        ).fillna(0)
        
        category_stats['income_ratio'] = 0.2 # Placeholder
        X = category_stats[['avg_expense', 'expense_volatility', 'income_ratio']]
        return X, category_stats

    def recommend_budgets(self, df, user_id):
        if df.empty: return []
        
        _, stats = self.prepare_features(df)
        
        recommendations = []
        for cat, row in stats.iterrows():
            # ElasticNet style recommendation logic
            # Final budget = avg + alpha * std (simplified)
            recommended_limit = row['avg_expense'] + 1.2 * row['expense_volatility']
            recommended_limit = round(recommended_limit / 100) * 100 # Round to nearest 100
            
            recommendations.append({
                "user_id": user_id,
                "category": cat,
                "recommended_budget": float(recommended_limit)
            })
            
        return recommendations

    def save_recommendations(self, recommendations):
        session = get_session()
        try:
            for rec in recommendations:
                # Update or Insert
                existing = session.query(Budget).filter(Budget.user_id == rec['user_id'], Budget.category == rec['category']).first()
                if existing:
                    existing.recommended_budget = rec['recommended_budget']
                else:
                    new_budget = Budget(**rec)
                    session.add(new_budget)
            session.commit()
        finally:
            session.close()

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import load_data
    
    data = load_data()
    if not data.empty:
        engine = AdvancedBudgetEngine()
        recs = engine.recommend_budgets(data, user_id=1)
        print("Budget Recommendations:")
        for r in recs:
            print(f"{r['category']}: ₹{r['limit_amount']}")
