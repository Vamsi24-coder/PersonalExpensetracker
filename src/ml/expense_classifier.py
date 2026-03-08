import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

class ExpenseClassifier:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'expense_classifier.pkl')
        self.categories = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Healthcare", "Education", "Others"]
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ])
        self.is_trained = False

    def train(self, data_df):
        if data_df.empty or 'description' not in data_df.columns:
            print("Insufficient data for SVM training.")
            return None
            
        X = data_df['description']
        y = data_df['category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"SVM Expense Classifier saved. Weighted F1: {report['weighted avg']['f1-score']}")
        return report

    def predict(self, description):
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
                self.is_trained = True
            else:
                # Fallback to simple rule-based if not trained
                desc = description.lower()
                if any(k in desc for k in ['food', 'restaurant', 'dinner', 'pizza']): return "Food"
                if any(k in desc for k in ['uber', 'taxi', 'fuel', 'bus']): return "Transport"
                if any(k in desc for k in ['shopping', 'amazon', 'mall']): return "Shopping"
                return "Others"
        
        return self.pipeline.predict([description])[0]

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessing import load_data
    
    data = load_data()
    if not data.empty:
        classifier = ExpenseClassifier()
        classifier.train(data)
        print(f"Prediction for 'Starbucks coffee': {classifier.predict('Starbucks coffee')}")
