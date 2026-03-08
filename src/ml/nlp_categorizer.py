import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

class NLPCategorizer:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'nlp_categorizer.joblib')
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        self.is_trained = False

    def train(self, descriptions, categories):
        if not descriptions or not categories:
            return
        self.pipeline.fit(descriptions, categories)
        self.is_trained = True
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model trained and saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            self.is_trained = True
            return True
        return False

    def predict(self, description):
        if not self.is_trained:
            if not self.load():
                return "Other"
        return self.pipeline.predict([description])[0]

# Sample training data
TRAINING_DATA = [
    ("Paid 450 for pizza", "Food"),
    ("Dinner at Italian restaurant", "Food"),
    ("Uber ride to office", "Transport"),
    ("Gas station refueling", "Transport"),
    ("Netflix monthly subscription", "Subscription"),
    ("Movie tickets for TWO", "Entertainment"),
    ("Grocery shopping at Walmart", "Shopping"),
    ("New pair of shoes", "Shopping"),
    ("Electricity bill payment", "Utilities"),
    ("Water bill", "Utilities"),
    ("Doctor consultation fee", "Health"),
    ("Pharmacy medicine", "Health"),
    ("Zomato order", "Food"),
    ("Bus fare", "Transport"),
    ("Spotify premium", "Subscription"),
    ("Concert tickets", "Entertainment"),
    ("Amazon purchase", "Shopping"),
]

if __name__ == "__main__":
    categorizer = NLPCategorizer()
    descriptions = [d for d, c in TRAINING_DATA]
    categories = [c for d, c in TRAINING_DATA]
    
    categorizer.train(descriptions, categories)
    
    # Test
    test_desc = "Ordered burger from Swiggy"
    print(f"Prediction for '{test_desc}': {categorizer.predict(test_desc)}")
