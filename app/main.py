from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from database import get_session, Expense, User, Budget, Anomaly
from ml.nlp_categorizer import NLPCategorizer
from ml.anomaly_detection import AnomalyDetector
from ml.expense_prediction import ExpensePredictor
from preprocessing import load_data, aggregate_by_day

app = FastAPI(title="Personal Expense Prediction API")

# Initialize models
categorizer = NLPCategorizer()
categorizer.load()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Personal Expense Prediction API"}

@app.post("/categorize")
def categorize_expense(description: str):
    category = categorizer.predict(description)
    return {"description": description, "category": category}

@app.get("/anomalies/{user_id}")
def get_anomalies(user_id: int):
    session = get_session()
    anomalies = session.query(Anomaly).filter(Anomaly.user_id == user_id).all()
    session.close()
    return anomalies

@app.get("/predictions/{user_id}")
def get_predictions(user_id: int):
    data = load_data()
    if data.empty:
        raise HTTPException(status_code=404, detail="No data found for predictions")
    
    daily_data = aggregate_by_day(data)
    predictor = ExpensePredictor()
    predictions = predictor.predict_next_days(daily_data)
    
    if predictions is None:
        return {"message": "Model not trained yet"}
        
    return predictions.to_dict(orient='records')

@app.get("/health_score/{user_id}")
def get_health_score(user_id: int):
    session = get_session()
    user = session.query(User).filter(User.user_id == user_id).first()
    if not user:
        session.close()
        raise HTTPException(status_code=404, detail="User not found")
        
    # Simplify: health score = (income - total_expenses_last_month) / income * 100
    # For demo, returning a calculated dummy
    import datetime
    last_month = datetime.date.today().replace(day=1) - datetime.timedelta(days=1)
    # logic to sum expenses...
    session.close()
    return {"status": "Good", "score": 75}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
