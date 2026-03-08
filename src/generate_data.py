import random
import datetime
import pandas as pd
from sqlalchemy.orm import Session
from database import init_db, User, Expense, get_session, hash_password

CATEGORIES = {
    "Food": (200, 1000),
    "Transport": (50, 500),
    "Entertainment": (500, 5000),
    "Shopping": (1000, 10000),
    "Utilities": (500, 2000),
    "Health": (200, 5000),
    "Other": (50, 1000)
}

PAYMENT_METHODS = ["Cash", "Credit Card", "UPI", "Net Banking"]

def generate_synthetic_data(num_days=180):
    session = get_session()
    
    # Check if user exists, else create one
    user = session.query(User).filter(User.email == "user@example.com").first()
    if not user:
        user = User(
            name="Test User", 
            email="user@example.com", 
            password=hash_password("Finance@2026"),
            monthly_income=50000.0
        )
        session.add(user)
        session.commit()
    
    start_date = datetime.date.today() - datetime.timedelta(days=num_days)
    
    expenses = []
    
    for i in range(num_days):
        current_date = start_date + datetime.timedelta(days=i)
        
        # Monthly Rent
        if current_date.day == 1:
            expenses.append(Expense(
                user_id=user.user_id,
                amount=15000.0,
                category="Rent",
                payment_method="Net Banking",
                description="Monthly Apartment Rent",
                location="City Center",
                expense_date=current_date
            ))
            
        # Monthly Subscription
        if current_date.day == 5:
            expenses.append(Expense(
                user_id=user.user_id,
                amount=199.0,
                category="Subscription",
                payment_method="Credit Card",
                description="Netflix Subscription",
                location="Online",
                expense_date=current_date
            ))

        # Daily random expenses
        num_expenses = random.randint(1, 4)
        for _ in range(num_expenses):
            category = random.choice(list(CATEGORIES.keys()))
            min_val, max_val = CATEGORIES[category]
            
            # Weekend spikes
            if current_date.weekday() >= 5: # Sat or Sun
                max_val *= 1.5
                
            amount = round(random.uniform(min_val, max_val), 2)
            
            # Random Anomaly (0.5% chance)
            if random.random() < 0.005:
                amount *= 10
                description = f"Emergency {category} expense"
            else:
                description = f"Daily {category}"

            expenses.append(Expense(
                user_id=user.user_id,
                amount=amount,
                category=category,
                payment_method=random.choice(PAYMENT_METHODS),
                description=description,
                location="Local Store",
                expense_date=current_date
            ))
            
    session.add_all(expenses)
    session.commit()
    print(f"Generated {len(expenses)} expenses for {user.name}")
    session.close()

if __name__ == "__main__":
    init_db()
    generate_synthetic_data()
