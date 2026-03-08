from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'expenses_system.db')
ENGINE_URL = f'sqlite:///{DB_PATH}'

import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String) # Hashed
    monthly_income = Column(Float)
    currency = Column(String, default="₹")
    created_at = Column(Date, default=datetime.date.today)

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class Expense(Base):
    __tablename__ = 'expenses'
    expense_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    amount = Column(Float)
    category = Column(String)
    payment_method = Column(String)
    description = Column(Text)
    location = Column(String)
    expense_date = Column(Date)

class Budget(Base):
    __tablename__ = 'budgets'
    budget_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    category = Column(String)
    recommended_budget = Column(Float)
    user_budget = Column(Float)
    created_at = Column(Date, default=datetime.date.today)

class Anomaly(Base):
    __tablename__ = 'anomalies'
    anomaly_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    expense_id = Column(Integer, ForeignKey('expenses.expense_id'))
    anomaly_score = Column(Float)
    message = Column(Text)
    detected_at = Column(Date, default=datetime.date.today)

class RecurringExpense(Base):
    __tablename__ = 'recurring_expenses'
    recurring_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    category = Column(String)
    amount = Column(Float)
    frequency = Column(String)
    detected_at = Column(Date, default=datetime.date.today)

def init_db():
    dir_path = os.path.dirname(DB_PATH)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    engine = create_engine(ENGINE_URL)
    Base.metadata.create_all(engine)
    return engine

def get_session():
    engine = create_engine(ENGINE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
