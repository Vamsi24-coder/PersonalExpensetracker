import re
import streamlit as st
from database import get_session, User, hash_password, verify_password

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_password(password):
    # Min 8 chars, 1 upper, 1 lower, 1 number, 1 special
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c in "!@#$%^&*()-_+=<>?" for c in password):
        return False
    return True

def signup_user(name, email, password, income):
    session = get_session()
    try:
        if session.query(User).filter(User.email == email).first():
            return False, "Email already registered."
        
        new_user = User(
            name=name,
            email=email,
            password=hash_password(password),
            monthly_income=float(income)
        )
        session.add(new_user)
        session.commit()
        return True, "Account created successfully!"
    except Exception as e:
        return False, str(e)
    finally:
        session.close()

def login_user(email, password):
    session = get_session()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user and verify_password(password, user.password):
            return True, user
        return False, "Invalid email or password."
    finally:
        session.close()

def init_auth_session():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None

def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()
