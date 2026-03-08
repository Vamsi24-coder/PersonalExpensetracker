import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import datetime
from sqlalchemy.orm import Session

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from database import get_session, User, Expense, Budget, Anomaly, RecurringExpense, hash_password
from auth import init_auth_session, login_user, signup_user, logout, is_valid_email, is_valid_password
from preprocessing import load_data, feature_engineering, aggregate_by_day
from ml.xgboost_expense_prediction import XGBoostPredictor
from ml.lstm_forecasting import LSTMPredictor
from ml.expense_classifier import ExpenseClassifier
from ml.anomaly_detection import AnomalyDetector
from ml.budget_engine import AdvancedBudgetEngine
from ml.clustering import SpendingClustering

# --- Performance Caching ---
@st.cache_data
def get_cached_data(user_id):
    df = load_data()
    if not df.empty:
        df = df[df['user_id'] == user_id]
        if not df.empty:
            df = feature_engineering(df)
    return df

@st.cache_resource
def load_ml_models():
    return {
        "xgboost": XGBoostPredictor(),
        "lstm": LSTMPredictor(),
        "svm": ExpenseClassifier(),
        "anomaly": AnomalyDetector(),
        "clustering": SpendingClustering(),
        "budget": AdvancedBudgetEngine()
    }

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanceAI Pro | Smart Budgeting",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Management ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def get_theme_palette():
    theme = st.session_state.theme
    if theme == "dark":
        return {
            "primary": "#3b82f6",
            "bg": "#0f172a",
            "card_bg": "#1e293b",
            "text": "#f8fafc",
            "sidebar_bg": "#1e293b",
            "border": "#334155"
        }
    return {
        "primary": "#2563eb",
        "bg": "#f8fafc",
        "card_bg": "#ffffff",
        "text": "#0f172a",
        "sidebar_bg": "#f1f5f9",
        "border": "#e2e8f0"
    }

# --- Advanced Design System ---
def apply_design_system():
    p = get_theme_palette()
    bg, text, card_bg, sidebar_bg, border, primary = p["bg"], p["text"], p["card_bg"], p["sidebar_bg"], p["border"], p["primary"]

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@600&display=swap');
        
        /* Global Background and Text */
        .stApp, .main, [data-testid="stHeader"] {{
            background-color: {bg} !important;
            color: {text} !important;
        }}
        
        /* Ensure all text elements inherit color */
        h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stText {{
            color: {text} !important;
            font-family: 'Inter', sans-serif;
        }}
        
        h1, h2, h3 {{
            font-family: 'Poppins', sans-serif;
        }}

        [data-testid="stSidebar"], [data-testid="stSidebarNav"] {{
            background-color: {sidebar_bg} !important;
        }}

        /* Styled Containers */
        [data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 16px;
            border: 1px solid {border};
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}

        .stMetric {{
            background: {card_bg} !important;
            border: 1px solid {border} !important;
            padding: 20px !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}
        
        .stMetric [data-testid="stMetricValue"], .stMetric [data-testid="stMetricLabel"] {{
            color: {text} !important;
        }}

        .stButton>button {{
            background-color: {primary} !important;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
            color: white !important;
        }}
        
        /* Dark mode inputs */
        .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input {{
            background-color: {card_bg} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Authentication Pages ---
def auth_page():
    apply_design_system()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
        st.title("💰 FinanceAI Pro")
        st.markdown("</div>", unsafe_allow_html=True)
        
        tabs = st.tabs(["Login", "Sign Up"])
        
        with tabs[0]:
            st.markdown("### Welcome Back")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            
            c1, c2 = st.columns([1, 1])
            with c1: st.checkbox("Remember Me")
            with c2: 
                if st.button("Forgot Password?", key="forgot"):
                    st.info("Reset link has been sent to your email (Simulated).")
            
            if st.button("Sign In", use_container_width=True):
                success, result = login_user(email, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.rerun()
                else:
                    st.error(result)
                    
        with tabs[1]:
            st.markdown("### Join FinanceAI")
            new_name = st.text_input("Full Name")
            new_email = st.text_input("Email Address")
            new_pass = st.text_input("Create Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            income = st.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=500.0, format="%.2f")
            
            if st.button("Create Account", use_container_width=True):
                if not is_valid_email(new_email):
                    st.error("Invalid email format.")
                elif not is_valid_password(new_pass):
                    st.error("Password must be 8+ chars, with upper, lower, number, and special char.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    success, msg = signup_user(new_name, new_email, new_pass, income)
                    if success:
                        st.success(msg)
                        st.info("Switch to Login tab to enter.")
                    else:
                        st.error(msg)

# --- Component Views ---
def dashboard_view(user, data):
    p = get_theme_palette()
    text = p["text"]
    st.markdown(f"## 🏠 Dashboard - {user.name}")
    
    # Notifications/Alerts in Header
    session = get_session()
    alerts = session.query(Anomaly).filter(Anomaly.user_id == user.user_id).order_by(Anomaly.detected_at.desc()).limit(1).first()
    if alerts:
        st.warning(f"🔔 **Alert:** {alerts.message}")
    session.close()

    total_spent = data['amount'].sum() if not data.empty else 0
    # Average monthly based on unique months in data or income
    avg_monthly = total_spent / max(1, len(data['expense_date'].dt.month.unique())) if not data.empty else 0
    savings = user.monthly_income - avg_monthly
    health_score = max(0, min(100, (savings / user.monthly_income) * 100)) if user.monthly_income > 0 else 0
    
    # Top Row Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Monthly Income", f"{user.currency}{user.monthly_income:,.2f}")
    with c2:
        with st.container(border=True):
            st.metric("Total Expenses", f"{user.currency}{total_spent:,.2f}", delta=f"-{avg_monthly:,.0f}/mo", delta_color="inverse")
    with c3:
        with st.container(border=True):
            st.metric("Savings", f"{user.currency}{savings:,.2f}", delta=f"{(savings/user.monthly_income)*100:.1f}% ratio" if user.monthly_income > 0 else "0%")
    
    # Health Score & Distribution
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.container(border=True):
            st.markdown("### Financial Health")
            fig_score = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': text},
                    'bar': {'color': "#3b82f6"},
                    'bgcolor': "white" if st.session_state.theme == "light" else "#334155",
                    'steps' : [
                        {'range': [0, 20], 'color': "#ef4444"},
                        {'range': [20, 50], 'color': "#f59e0b"},
                        {'range': [50, 80], 'color': "#10b981"},
                        {'range': [80, 100], 'color': "#047857"}]
                }
            ))
            fig_score.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), 
                                paper_bgcolor='rgba(0,0,0,0)', font={'color': text})
            st.plotly_chart(fig_score, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("### Spending by Category")
            if not data.empty:
                cat_totals = data.groupby('category')['amount'].sum().reset_index()
                fig_pie = px.pie(cat_totals, values='amount', names='category', hole=0.6,
                                color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0),
                                    showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                                    font={'color': text})
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data yet.")

    # Recent Transactions
    st.markdown("### 📜 Recent Transactions")
    if not data.empty:
        st.dataframe(data[['expense_date', 'category', 'amount', 'description', 'payment_method']].sort_values('expense_date', ascending=False).head(10), 
                    use_container_width=True, hide_index=True)
    else:
        st.markdown("<p style='text-align: center; color: #94a3b8;'>No transactions recorded yet. Start tracking to see insights!</p>", unsafe_allow_html=True)

def add_expense_view(user):
    st.markdown("## ➕ Add New Expense")
    st.markdown("Track your spending to build your financial profile.")
    
    with st.container(border=True):
        with st.form("expense_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            amount = c1.number_input("Amount (₹)", min_value=0.01, step=1.0, format="%.2f")
            description = c2.text_input("Description", placeholder="What did you buy?")
            
            c3, c4 = st.columns(2)
            date = c3.date_input("Date", value=datetime.date.today())
            payment = c4.selectbox("Payment Method", ["UPI", "Cash", "Credit Card", "Debit Card", "Net Banking"])
            
            c5, c6 = st.columns(2)
            location = c5.text_input("Location (Optional)")
            time = c6.time_input("Time", value=datetime.datetime.now().time())
            
            submitted = st.form_submit_button("Log Transaction", use_container_width=True)
            if submitted:
                models = load_ml_models()
                classifier = models["svm"]
                category = classifier.predict(description)
                
                session = get_session()
                # ...
                new_expense = Expense(
                    user_id=user.user_id,
                    amount=amount,
                    category=category,
                    payment_method=payment,
                    description=description,
                    location=location,
                    expense_date=date
                )
                session.add(new_expense)
                session.commit()
                session.close()
                st.balloons()
                st.success(f"Expense logged! Categorized as **{category}**")
                get_cached_data.clear()

def history_view(data):
    st.markdown("## 📜 Transaction History")
    if data.empty:
        st.markdown('<div style="text-align:center; padding: 50px;"><h2 style="color:#94a3b8;">No History Found</h2><p>Record your first expense to see it here.</p></div>', unsafe_allow_html=True)
        return
        
    col1, col2 = st.columns([3, 1])
    search = col1.text_input("🔍 Search description...", placeholder="e.g. Netflix, Rent")
    cat_filter = col2.multiselect("🏷️ Category", options=sorted(data['category'].unique()))
    
    filtered_df = data.copy()
    if search:
         filtered_df = filtered_df[filtered_df['description'].str.contains(search, case=False)]
    if cat_filter:
         filtered_df = filtered_df[filtered_df['category'].isin(cat_filter)]
         
    # Data Table
    st.dataframe(filtered_df[['expense_date', 'category', 'amount', 'description', 'payment_method']].sort_values('expense_date', ascending=False), 
                use_container_width=True, height=500)

def predictions_view(data):
    st.markdown("## 📊 AI Spending Forecast")
    if data.empty or len(data) < 40:
        st.info("💡 **Insight:** We need at least 40 days of history to generate high-accuracy forecasts. Keep tracking!")
        return
        
    daily_data = aggregate_by_day(data)
    models = load_ml_models()
    p = get_theme_palette()
    text = p["text"]
    
    t1, t2 = st.tabs(["XGBoost Prediction", "LSTM Deep Learning"])
    
    with t1:
        st.subheader("30-Day Trend Forecast")
        with st.spinner("Analyzing historical patterns..."):
            future_df = models["xgboost"].predict_future(daily_data, num_days=30)
            
        if future_df is not None:
             fig = go.Figure()
             fig.add_trace(go.Scatter(x=daily_data['expense_date'].tail(30), y=daily_data['amount'].tail(30), name="Actual", line=dict(color='#3b82f6', width=3)))
             fig.add_trace(go.Scatter(x=future_df['date'], y=future_df['predicted_amount'], name="Forecast", line=dict(color='#ef4444', width=3, dash='dash')))
             fig.update_layout(template="plotly_dark" if st.session_state.theme == "dark" else "none", 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font={'color': text},
                              xaxis_title="Date", yaxis_title="Amount", margin=dict(l=0, r=0, t=10, b=0))
             st.plotly_chart(fig, use_container_width=True)
             st.success(f"Estimated spending for the next 30 days: **{st.session_state.user.currency}{future_df['predicted_amount'].sum():,.2f}**")

def budget_view(data, user):
    st.markdown("## 💰 Smart Budget Planner")
    
    models = load_ml_models()
    recs = models["budget"].recommend_budgets(data, user.user_id)
    
    if not recs:
        st.info("Record more data to unlock AI budget recommendations.")
        return
        
    st.markdown("### Budget vs. Actual")
    for rec in recs:
        cat = rec['category']
        limit = rec['recommended_budget']
        actual = data[data['category'] == cat]['amount'].sum() / 6 # Monthly average
        progress = min(1.0, actual / limit) if limit > 0 else 0
        
        col1, col2 = st.columns([3, 1])
        with col1:
             st.markdown(f"**{cat}** ({user.currency}{actual:,.0f} / {user.currency}{limit:,.0f})")
             color = "#10b981" if progress < 0.8 else "#f59e0b" if progress < 1.0 else "#ef4444"
             st.markdown(f"""
                 <div style="background: #e2e8f0; border-radius: 10px; height: 12px; width: 100%;">
                     <div style="background: {color}; height: 12px; width: {progress*100}%; border-radius: 10px;"></div>
                 </div>
             """, unsafe_allow_html=True)
        with col2:
             if progress >= 1.0:
                 st.error("Budget Exceeded")
             elif progress >= 0.8:
                 st.warning("Near Limit")
             else:
                 st.success("Safe")
        st.divider()

def profile_view(user, data):
    st.markdown("## 👤 User Profile")
    
    models = load_ml_models()
    behavior = models["clustering"].predict(data)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {user.name}")
        st.write(f"Email: {user.email}")
        st.markdown(f"""
            <div style="background: #3b82f6; color:white; padding: 5px 15px; border-radius: 20px; display: inline-block; margin-top: 10px;">
                {behavior}
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### Settings")
        new_income = st.number_input("Monthly Income (₹)", value=user.monthly_income, step=500.0, format="%.2f")
        new_currency = st.selectbox("Currency", ["₹", "$", "€", "£"])
        
        if st.button("Save Changes"):
            session = get_session()
            db_user = session.query(User).filter(User.user_id == user.user_id).first()
            db_user.monthly_income = new_income
            db_user.currency = new_currency
            session.commit()
            st.session_state.user.monthly_income = new_income
            st.session_state.user.currency = new_currency
            session.close()
            st.success("Profile updated successfully!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main App Execution ---
init_auth_session()

if not st.session_state.authenticated:
    auth_page()
else:
    apply_design_system()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>💼 AI Finance</h2>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("Navigation", 
            ["🏠 Dashboard", "➕ Add Expense", "📜 History", "📊 Predictions", "💰 Budget Planner", "👤 Profile"],
            label_visibility="collapsed")
        
        st.divider()
        if st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark")):
            if st.session_state.theme == "light":
                st.session_state.theme = "dark"
                st.rerun()
        elif st.session_state.theme == "dark":
            st.session_state.theme = "light"
            st.rerun()
            
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            
    # Load Data
    user = st.session_state.user
    data = get_cached_data(user.user_id)
    
    # Page Routing
    if "🏠 Dashboard" in menu:
        dashboard_view(user, data)
    elif "➕ Add Expense" in menu:
        add_expense_view(user)
    elif "📜 History" in menu:
        history_view(data)
    elif "📊 Predictions" in menu:
        predictions_view(data)
    elif "💰 Budget Planner" in menu:
        budget_view(data, user)
    elif "👤 Profile" in menu:
        profile_view(user, data)

    # Footer
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #94a3b8;'>FinanceAI Pro &copy; {datetime.date.today().year} | Smart Financial Assistant</p>", unsafe_allow_html=True)
