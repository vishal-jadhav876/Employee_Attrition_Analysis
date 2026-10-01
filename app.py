import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS for HR Dark UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Employee Attrition Analytics",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0c0f16;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #151922 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2a2e38;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 30px;
        font-weight: 800;
    }
    .pos-val { color: #2ecc71; }
    .neg-val { color: #e74c3c; }
    .tot-val { color: #3498db; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Synthetic HR Analytics Dataset Generator
# -----------------------------------------------------------------------------
@st.cache_data
def load_hr_data():
    np.random.seed(42)
    n_samples = 1000

    age = np.random.randint(20, 60, n_samples)
    monthly_income = np.random.normal(6500, 2500, n_samples).clip(2000, 20000)
    years_at_company = np.random.randint(1, 20, n_samples)
    distance_from_home = np.random.randint(1, 30, n_samples)
    overtime = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    work_life_balance = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.1, 0.3, 0.4, 0.2])
    job_satisfaction = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.25, 0.4, 0.2])
    
    attrition_prob = (
        (60 - age) / 100 * 0.2 +
        (20000 - monthly_income) / 20000 * 0.3 +
        (overtime * 0.3) +
        (5 - work_life_balance) / 5 * 0.2
    )
    attrition_prob = np.clip(attrition_prob, 0.05, 0.85)
    attrition = np.random.binomial(1, attrition_prob / 2.2)

    df = pd.DataFrame({
        'Age': age,
        'Monthly_Income': np.round(monthly_income, 2),
        'Years_At_Company': years_at_company,
        'Distance_From_Home': distance_from_home,
        'OverTime': overtime,
        'Work_Life_Balance': work_life_balance,
        'Job_Satisfaction': job_satisfaction,
        'Attrition': attrition
    })
    return df

# -----------------------------------------------------------------------------
# 3. Model Training Pipeline (Corrected train_test_split)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_hr_model(df):
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    # Corrected function call
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return model, acc, auc, X.columns

# -----------------------------------------------------------------------------
# 4. Dashboard Layout
# -----------------------------------------------------------------------------
st.title("👥 Executive HR Attrition & Retention Dashboard")
st.caption("Predict employee turnover, analyze resignation drivers, and plan retention strategies.")
st.markdown("---")

df = load_hr_data()

st.sidebar.title("⚙️ HR Management Portal")
mode = st.sidebar.radio("Navigation:", ["📊 HR KPI Analytics & Risk Factors", "🎯 Employee Flight Risk Predictor"])

model, acc, auc, feature_names = train_hr_model(df)

# -----------------------------------------------------------------------------
# Page 1: HR KPI Analytics & Risk Factors
# -----------------------------------------------------------------------------
if mode == "📊 HR KPI Analytics & Risk Factors":
    st.subheader("📌 HR Key Performance Indicators (KPIs)")

    tot_emp = len(df)
    attrition_cases = df['Attrition'].sum()
    attrition_rate = (attrition_cases / tot_emp) * 100
    avg_income = df['Monthly_Income'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div>Total Headcount</div><div class="metric-val tot-val">{tot_emp}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div>Total Turnover</div><div class="metric-val neg-val">{attrition_cases}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div>Attrition Rate</div><div class="metric-val neg-val">{attrition_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div>Avg Monthly Salary</div><div class="metric-val pos-val">${avg_income:,.0f}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("📋 HR Employee Data Log")
        st.dataframe(df.head(10), use_container_width=True)

    with right_col:
        st.subheader("🔍 Top Drivers for Employee Resignation")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'HR Factor': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(5, 3.8))
        fig.patch.set_facecolor('#0c0f16')
        ax.set_facecolor('#0c0f16')
        ax.barh(feat_df['HR Factor'], feat_df['Importance'], color='#3498db')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#2a2e38')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2a2e38')
        ax.set_xlabel("Impact Score", color="white")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Page 2: Employee Flight Risk Predictor
# -----------------------------------------------------------------------------
else:
    st.subheader("🎯 Test Employee Resignation Risk (Flight Risk)")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Employee Age", 18, 65, 32)
        income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
        tenure = st.slider("Years at Company", 0, 30, 3)
        dist = st.slider("Distance From Home (KM)", 1, 50, 12)
    
    with col2:
        overtime_str = st.selectbox("Requires OverTime Working?", ["No", "Yes"])
        wlb = st.slider("Work-Life Balance Rating (1 Bad - 4 Best)", 1, 4, 2)
        satisfaction = st.slider("Job Satisfaction Rating (1 Low - 4 High)", 1, 4, 2)

    overtime = 1 if overtime_str == "Yes" else 0

    if st.button("🚨 Assess Resignation Risk"):
        input_data = pd.DataFrame([[age, income, tenure, dist, overtime, wlb, satisfaction]], columns=feature_names)
        
        attrition_prob = model.predict_proba(input_data)[0][1] * 100
        prediction = model.predict(input_data)[0]

        st.write("")
        st.subheader("📋 HR Risk Assessment Results")

        if prediction == 1 or attrition_prob > 45.0:
            st.error(f"🚨 **HIGH FLIGHT RISK (LIKELY TO RESIGN)** | Resignation Probability: `{attrition_prob:.1f}%`")
            st.warning("Recommended HR Action: Review compensation plan, reduce overtime pressure, or discuss career progression opportunities.")
        else:
            st.success(f"✅ **LOW FLIGHT RISK (STABLE / RETAINED)** | Resignation Probability: `{attrition_prob:.1f}%`")
            st.info("The employee demonstrates strong retention indicators and balanced workload.")