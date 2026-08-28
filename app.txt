import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Analytics & Prediction",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Employee Attrition Intelligence & Risk Predictor")
st.markdown("Automated HR analytics pipeline using Random Forest classification to identify turnover factors and forecast employee attrition.")

# -------------------------------------------------------------
# 2. DATA GENERATION / PREPROCESSING
# -------------------------------------------------------------
@st.cache_data
def load_hr_dataset():
    np.random.seed(42)
    n_samples = 1200
    
    departments = ["Sales", "Research & Development", "Human Resources"]
    job_roles = [
        "Sales Executive", "Research Scientist", "Software Engineer", 
        "HR Specialist", "Manager", "Laboratory Technician"
    ]
    
    age = np.random.randint(22, 60, size=n_samples)
    department = np.random.choice(departments, size=n_samples, p=[0.45, 0.45, 0.10])
    job_role = np.random.choice(job_roles, size=n_samples)
    monthly_income = np.random.randint(2500, 20000, size=n_samples)
    years_at_company = np.random.randint(0, 25, size=n_samples)
    job_satisfaction = np.random.randint(1, 5, size=n_samples)  # 1 to 4 scale
    work_life_balance = np.random.randint(1, 5, size=n_samples) # 1 to 4 scale
    overtime = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    distance_from_home = np.random.randint(1, 30, size=n_samples)

    df = pd.DataFrame({
        "Age": age,
        "Department": department,
        "JobRole": job_role,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "OverTime": overtime,
        "DistanceFromHome": distance_from_home
    })

    # Rule-based calculation to simulate realistic attrition trends
    churn_probability = (
        (df["OverTime"] == "Yes").astype(int) * 0.35 +
        (df["JobSatisfaction"] <= 2).astype(int) * 0.25 +
        (df["MonthlyIncome"] < 5000).astype(int) * 0.20 +
        (df["WorkLifeBalance"] == 1).astype(int) * 0.15 +
        (df["YearsAtCompany"] < 2).astype(int) * 0.10 -
        (df["MonthlyIncome"] > 12000).astype(int) * 0.20 -
        (df["YearsAtCompany"] > 8).astype(int) * 0.15
    )
    churn_probability = np.clip(churn_probability, 0.05, 0.90)
    df["Attrition"] = np.where(np.random.rand(n_samples) < churn_probability, "Yes", "No")

    return df

df = load_hr_dataset()

# -------------------------------------------------------------
# 3. MODEL TRAINING & FEATURE IMPORTANCE
# -------------------------------------------------------------
@st.cache_resource
def train_attrition_model(data):
    features = [
        "Age", "MonthlyIncome", "YearsAtCompany", 
        "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome"
    ]
    
    X = data[features].copy()
    X["OverTime_Yes"] = (data["OverTime"] == "Yes").astype(int)
    
    # Encode categorical Department features
    dept_dummies = pd.get_dummies(data["Department"], drop_first=True)
    X = pd.concat([X, dept_dummies], axis=1)
    
    y = (data["Attrition"] == "Yes").astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=True)
    
    return clf, X.columns.tolist(), auc_score, importance_df

model, feature_names, model_auc, feature_importance_df = train_attrition_model(df)

# -------------------------------------------------------------
# 4. SIDEBAR FILTERS
# -------------------------------------------------------------
st.sidebar.header("Filter Analytics")
dept_filter = st.sidebar.multiselect(
    "Select Department",
    options=df["Department"].unique(),
    default=df["Department"].unique()
)

filtered_df = df[df["Department"].isin(dept_filter)]

# -------------------------------------------------------------
# 5. KEY PERFORMANCE INDICATOR CARDS
# -------------------------------------------------------------
total_employees = len(filtered_df)
attrition_count = (filtered_df["Attrition"] == "Yes").sum()
attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
avg_salary = filtered_df["MonthlyIncome"].mean() if total_employees > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Headcount", f"{total_employees:,}")
kpi2.metric("Total Resignations", f"{attrition_count:,}")
kpi3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
kpi4.metric("Avg Monthly Income", f"${avg_salary:,.0f}")

st.markdown("---")

# -------------------------------------------------------------
# 6. CHARTS & INFLUENCING FACTORS
# -------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    fig_dept = px.histogram(
        filtered_df, x="Department", color="Attrition", barmode="group",
        title="Departmental Attrition Distribution",
        color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"}
    )
    fig_dept.update_layout(height=380)
    st.plotly_chart(fig_dept, use_container_width=True)

with c2:
    fig_imp = px.bar(
        feature_importance_df, x="Importance", y="Feature", orientation="h",
        title="Top Drivers of Employee Turnover",
        color="Importance", color_continuous_scale="Blues"
    )
    fig_imp.update_layout(height=380)
    st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------
# 7. INTERACTIVE RISK PREDICTOR FOR HR
# -------------------------------------------------------------
st.subheader("🔍 Individual Employee Turnover Risk Predictor")
st.markdown("Input employee parameters to calculate individual churn probability via the trained Random Forest classifier.")

with st.form("risk_assessment_form"):
    col_a, col_b, col_c = st.columns(3)
    input_age = col_a.slider("Age", 18, 65, 29)
    input_salary = col_b.number_input("Monthly Income ($)", min_value=1500, max_value=25000, value=3800, step=500)
    input_years = col_c.slider("Years at Current Company", 0, 30, 2)
    
    col_d, col_e, col_f = st.columns(3)
    input_dept = col_d.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    input_satisfaction = col_e.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=1, format_func=lambda x: f"{x} - Low" if x==1 else (f"{x} - High" if x==4 else str(x)))
    input_overtime = col_f.selectbox("Works Overtime?", ["Yes", "No"])
    
    col_g, col_h = st.columns(2)
    input_wlb = col_g.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=2)
    input_dist = col_h.slider("Distance From Home (Miles)", 1, 40, 15)
    
    submit_button = st.form_submit_button("⚡ Predict Resignation Risk")

if submit_button:
    # Construct input vector matching feature columns
    input_record = {
        "Age": input_age,
        "MonthlyIncome": input_salary,
        "YearsAtCompany": input_years,
        "JobSatisfaction": input_satisfaction,
        "WorkLifeBalance": input_wlb,
        "DistanceFromHome": input_dist,
        "OverTime_Yes": 1 if input_overtime == "Yes" else 0,
        "Research & Development": 1 if input_dept == "Research & Development" else 0,
        "Sales": 1 if input_dept == "Sales" else 0
    }
    
    input_df = pd.DataFrame([input_record])[feature_names]
    risk_score = model.predict_proba(input_df)[0][1]
    risk_pct = risk_score * 100
    
    st.markdown("### Risk Assessment Result")
    if risk_score >= 0.50:
        st.error(f"⚠️ **High Attrition Risk:** This employee has an estimated **{risk_pct:.1f}%** chance of resigning.")
        st.info("💡 **Recommended HR Retention Actions:** Review overtime workload, consider a compensation/market adjustment, and schedule a 1-on-1 feedback session.")
    else:
        st.success(f"✅ **Low Attrition Risk:** This employee has an estimated **{risk_pct:.1f}%** chance of resigning.")