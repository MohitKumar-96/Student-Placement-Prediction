import streamlit as st
import pandas as pd
import joblib
import os
import xgboost as xgb

# Configure Page
st.set_page_config(page_title="AHPP Student Placement Predictor", layout="centered")
st.title("🎓 Adaptive Hybrid Student Placement Predictor (AHPP)")
st.write("Enter student details below to predict their placement probability.")

MODEL_PATH = "student_placement_AHPP_model (1).joblib"
FALLBACK_MODEL_PATH = "best_individual_model_random_forest (1).joblib"

@st.cache_resource
def load_assets():
    global MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        alt_path = "placement_research_results/student_placement_AHPP_model.joblib"
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
        else:
            st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure your joblib file is in the same folder.")
            return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning("⚠️ XGBoost/Pickle Version Mismatch detected in full bundle. Falling back to robust individual Random Forest model...")
        if os.path.exists(FALLBACK_MODEL_PATH):
            return joblib.load(FALLBACK_MODEL_PATH)
        elif os.path.exists("best_individual_model_random_forest.joblib"):
            return joblib.load("best_individual_model_random_forest.joblib")
        else:
            st.error(f"Could not load fallback model: {e}")
            return None

bundle = load_assets()

if bundle:
    if isinstance(bundle, dict):
        models = bundle.get("models", {})
        weights = bundle.get("weights", {})
        features = bundle.get("features", [])
    else:
        models = {"Random Forest (Fallback)": bundle}
        weights = {"Random Forest (Fallback)": 1.0}
        features = [
            'branch', 'college_tier', 'cgpa', 'backlogs', 'coding_skills',
            'dsa_score', 'aptitude_score', 'communication_skills', 'ml_knowledge',
            'system_design', 'internships', 'projects_count', 'certifications',
            'hackathons', 'open_source_contributions', 'extracurriculars', 'salary_package_lpa',
            'engineered_skill_score', 'engineered_experience_score', 'engineered_academic_score', 'engineered_academic_risk'
        ]

    # Display current ensemble config
    with st.expander("ℹ️ Adaptive Hybrid Weights"):
        st.json(weights)

    st.subheader("Student Parameters Input")

    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs["branch"] = st.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "EE"])
        inputs["college_tier"] = st.slider("College Tier", 1, 3, 2)
        inputs["cgpa"] = st.slider("CGPA", 3.0, 10.0, 7.5, step=0.1)
        inputs["backlogs"] = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
        inputs["coding_skills"] = st.slider("Coding Skills (Score)", 10, 100, 65)
        inputs["dsa_score"] = st.slider("DSA Score", 10, 100, 65)
        inputs["aptitude_score"] = st.slider("Aptitude Score", 10, 100, 65)
        inputs["communication_skills"] = st.slider("Communication Skills", 10, 100, 65)

    with col2:
        inputs["ml_knowledge"] = st.slider("ML Knowledge Score", 10, 100, 60)
        inputs["system_design"] = st.slider("System Design Score", 10, 100, 60)
        inputs["internships"] = st.number_input("Internships Done", min_value=0, max_value=10, value=1)
        inputs["projects_count"] = st.number_input("Projects Completed", min_value=0, max_value=10, value=2)
        inputs["certifications"] = st.number_input("Certifications", min_value=0, max_value=10, value=1)
        inputs["hackathons"] = st.number_input("Hackathons Participated", min_value=0, max_value=10, value=0)
        inputs["open_source_contributions"] = st.selectbox("Open Source Contributions", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        inputs["extracurriculars"] = st.selectbox("Extracurriculars", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        inputs["salary_package_lpa"] = st.number_input("Salary Package LPA (Estimated)", min_value=0.0, max_value=40.0, value=6.0, step=0.5)

    def engineer_features_inference(df_row):
        def get_num(col):
            return pd.to_numeric(df_row[col], errors='coerce')

        df_row['engineered_skill_score'] = (
            get_num('coding_skills') + get_num('aptitude_score') + get_num('communication_skills')
        ) / 3

        df_row['engineered_experience_score'] = (
            get_num('internships') + get_num('projects_count') + get_num('certifications') + get_num('hackathons')
        ) / 4

        df_row['engineered_academic_score'] = (
            (get_num('cgpa') * 10) + 100
        ) / 2

        df_row['engineered_academic_risk'] = get_num('backlogs') / (get_num('cgpa') + 0.1)
        return df_row

    if st.button("🔮 Predict Placement Status"):
        raw_row = pd.DataFrame([inputs])
        processed_row = engineer_features_inference(raw_row)
        processed_row = processed_row.reindex(columns=features)

        # 1. Calculate raw ML probability
        probs = {}
        for name, model in models.items():
            probs[name] = float(model.predict_proba(processed_row)[0, 1])

        calculated_prob = sum(weights[name] * probs[name] for name in models if name in weights)

        # 2. Apply penalty for excessive backlogs / low CGPA
        if inputs["backlogs"] > 2 or inputs["cgpa"] < 6.0:
            # Reduce calculated probability by 80% for failing cutoff rules
            calculated_prob = calculated_prob * 0.20

        st.markdown("---")
        st.subheader("Prediction Summary")

        # 3. Display real scaled probability
        if calculated_prob >= 0.50:
            st.success(f"Prediction: **PLACED** (Probability: {calculated_prob:.2%})")
        else:
            st.error(f"Prediction: **NOT PLACED** (Probability: {calculated_prob:.2%})")
            if inputs["backlogs"] > 2 or inputs["cgpa"] < 6.0:
                st.caption("🚨 Note: Placement probability significantly reduced due to active backlogs (>2) or low CGPA (<6.0).")

        st.subheader("Model Breakdown")
        st.bar_chart(pd.Series(probs))