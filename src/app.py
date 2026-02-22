import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import glob
import re

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Diabetes Clinical System",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# 2. PATHS & CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "Models")
DATA_PATH = os.path.join(PROJECT_ROOT, "Diabetes Dataset", "Diabetes_and_LifeStyle_Dataset.csv")

# 3. MODEL DESCRIPTIONS (Add notes for specific versions)
MODEL_META = {
    "patient": {
        "v1.0.0": "Initial baseline",
        "v1.0.1": "Optimized model",
    },
    "doctor": {
        "v1.0.0": "Initial baseline",
        "v1.0.1": "10 ka 5 layers, better performance",
        "v1.0.2": "5 ka 10 layers, more regularization",
    }
}

# 4. HELPER: VERSION DISCOVERY (Updated for Subfolders)
def get_available_versions(base_dir, model_name):
    """
    Scans Models/{model_name}/ for .h5 files.
    Example: Models/patient_side_model/patient_side_model_v1.0.1.h5
    """
    target_dir = os.path.join(base_dir, model_name)
    
    if not os.path.exists(target_dir):
        return []
        
    files = glob.glob(os.path.join(target_dir, f"{model_name}_v*.h5"))
    
    # Regex to extract version: v1.0.1
    pattern = re.compile(r"(v\d+\.\d+\.\d+)\.h5$")
    
    versions = []
    for f in files:
        match = pattern.search(f)
        if match:
            versions.append(match.group(1))
            
    try:
        versions.sort(key=lambda s: list(map(int, s[1:].split('.'))), reverse=True)
    except:
        versions.sort(reverse=True)
        
    return versions

# 5. SIDEBAR CONFIGURATION
st.sidebar.header("System Configuration")

# Format function for dropdown
def format_func(option, mode):
    ver = option
    desc = MODEL_META.get(mode, {}).get(ver, "")
    if desc:
        return f"{ver} — {desc}"
    return ver

pat_folder_name = "patient_side_model"
pat_versions = get_available_versions(MODEL_DIR, pat_folder_name)

if not pat_versions:
    st.sidebar.error(f"No models found in Models/{pat_folder_name}/")
    selected_pat_ver = None
else:
    selected_pat_ver = st.sidebar.selectbox(
        "Patient Model", 
        pat_versions, 
        format_func=lambda x: format_func(x, "patient")
    )

doc_folder_name = "doctor_side_model"
doc_versions = get_available_versions(MODEL_DIR, doc_folder_name)

if not doc_versions:
    st.sidebar.error(f"No models found in Models/{doc_folder_name}/")
    selected_doc_ver = None
else:
    selected_doc_ver = st.sidebar.selectbox(
        "Doctor Model", 
        doc_versions,
        format_func=lambda x: format_func(x, "doctor")
    )

# Debug Expander (To verify paths)
with st.sidebar.expander("Debug Paths"):
    st.write(f"**Root:** `{MODEL_DIR}`")
    st.write(f"**Patient Folder:** `{pat_folder_name}` found? {os.path.exists(os.path.join(MODEL_DIR, pat_folder_name))}")
    st.write(f"**Doctor Folder:** `{doc_folder_name}` found? {os.path.exists(os.path.join(MODEL_DIR, doc_folder_name))}")

st.sidebar.divider()

# 6. LOAD ASSETS
@st.cache_resource(show_spinner="Loading Models...")
def load_assets(pat_ver, doc_ver):
    assets = {"patient": {}, "doctor": {}}
    
    configs = {
        "patient": {"folder": pat_folder_name, "ver": pat_ver},
        "doctor":  {"folder": doc_folder_name, "ver": doc_ver}
    }
    
    for mode, cfg in configs.items():
        try:
            if not cfg["ver"]: raise ValueError("No version selected.")
                
            filename = f"{cfg['folder']}_{cfg['ver']}.h5"
            model_path = os.path.join(MODEL_DIR, cfg["folder"], filename)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing file: {model_path}")
            assets[mode]["model"] = tf.keras.models.load_model(model_path)
            
            def load_artifact(name):
                path1 = os.path.join(MODEL_DIR, cfg["folder"], name)
                if os.path.exists(path1): return joblib.load(path1)
                path2 = os.path.join(MODEL_DIR, name)
                if os.path.exists(path2): return joblib.load(path2)
                raise FileNotFoundError(f"Could not find artifact {name} in {path1} or {path2}")

            assets[mode]["pipeline"] = load_artifact(f"{mode}_pipeline.pkl")
            assets[mode]["target"] = load_artifact(f"{mode}_target_enc.pkl")
            assets[mode]["features"] = load_artifact(f"{mode}_features.pkl")
            
            try:
                ohe = assets[mode]["pipeline"].named_transformers_['cat']
                cat_names = assets[mode]["features"]['cat']
                assets[mode]["valid_cats"] = {
                    col: list(ohe.categories_[i]) 
                    for i, col in enumerate(cat_names)
                }
            except:
                assets[mode]["valid_cats"] = {}

            assets[mode]["status"] = True
            
        except Exception as e:
            assets[mode]["status"] = False
            assets[mode]["error"] = str(e)
            
    try:
        assets["dataset"] = pd.read_csv(DATA_PATH)
    except:
        assets["dataset"] = None
        
    return assets

assets = load_assets(selected_pat_ver, selected_doc_ver)

# 7. LOGIC & UI
if "patient_inputs" not in st.session_state: st.session_state["patient_inputs"] = {}
if "doctor_inputs" not in st.session_state: st.session_state["doctor_inputs"] = {}

def randomize(mode):
    if assets["dataset"] is not None:
        row = assets["dataset"].sample(1).iloc[0].to_dict()
        st.session_state[f"{mode}_inputs"] = row

def get_val(mode, col, default):
    return st.session_state[f"{mode}_inputs"].get(col, default)

st.title("Intelligent Diabetes Classification System")

# Badges
if selected_pat_ver and selected_doc_ver:
    c1, c2 = st.columns(2)
    c1.caption(f"Patient Logic: **{selected_pat_ver}**")
    c2.caption(f"Doctor Logic: **{selected_doc_ver}**")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; padding-top:10px;}
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

tab_patient, tab_doctor = st.tabs(["Patient Screening", "Clinical Dashboard"])

with tab_patient:
    if not assets["patient"]["status"]:
        st.error(f"Error Loading Model:\n{assets['patient'].get('error')}")
    else:
        c_head, c_btn = st.columns([3, 1])
        c_head.info("**Screening Mode:** Lifestyle data only.")
        if c_btn.button("Randomize Patient", key="rand_pat", use_container_width=True):
            randomize("patient")
            st.rerun()

        with st.form("patient_form"):
            st.markdown("### Demographics")
            c1, c2, c3, c4 = st.columns(4)
            p_age = c1.number_input("Age (Years)", 18, 100, int(get_val("patient", "Age", 30)), help="Risk increases with age, particularly over 45.")
            p_bmi = c2.number_input("BMI (kg/m²)", 10.0, 60.0, float(get_val("patient", "bmi", 24.0)), help="Normal: 18.5-24.9\nOverweight: 25-29.9\nObese: 30+")
            p_whr = c3.number_input("Waist/Hip Ratio", 0.5, 2.0, float(get_val("patient", "waist_to_hip_ratio", 0.9)))
            
            valid_cats = assets["patient"]["valid_cats"]
            
            def safe_selectbox(label, col_name, default_opts, tooltip=None):
                opts = valid_cats.get(col_name, default_opts)
                curr = get_val("patient", col_name, opts[0])
                idx = list(opts).index(curr) if curr in opts else 0
                return c4.selectbox(label, opts, index=idx, help=tooltip)

            p_gen = safe_selectbox("Gender", "gender", ["Male", "Female"])

            st.markdown("### History & Lifestyle")
            h1, h2, h3 = st.columns(3)
            
            def safe_hist_box(col, container):
                opts = valid_cats.get(col, [0, 1])
                curr = get_val("patient", col, opts[0])
                try: idx = list(opts).index(curr)
                except: idx = 0
                return container.selectbox(col.replace("_", " ").title(), opts, index=idx, help="Family history significantly increases risk.")

            p_fam = safe_hist_box("family_history_diabetes", h1)
            p_hyp = safe_hist_box("hypertension_history", h2)
            p_card = safe_hist_box("cardiovascular_history", h3)

            l1, l2, l3 = st.columns(3)
            
            def safe_cat_box(col, container, tooltip=None):
                opts = valid_cats.get(col, ["Other"])
                curr = get_val("patient", col, opts[0])
                try: idx = list(opts).index(curr)
                except: idx = 0
                return container.selectbox(col.replace("_", " ").title(), opts, index=idx, help=tooltip)

            p_smoke = safe_cat_box("smoking_status", l1, "Smoking affects cardiovascular health and blood sugar.")
            p_eth = safe_cat_box("ethnicity", l2)
            p_alc = l3.number_input("Alcohol (Drinks/Wk)", 0, 50, int(get_val("patient", "alcohol_consumption_per_week", 2)))

            l4, l5, l6 = st.columns(3)
            p_phys = l4.number_input("Exercise (Mins/Wk)", 0, 1000, int(get_val("patient", "physical_activity_minutes_per_week", 150)))
            p_sleep = l5.number_input("Sleep (Hrs)", 2, 24, int(get_val("patient", "sleep_hours_per_day", 7)))
            p_screen = l6.number_input("Screen Time (Hrs)", 0, 24, int(get_val("patient", "screen_time_hours_per_day", 6)))

            submitted_p = st.form_submit_button("Assess Patient Risk", use_container_width=True)

        if submitted_p:
            try:
                feats = assets["patient"]["features"]
                raw_data = {
                    'Age': [p_age], 'bmi': [p_bmi], 'waist_to_hip_ratio': [p_whr],
                    'alcohol_consumption_per_week': [p_alc],
                    'physical_activity_minutes_per_week': [p_phys],
                    'sleep_hours_per_day': [p_sleep], 'screen_time_hours_per_day': [p_screen],
                    'gender': [p_gen], 'ethnicity': [p_eth], 'smoking_status': [p_smoke],
                    'family_history_diabetes': [p_fam], 'hypertension_history': [p_hyp],
                    'cardiovascular_history': [p_card],
                    'diabetes_risk_score': [0], 'diet_score': [0]
                }
                
                df = pd.DataFrame(raw_data)
                X_proc = assets["patient"]["pipeline"].transform(df)
                probs = assets["patient"]["model"].predict(X_proc)
                pred_label = assets["patient"]["target"].inverse_transform([np.argmax(probs)])[0]
                conf = np.max(probs) * 100
                
                st.success(f"### Result: {pred_label}")
                st.metric("Confidence", f"{conf:.1f}%")
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")

with tab_doctor:
    if not assets["doctor"]["status"]:
        st.error(f"Error Loading Model:\n{assets['doctor'].get('error')}")
    else:
        d_head, d_btn = st.columns([3, 1])
        d_head.info("**Clinical Mode:** Full dataset features + Lab results.")
        if d_btn.button("Randomize Case", key="rand_doc", use_container_width=True):
            randomize("doctor")
            st.rerun()

        with st.form("doctor_form"):
            st.markdown("### Vitals & Labs")
            d1, d2, d3, d4 = st.columns(4)
            d_age = d1.number_input("Age", 18, 100, int(get_val("doctor", "Age", 45)))
            d_bmi = d2.number_input("BMI (kg/m²)", 10.0, 60.0, float(get_val("doctor", "bmi", 28.0)), help="Healthy: 18.5-24.9\nHigh Risk: ≥ 25.0")
            d_sys = d3.number_input("Systolic BP (mmHg)", 80, 250, int(get_val("doctor", "systolic_bp", 120)), help="Normal: < 120\nElevated: 120-129\nHigh: ≥ 130")
            d_dia = d4.number_input("Diastolic BP (mmHg)", 50, 150, int(get_val("doctor", "diastolic_bp", 80)), help="Normal: < 80\nHigh: ≥ 80")
            
            L1, L2, L3 = st.columns(3)
            d_gluc_f = L1.number_input("Fasting Glucose (mg/dL)", 50, 500, int(get_val("doctor", "glucose_fasting", 100)), help="Normal: < 100 mg/dL\nDiabetes: ≥ 126 mg/dL")
            d_gluc_p = L2.number_input("Post-Prandial (mg/dL)", 50, 600, int(get_val("doctor", "glucose_postprandial", 140)), help="Normal: < 140 mg/dL\nDiabetes: ≥ 200 mg/dL")
            d_hba1c = L3.number_input("HbA1c (%)", 3.0, 20.0, float(get_val("doctor", "hba1c", 5.5)), help="Normal: < 5.7%\nDiabetes: ≥ 6.5%")
            
            L4, L5, L6 = st.columns(3)
            d_ins = L4.number_input("Insulin (mIU/L)", 0, 200, int(get_val("doctor", "insulin_level", 15)), help="Normal: < 25 mIU/L")
            d_chol = L5.number_input("Cholesterol (mg/dL)", 50, 600, int(get_val("doctor", "cholesterol_total", 190)), help="Desirable: < 200 mg/dL")
            d_risk = L6.number_input("Risk Score (Calc)", 0, 100, int(get_val("doctor", "diabetes_risk_score", 0)))
            
            st.markdown("### History & Demographics")
            
            h1, h2, h3 = st.columns(3)
            def get_bin_idx(col):
                val = get_val("doctor", col, 0)
                return 1 if val in [1, "Yes", "1"] else 0
            
            d_fam = h1.selectbox("Family Hist", ["No", "Yes"], index=get_bin_idx("family_history_diabetes"), key="d_fam")
            d_hyp = h2.selectbox("Hypertension", ["No", "Yes"], index=get_bin_idx("hypertension_history"), key="d_hyp")
            d_card = h3.selectbox("Heart Disease", ["No", "Yes"], index=get_bin_idx("cardiovascular_history"), key="d_card")

            valid_cats_doc = assets["doctor"]["valid_cats"]
            
            def safe_doc_box(col, container):
                opts = valid_cats_doc.get(col, ["Unknown"])
                curr = get_val("doctor", col, opts[0])
                try: idx = list(opts).index(curr)
                except: idx = 0
                return container.selectbox(col.replace("_", " ").title(), opts, index=idx)
            
            c1, c2, c3 = st.columns(3)
            d_gen = safe_doc_box("gender", c1)
            d_eth = safe_doc_box("ethnicity", c2)
            d_edu = safe_doc_box("education_level", c3)
            
            c4, c5, c6 = st.columns(3)
            d_inc = safe_doc_box("income_level", c4)
            d_emp = safe_doc_box("employment_status", c5)
            d_smk = safe_doc_box("smoking_status", c6)

            submitted_d = st.form_submit_button("Run Clinical Diagnosis", use_container_width=True)

        if submitted_d:
            try:
                fam_v = 1 if d_fam == "Yes" else 0
                hyp_v = 1 if d_hyp == "Yes" else 0
                card_v = 1 if d_card == "Yes" else 0

                raw_data = {
                    'Age': [d_age], 'bmi': [d_bmi], 
                    'systolic_bp': [d_sys], 'diastolic_bp': [d_dia],
                    'glucose_fasting': [d_gluc_f], 'glucose_postprandial': [d_gluc_p],
                    'hba1c': [d_hba1c], 'insulin_level': [d_ins],
                    'cholesterol_total': [d_chol], 'diabetes_risk_score': [d_risk],
                    'family_history_diabetes': [fam_v],
                    'hypertension_history': [hyp_v],
                    'cardiovascular_history': [card_v],
                    'alcohol_consumption_per_week': [get_val("doctor", "alcohol_consumption_per_week", 2)],
                    'sleep_hours_per_day': [get_val("doctor", "sleep_hours_per_day", 7)],
                    'screen_time_hours_per_day': [get_val("doctor", "screen_time_hours_per_day", 4)],
                    'waist_to_hip_ratio': [get_val("doctor", "waist_to_hip_ratio", 0.9)],
                    'heart_rate': [get_val("doctor", "heart_rate", 75)],
                    'hdl_cholesterol': [get_val("doctor", "hdl_cholesterol", 50)],
                    'ldl_cholesterol': [get_val("doctor", "ldl_cholesterol", 100)],
                    'triglycerides': [get_val("doctor", "triglycerides", 150)],
                    'gender': [d_gen], 'ethnicity': [d_eth], 'education_level': [d_edu],
                    'income_level': [d_inc], 'employment_status': [d_emp], 'smoking_status': [d_smk]
                }
                
                df = pd.DataFrame(raw_data)
                X_proc = assets["doctor"]["pipeline"].transform(df)
                probs = assets["doctor"]["model"].predict(X_proc)
                pred_label = assets["doctor"]["target"].inverse_transform([np.argmax(probs)])[0]
                conf = np.max(probs) * 100
                
                st.divider()
                st.success(f"### Diagnosis: {pred_label}")
                st.metric("Confidence", f"{conf:.1f}%")
                
                st.bar_chart({
                    "Glucose": d_gluc_f,
                    "HbA1c (x15)": d_hba1c * 15,
                    "BMI (x3)": d_bmi * 3
                })

            except Exception as e:
                st.error(f"Error: {e}")