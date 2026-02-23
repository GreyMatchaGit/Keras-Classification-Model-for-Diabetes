import streamlit as st
import pandas as pd

def display_dataset_overview(DATA_PATH):
    st.markdown("# Health & Lifestyle Data for Diabetes Prediction")
    dataset = pd.read_csv(DATA_PATH)
    st.success("**About Dataset:** Provides patient health data, which integrates multiple health dimensions of their demographic, lifestyle, and clinical information to enable robust data-driven insights for diabetes progression and prevention.")
    st.dataframe(dataset)
    st.link_button("📊 Link to Kaggle Dataset","https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction")


    ds_cat1, ds_cat2, ds_cat3 = st.columns([1,1,1])
    with ds_cat1:
        st.markdown("### 🌍 Demographic Information")
        dg_1, dg_2, dg_3 = st.columns([1,1,1.21])
        with dg_1:
            st.success("Age")
            st.success("Employment")
        with dg_2:
            st.success("Gender")
            st.success("Ethnicity")
        with dg_3:
            st.success("Income Category")
            st.success("Education Level")

    with ds_cat2:
        st.markdown("### 🚬 Lifestyle Indicators")
        ls_1, ls_2 = st.columns([1,1])
        with ls_1:
            st.error("Smoking")
            st.error("Alcohol Consumption")
        with ls_2:
            st.error("Sleep Patterns")
            st.error("Physical Activity")

    with ds_cat3:
        st.markdown("### 🏥 Clinical Information")
        ci_1, ci_2, ci_3 = st.columns([1,1,1.21])
        with ci_1:
            st.info("Age")
            st.info("Ethnicity")
        with ci_2:
            st.info("Gender")
            st.info("Employment")
        with ci_3:
            st.info("Education Level")
            st.info("Income Category")