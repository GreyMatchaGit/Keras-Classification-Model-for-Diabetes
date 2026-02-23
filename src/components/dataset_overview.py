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
            st.info("Hypertension")
            st.info("BMI")
        with ci_2:
            st.info("Triglycerides")
            st.info("Insulin")
        with ci_3:
            st.info("HbA1c")
            st.info("PM Glucose")
            
    st.divider()
    st.markdown("## Why We Chose This Dataset")
    
    why_1, why_2, why_3 = st.columns(3, gap="medium") 

    with why_1:
        with st.container(border=True):
            st.markdown("### Real World Relevance")
            st.warning("**Top 5** Leading Cause of Death (PH)")
            st.markdown("""
            Diabetes is a persistent metabolic condition affecting millions globally. It hits close to home for many of us—we've all heard the warnings:
            
            > *"Nak, ayaw palabi ug chocolate, magka-diabetes ka!"*
            
            This prevalence urged us to explore a solution that creates awareness and early detection.

            """)
    with why_2:
        with st.container(border=True):
            st.markdown("### Practicality & Quality")
            st.success("**High Feature Richness**")
            st.markdown("""
            We selected this dataset because it already contains essential risk factors, allowing us to focus on Model Architecture:
            
            * **Demographics** (Age, Gender)
            * **Behaviors** (Smoking, Diet)
            * **Clinical** (BMI, BP)
            
            Its public availability ensures our research is reproducible and transparent.
            """)
    with why_3:
        with st.container(border=True):
            st.markdown("### Domain Interest")
            st.info("**HealthTech Focus**")
            st.markdown("""
            Our group shares a distinct interest in HealthTech—the intersection of technology and medicine. 
            
            While we are Computer Science students, the medical field has always drawn our curiosity. This dataset served as the perfect foundation for our goal of solving human-centric health problems.
            """)