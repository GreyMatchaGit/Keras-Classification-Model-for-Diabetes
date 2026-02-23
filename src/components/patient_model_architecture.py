import streamlit as st
from pathlib import Path
import os



def display_patient_model_architecture(PROJECT_ROOT: Path):
    assets_path = Path(PROJECT_ROOT + os.sep + "src" + os.sep + "assets")
    v0_graph_name = "accuracyloss_graph.png"
    v1_model_architecture = "bpm.png"
    v2_model_architecture = "bpm1.png"

    st.markdown("## Patient Side Model v1.0.0")
    st.image(Path(f"{assets_path.absolute()}{os.sep}{v0_graph_name}"))
    st.markdown("## Model Architecture")
    st.image(Path(f"{assets_path.absolute()}{os.sep}{v1_model_architecture}"))
    st.image(Path(f"{assets_path.absolute()}{os.sep}{v2_model_architecture}"))
    pass