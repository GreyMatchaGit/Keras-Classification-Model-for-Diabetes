import streamlit as st
from pathlib import Path
import os


def display_clinic_model_architecture(PROJECT_ROOT: Path):
    assets_path = Path(PROJECT_ROOT + os.sep + "src" + os.sep + "assets")
    v0_graph_name = "accuracyloss_graph_v1.0.0.png"
    v1_graph_name = "accuracyloss_graph_v1.0.1.png"
    v2_graph_name = "accuracyloss_graph_v1.0.2.png"

    st.markdown("## Clinic Side Model v1.0.0 Architecture")
    st.image(Path(f"{assets_path.absolute()}{os.sep}{v0_graph_name}"))
    st.markdown("## Clinic Side Model v1.0.1 Architecture")
    st.image(Path(f"{assets_path.absolute()}{os.sep}{v1_graph_name}"))
    st.markdown("## Clinic Side Model v1.0.2 Architecture")
    st.image(Path(f"{assets_path.absolute()}{os.sep}{v2_graph_name}"))
    pass