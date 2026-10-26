import pandas as pd
import streamlit as st

import config
from src.knowledge_base import ROLES, SKILLS
from ui import common


def render():
    st.title("AI-Based Resume Analysis and Career Recommendation System")
    st.caption("Semester V Machine Learning project - runs locally, trained on synthetic data")

    m = common.get_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Career roles", len(ROLES))
    c2.metric("Skills in knowledge base", len(SKILLS))
    c3.metric("Hold-out accuracy", f"{m['holdout_test']['accuracy']:.1%}" if m else "n/a")
    c4.metric("Top-3 accuracy", f"{m['holdout_test']['top3_accuracy']:.1%}" if m else "n/a")

    st.subheader("Get started")
    st.markdown("Open **Resume Analyzer** in the sidebar, pick a sample resume (or upload your own) and click *Analyze*. "
                "Then explore Career Recommendations, Job Matching and Skill Gap.")

    st.subheader("Project workflow")
    st.markdown(
        "`Upload resume` -> `Extract text (PyMuPDF / python-docx)` -> `spaCy preprocessing` -> `Skill extraction` -> "
        "`TF-IDF + trained classifier` -> `Top-3 careers` -> `Job matching` -> `Skill gap` -> `SQLite history`")

    st.subheader("Build status")
    status = pd.DataFrame(config.MODULES, columns=["ID", "Module", "Status"])
    status["Status"] = status["Status"].map({"done": "Done", "pending": "Pending"})
    st.dataframe(status, hide_index=True)
