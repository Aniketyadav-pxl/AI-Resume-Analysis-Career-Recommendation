import pandas as pd
import streamlit as st

import config
from src.knowledge_base import ROLES, SKILLS


def render():
    st.title("AI-Based Resume Analysis and Career Recommendation System")
    st.caption("Semester V Machine Learning project - local, privacy-friendly, trained on synthetic data")

    c1, c2, c3 = st.columns(3)
    c1.metric("Career roles", len(ROLES))
    c2.metric("Skills in knowledge base", len(SKILLS))
    done = sum(1 for _, _, s in config.MODULES if s == "done")
    c3.metric("Modules completed", f"{done} / {len(config.MODULES)}")

    st.subheader("Project workflow")
    st.markdown(
        "`Upload resume` -> `Extract text` -> `NLP preprocessing` -> `Skill extraction` -> "
        "`TF-IDF features + trained classifier` -> `Career recommendations` -> "
        "`Job matching` -> `Skill gap analysis` -> `Dashboard + SQLite history`"
    )

    st.subheader("Build status")
    status = pd.DataFrame(config.MODULES, columns=["ID", "Module", "Status"])
    status["Status"] = status["Status"].map({"done": "Done", "pending": "Pending"})
    st.dataframe(status, hide_index=True, use_container_width=True)
