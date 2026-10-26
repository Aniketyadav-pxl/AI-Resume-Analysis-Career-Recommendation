"""Streamlit entry point:  streamlit run app.py"""
import logging

import streamlit as st

from ui import analyzer_page, dataset_page, gap_page, history_page, home, jobs_page, model_page, recommendations_page

st.set_page_config(page_title="Resume AI", page_icon=":page_facing_up:", layout="wide")

PAGES = {
    "Home": home.render,
    "Resume Analyzer": analyzer_page.render,
    "Career Recommendations": recommendations_page.render,
    "Job Matching": jobs_page.render,
    "Skill Gap": gap_page.render,
    "History": history_page.render,
    "Model Insights": model_page.render,
    "Dataset Explorer": dataset_page.render,
}

with st.sidebar:
    st.header("Resume AI")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")
    st.divider()
    st.caption("Runs locally. Synthetic training data only.")
    st.markdown("Made by **Aniket Yadav**  \n[LinkedIn](https://www.linkedin.com/in/aniket-yadav07)")

try:
    PAGES[choice]()
except Exception:
    logging.getLogger("resume_ai").exception("Unhandled error on page %s", choice)
    st.error("This page hit an unexpected error. Reload the page or go back to Home.")
