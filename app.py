"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

from ui import dataset_page, home, placeholder

st.set_page_config(page_title="Resume AI", page_icon=":page_facing_up:", layout="wide")

PAGES = {
    "Home": lambda: home.render(),
    "Dataset Explorer": lambda: dataset_page.render(),
    "Resume Analyzer": lambda: placeholder.render(
        "Resume Analyzer", "M1-M3", "Upload a PDF/DOCX/TXT and get extracted skills, sections and a quality score."),
    "Career Recommendations": lambda: placeholder.render(
        "Career Recommendations", "M4-M5", "Trained classifier predicts the best-fit roles with probabilities."),
    "Job Matching": lambda: placeholder.render(
        "Job Matching", "M6", "Rank job postings by cosine similarity with your resume."),
    "Skill Gap": lambda: placeholder.render(
        "Skill Gap Analysis", "M7", "Compare your skills with a target role and get a learning list."),
    "History": lambda: placeholder.render(
        "History", "M8", "Past analyses stored locally in SQLite."),
}

with st.sidebar:
    st.header("Resume AI")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Runs 100% locally. Synthetic data only.")

PAGES[choice]()
