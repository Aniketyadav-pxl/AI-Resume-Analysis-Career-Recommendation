"""Shared Streamlit helpers: cached resources and session state access."""
import json

import streamlit as st

import config
from src.job_matcher import JobMatcher, load_jobs
from src.model_store import load_bundle


@st.cache_resource(show_spinner="Loading career model ...")
def get_bundle():
    return load_bundle()  # (bundle, status)


@st.cache_resource(show_spinner="Preparing job index ...")
def get_matcher():
    return JobMatcher(load_jobs())


@st.cache_data
def get_metrics():
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return None


def current_analysis():
    return st.session_state.get("analysis")


def require_analysis():
    """Return the current AnalysisResult, or show a hint and return None."""
    result = current_analysis()
    if result is None:
        st.info("No resume analysed yet. Go to **Resume Analyzer**, upload a resume (or pick a sample) and click Analyze.")
    return result
