import pandas as pd
import streamlit as st

from src.knowledge_base import display_name
from ui import common


def render():
    st.title("Job Matching")
    r = common.require_analysis()
    if r is None:
        return
    matcher = common.get_matcher()
    locations = ["Any"] + sorted(matcher.jobs["location"].unique())
    c1, c2, c3 = st.columns(3)
    loc = c1.selectbox("Location", locations, key="job_loc")
    min_score = c2.slider("Minimum match %", 0, 100, 0, key="job_min")
    top_n = c3.slider("Show top", 3, 30, 10, key="job_top")

    jobs = [j for j in r.job_matches if (loc == "Any" or j.location == loc) and j.match_score >= min_score][:top_n]
    if not jobs:
        st.warning("No jobs match these filters. Try lowering the minimum match or changing location.")
        return

    st.dataframe(pd.DataFrame([{
        "Match %": j.match_score, "Title": j.title, "Company": j.company, "Location": j.location,
        "Min. exp (yrs)": j.min_experience, "Required skills matched": f"{len(j.matched_required)}/{len(j.matched_required) + len(j.missing_required)}",
    } for j in jobs]), hide_index=True)

    st.subheader("Details")
    for j in jobs[:8]:
        with st.expander(f"{j.match_score:.0f}%  -  {j.title} @ {j.company} ({j.location})"):
            st.progress(min(1.0, j.match_score / 100))
            st.caption(f"TF-IDF cosine similarity: {j.cosine:.2f}  |  weighted skill coverage: {j.coverage:.0%}")
            fmt = lambda xs: " ".join(f"`{display_name(s)}`" for s in xs) or "-"
            st.markdown(f"**You have (required):** {fmt(j.matched_required)}")
            st.markdown(f"**Missing (required):** {fmt(j.missing_required)}")
            st.markdown(f"**Nice-to-have you have:** {fmt(j.matched_nice)}")
    st.caption("Match % = 60% weighted skill coverage + 40% TF-IDF cosine similarity (normalised). Job postings are synthetic.")
