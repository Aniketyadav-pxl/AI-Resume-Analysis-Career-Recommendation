import plotly.graph_objects as go
import streamlit as st

import config
from src import database as db
from src.knowledge_base import display_name
from src.pipeline import analyze_file, analyze_resume
from src.text_extractor import ExtractionError
from ui import common


def _run(label, fn):
    try:
        with st.spinner(f"Analysing {label} ..."):
            bundle, _ = common.get_bundle()
            st.session_state["analysis"] = fn(bundle, common.get_matcher())
            st.session_state["saved_id"] = None
    except ExtractionError as exc:
        st.error(str(exc))
    except Exception:  # never show a raw traceback to the user
        st.error("Something went wrong while analysing this resume. Please try another file.")


def _gauge(score, grade):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": grade},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4F46E5"},
               "steps": [{"range": [0, 50], "color": "#FEE2E2"}, {"range": [50, 70], "color": "#FEF3C7"},
                         {"range": [70, 85], "color": "#DCFCE7"}, {"range": [85, 100], "color": "#BBF7D0"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _breakdown(q):
    names = list(q.breakdown)
    got = [q.breakdown[n][0] for n in names]
    mx = [q.breakdown[n][1] for n in names]
    fig = go.Figure()
    fig.add_bar(y=names, x=mx, orientation="h", name="Maximum", marker_color="#E5E7EB")
    fig.add_bar(y=names, x=got, orientation="h", name="Your score", marker_color="#4F46E5")
    fig.update_layout(barmode="overlay", height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.2))
    return fig


def _show(r):
    for w in r.warnings:
        st.warning(w)
    st.success(f"Analysed **{r.filename}** ({r.file_type.upper()}, {r.n_pages} page(s), {r.word_count} words)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality score", f"{r.quality.score:.0f} / 100")
    c2.metric("Technical skills", len(r.skills.technical))
    c3.metric("Best-fit role", r.recommendations[0].role)
    c4.metric("Model confidence", f"{r.recommendations[0].probability:.0%}")

    left, right = st.columns(2)
    left.plotly_chart(_gauge(r.quality.score, r.quality.grade))
    right.markdown("**Score breakdown**")
    right.plotly_chart(_breakdown(r.quality))
    st.caption("The quality score is a transparent rule-based heuristic (structure, skills, impact, contact, length, formatting). "
               "The machine learning model is used for the career recommendations.")

    st.subheader("How to improve this resume")
    if r.quality.suggestions:
        for tip in r.quality.suggestions:
            st.markdown(f"- {tip}")
    else:
        st.success("No major issues found.")

    st.subheader("Detected skills")
    if r.skills.by_category:
        for cat, items in r.skills.by_category.items():
            st.markdown(f"**{cat}:** " + " ".join(f"`{display_name(s)}`" for s in items))
    else:
        st.write("No technical skills detected.")
    if r.skills.soft:
        st.markdown("**Soft skills:** " + " ".join(f"`{s.title()}`" for s in r.skills.soft))

    st.divider()
    saved = st.session_state.get("saved_id")
    if saved:
        st.success(f"Saved to history (analysis #{saved}). Only scores and skills are stored, never the resume text.")
    elif st.button("Save to history", key="save_history"):
        try:
            st.session_state["saved_id"] = db.save_analysis(r.to_record())
            st.rerun()
        except Exception:
            st.error("Could not save to the local database.")
    st.caption("Next: see **Career Recommendations**, **Job Matching** and **Skill Gap** in the sidebar.")


def render():
    st.title("Resume Analyzer")
    st.caption("Everything runs locally. Files are processed in memory and never uploaded anywhere else.")
    tab_up, tab_sample, tab_paste = st.tabs(["Upload", "Sample resumes", "Paste text"])

    with tab_up:
        f = st.file_uploader("PDF, DOCX or TXT (max 5 MB)", type=config.ALLOWED_UPLOAD_TYPES, key="uploader")
        if st.button("Analyze uploaded resume", key="analyze_upload", disabled=f is None):
            data = f.getvalue()
            _run(f.name, lambda b, m: analyze_file(data, f.name, b, m))

    with tab_sample:
        samples = sorted(p for p in config.SAMPLE_DIR.glob("*.*") if p.suffix.lower() in (".pdf", ".docx", ".txt"))
        if not samples:
            st.warning("No sample resumes found. Run: python -m src.make_samples")
        else:
            st.caption("Synthetic or fictional resumes - no real people.")
            choice = st.selectbox("Choose a sample", samples, format_func=lambda p: p.name, key="sample_choice")
            if st.button("Analyze sample", key="analyze_sample"):
                _run(choice.name, lambda b, m: analyze_file(choice.read_bytes(), choice.name, b, m))

    with tab_paste:
        text = st.text_area("Paste resume text", height=250, key="paste_text")
        if st.button("Analyze pasted text", key="analyze_paste"):
            if len(text.strip()) < 50:
                st.error("Please paste at least a few lines of resume text.")
            else:
                _run("pasted text", lambda b, m: analyze_resume(text, b, m))

    result = common.current_analysis()
    if result is not None:
        st.divider()
        _show(result)
