import pandas as pd
import streamlit as st

import config
from src.knowledge_base import ROLES, SKILLS
from ui import common

AUTHOR = "Aniket Yadav"
LINKEDIN = "https://www.linkedin.com/in/aniket-yadav07"

CSS = """
<style>
.hero{padding:2.6rem 2rem;border-radius:22px;color:#fff;margin-bottom:1.2rem;
background:radial-gradient(circle at 15% 20%,#7c3aed 0,transparent 45%),
radial-gradient(circle at 85% 80%,#06b6d4 0,transparent 45%),linear-gradient(135deg,#312e81,#4f46e5 55%,#0ea5e9);
box-shadow:0 14px 40px rgba(79,70,229,.35)}
.hero h1{font-size:2.5rem;line-height:1.15;margin:.2rem 0 .6rem 0;color:#fff;font-weight:800}
.hero p{font-size:1.1rem;opacity:.92;max-width:760px;margin:0 0 1rem 0}
.pill{display:inline-block;padding:.25rem .8rem;margin:.15rem .25rem .15rem 0;border-radius:999px;font-size:.82rem;
background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);backdrop-filter:blur(4px)}
.eyebrow{letter-spacing:.14em;text-transform:uppercase;font-size:.78rem;opacity:.85}
.card{padding:1.1rem 1.2rem;border-radius:16px;border:1px solid rgba(128,128,160,.28);
background:rgba(128,128,160,.08);height:100%;transition:transform .15s,box-shadow .15s}
.card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(79,70,229,.18)}
.card h4{margin:.1rem 0 .35rem 0}.card p{margin:0;opacity:.85;font-size:.93rem}
.step{display:inline-block;padding:.45rem .9rem;margin:.2rem;border-radius:12px;font-weight:600;font-size:.88rem;
background:linear-gradient(135deg,rgba(79,70,229,.18),rgba(14,165,233,.18));border:1px solid rgba(79,70,229,.35)}
.arrow{opacity:.5;margin:0 .1rem}
.credit{margin-top:2rem;padding:1.4rem;border-radius:18px;text-align:center;border:1px solid rgba(128,128,160,.28);
background:linear-gradient(135deg,rgba(79,70,229,.10),rgba(14,165,233,.10))}
.credit .by{font-size:1.15rem;font-weight:700}
.credit a.li{display:inline-block;margin-top:.7rem;padding:.5rem 1.2rem;border-radius:999px;background:#0a66c2;color:#fff!important;
text-decoration:none;font-weight:600;font-size:.92rem}
.credit a.li:hover{background:#004182}
</style>
"""

FEATURES = [
    ("📄", "Smart resume parsing", "Reads PDF, DOCX and TXT files and detects sections and 94 skills, including aliases like sklearn and k8s."),
    ("🧠", "ML career prediction", "A trained TF-IDF classifier ranks your top 3 best-fit roles with real probabilities."),
    ("⭐", "Quality score", "A 0-100 score with clear, actionable tips to improve structure, impact and keywords."),
    ("💼", "Job matching", "Ranks job postings using TF-IDF cosine similarity plus weighted skill coverage."),
    ("🎯", "Skill gap plan", "Shows exactly which skills you are missing for a target role, in learning order."),
    ("🗂️", "Private history", "Your results are saved locally in SQLite. Resume text is never stored."),
]
STEPS = ["Upload", "Extract text", "spaCy NLP", "Skills", "TF-IDF + ML", "Top-3 careers", "Jobs", "Skill gap"]
STACK = ["Python", "Streamlit", "scikit-learn", "spaCy", "pandas", "NumPy", "SQLite", "PyMuPDF", "Plotly"]


def _go(page: str):
    st.session_state["nav"] = page


def render():
    st.markdown(CSS, unsafe_allow_html=True)

    pills = "".join(f'<span class="pill">{t}</span>' for t in STACK)
    st.markdown(
        '<div class="hero"><div class="eyebrow">Semester V · Machine Learning project</div>'
        "<h1>AI Resume Analysis &amp; Career Recommendation</h1>"
        "<p>Upload your resume and let a trained machine learning model find your best-fit careers, "
        "matching jobs and the skills you should learn next.</p>"
        f"{pills}</div>", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1.3, 1.3, 4])
    b1.button("🚀 Analyze my resume", type="primary", on_click=_go, args=("Resume Analyzer",), key="cta_analyze")
    b2.button("📊 See model results", on_click=_go, args=("Model Insights",), key="cta_model")

    m = common.get_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Career roles", len(ROLES))
    c2.metric("Skills tracked", len(SKILLS))
    c3.metric("Test accuracy", f"{m['holdout_test']['accuracy']:.1%}" if m else "n/a")
    c4.metric("Top-3 accuracy", f"{m['holdout_test']['top3_accuracy']:.1%}" if m else "n/a")
    st.caption("Metrics are measured on synthetic resumes; see Model Insights for baselines and limitations.")

    st.markdown("### What you get")
    for row in (FEATURES[:3], FEATURES[3:]):
        cols = st.columns(3)
        for col, (icon, title, text) in zip(cols, row):
            col.markdown(f'<div class="card"><h4>{icon} {title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
        st.write("")

    st.markdown("### How it works")
    st.markdown('<span class="arrow">→</span>'.join(f'<span class="step">{s}</span>' for s in STEPS), unsafe_allow_html=True)

    with st.expander("Project build status"):
        status = pd.DataFrame(config.MODULES, columns=["ID", "Module", "Status"])
        status["Status"] = status["Status"].map({"done": "Done", "pending": "Pending"})
        st.dataframe(status, hide_index=True)

    st.markdown(
        f'<div class="credit"><div class="by">Made with ❤️ by {AUTHOR}</div>'
        f'<div style="opacity:.75;margin-top:.2rem">BCA (AI &amp; ML) · Galgotias University</div>'
        f'<a class="li" href="{LINKEDIN}" target="_blank" rel="noopener noreferrer">🔗 Connect on LinkedIn</a></div>',
        unsafe_allow_html=True)
