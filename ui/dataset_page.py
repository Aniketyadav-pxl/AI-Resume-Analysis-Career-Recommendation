import pandas as pd
import streamlit as st

import config


@st.cache_data
def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def render():
    st.title("Synthetic Dataset Explorer")
    st.info("All resumes here are synthetically generated. No real personal data is used.")

    if not config.SYNTHETIC_RESUMES_CSV.exists():
        st.warning("Dataset not found. Generate it with: `python -m src.data_generator --n 1500`")
        return

    df = _load(str(config.SYNTHETIC_RESUMES_CSV))
    c1, c2, c3 = st.columns(3)
    c1.metric("Resumes", len(df))
    c2.metric("Classes", df["target_role"].nunique())
    avg_skills = df["skills_truth"].str.split("|").str.len().mean()
    c3.metric("Avg. skills / resume", f"{avg_skills:.1f}")

    st.subheader("Class distribution")
    st.bar_chart(df["target_role"].value_counts())

    st.subheader("Browse a resume")
    role = st.selectbox("Filter by role", ["All"] + sorted(df["target_role"].unique()))
    view = df if role == "All" else df[df["target_role"] == role]
    pick = st.selectbox("Resume", view["resume_id"].tolist())
    row = view[view["resume_id"] == pick].iloc[0]
    left, right = st.columns([2, 1])
    left.text_area("Resume text", row["resume_text"], height=380)
    right.markdown(f"**Label:** {row['target_role']}")
    right.markdown(f"**Experience:** {row['experience_years']} yrs")
    right.markdown("**Ground-truth skills:**")
    right.write(", ".join(row["skills_truth"].split("|")))

    with st.expander("Dataset schema"):
        st.table(pd.DataFrame({
            "column": ["resume_id", "target_role", "experience_years", "education", "skills_truth", "resume_text"],
            "type": ["str", "str (label)", "int", "str", "str ('|' separated)", "str"],
            "purpose": ["Unique id", "Class to predict", "Numeric feature", "Categorical feature",
                        "Ground truth to evaluate skill extractor", "Input to the TF-IDF model"],
        }))
