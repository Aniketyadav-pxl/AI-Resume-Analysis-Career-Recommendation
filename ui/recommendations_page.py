import plotly.graph_objects as go
import streamlit as st

from src.knowledge_base import display_name
from src.recommender import confidence_label
from ui import common


def render():
    st.title("Career Recommendations")
    r = common.require_analysis()
    if r is None:
        return
    recs = r.recommendations
    label = confidence_label(recs[0].probability, recs[1].probability)
    st.markdown(f"Model confidence: **{label}** (top probability {recs[0].probability:.0%}, runner-up {recs[1].probability:.0%})")
    if label == "Low":
        st.warning("The model is unsure. The profile may be mixed or the resume may list few skills. Treat the top 3 as options, not a verdict.")

    cols = st.columns(3)
    for i, (col, rec) in enumerate(zip(cols, recs), start=1):
        with col:
            st.subheader(f"#{i} {rec.role}")
            st.progress(min(1.0, rec.probability), text=f"Probability {rec.probability:.1%}")
            st.caption(rec.description)
            st.markdown(f"**Core skill coverage:** {rec.core_coverage:.0%}")
            if rec.matched_core:
                st.markdown("You have: " + " ".join(f"`{display_name(s)}`" for s in rec.matched_core))
            if rec.missing_core:
                st.markdown("Missing core: " + " ".join(f"`{display_name(s)}`" for s in rec.missing_core))

    st.subheader("Probability across all roles")
    items = sorted(r.probabilities.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[p for _, p in items], y=[k for k, _ in items], orientation="h", marker_color="#4F46E5"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig)

    with st.expander("How the model works"):
        _, status = common.get_bundle()
        m = common.get_metrics()
        st.markdown(
            "1. The resume text is cleaned with spaCy and skills are converted to protected tokens.\n"
            "2. **TF-IDF** turns the text into a weighted numeric vector (1-2 word phrases).\n"
            "3. A trained classifier outputs a **probability for each of the 10 roles**; the top 3 are shown.\n\n"
            "The model was trained on synthetic resumes only, so treat results as a demonstration, not career advice.")
        if m:
            h = m["holdout_test"]
            st.markdown(f"Model: **{m['best_model']}** | hold-out accuracy {h['accuracy']:.1%}, macro-F1 {h['f1_macro']:.3f}, top-3 accuracy {h['top3_accuracy']:.1%} (synthetic data).")
        st.caption(f"Model status: {status}")
