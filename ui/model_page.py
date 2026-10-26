import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from ui import common


@st.cache_data
def _extractor_eval():
    from src.skill_extractor import evaluate_extractor
    return evaluate_extractor(pd.read_csv(config.SYNTHETIC_RESUMES_CSV).head(300))


def render():
    st.title("Model Insights")
    m = common.get_metrics()
    if not m:
        st.warning("No metrics found. Train the model first: `python -m src.train`")
        return
    h, ind, base = m["holdout_test"], m["independent_synthetic_test"], m["baselines"]

    st.info("**Read this first:** all data is synthetic and generated from the same skill lists the model learns from, "
            "so these scores measure performance on *this* kind of resume, not on real-world resumes.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (hold-out)", f"{h['accuracy']:.1%}")
    c2.metric("Macro F1", f"{h['f1_macro']:.3f}")
    c3.metric("Top-3 accuracy", f"{h['top3_accuracy']:.1%}")
    c4.metric("Log loss", f"{h['log_loss']:.3f}")
    st.caption(f"Selected model: **{m['best_model']}** | trained on {m['dataset']['n_train']} resumes, tested on {m['dataset']['n_test']} unseen | trained {m['trained_at'][:10]}")

    t1, t2, t3, t4 = st.tabs(["Model comparison", "Per-role results", "What the model learned", "Limitations"])
    with t1:
        cv = pd.DataFrame(m["cv_5fold_on_train"]).T.reset_index(names="Model")
        cv["Accuracy (5-fold CV)"] = cv.apply(lambda r: f"{r.accuracy_mean:.3f} +/- {r.accuracy_std:.3f}", axis=1)
        cv["Macro F1 (5-fold CV)"] = cv.apply(lambda r: f"{r.f1_macro_mean:.3f} +/- {r.f1_macro_std:.3f}", axis=1)
        st.dataframe(cv[["Model", "Accuracy (5-fold CV)", "Macro F1 (5-fold CV)"]], hide_index=True)
        bars = {"Majority class": base["majority_class_accuracy"], "Rule-based skill overlap (no ML)": base["rule_based_core_skill_overlap"]["accuracy"],
                f"{m['best_model']} (hold-out)": h["accuracy"], f"{m['best_model']} (independent set)": ind["accuracy"]}
        fig = go.Figure(go.Bar(x=list(bars), y=list(bars.values()), marker_color=["#9CA3AF", "#9CA3AF", "#4F46E5", "#818CF8"]))
        fig.update_layout(height=300, yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=10, r=10, t=30, b=10), title="Accuracy vs baselines")
        st.plotly_chart(fig)
        st.caption("The ML model must beat the rule-based baseline to justify using machine learning.")
    with t2:
        cm = m["confusion_matrix"]
        fig = px.imshow(cm["matrix"], x=cm["labels"], y=cm["labels"], text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        pc = pd.DataFrame(m["per_class"]).T.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        st.dataframe(pc.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Test samples"}))
    with t3:
        st.write("Highest-weight TF-IDF terms per role (from a logistic regression; `sk_` prefix removed):")
        role = st.selectbox("Role", list(m["top_terms_per_role"]), key="terms_role")
        st.write(", ".join(m["top_terms_per_role"][role]))
        e = _extractor_eval()
        st.markdown(f"**Skill extractor** vs ground truth on {e['n_resumes']} resumes: precision {e['precision']:.3f}, recall {e['recall']:.3f}, F1 {e['f1']:.3f}.")
    with t4:
        for item in m["limitations"]:
            st.markdown(f"- {item}")
