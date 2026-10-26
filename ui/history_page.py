import pandas as pd
import streamlit as st

from src import database as db


def render():
    st.title("History")
    try:
        rows = db.list_analyses()
    except Exception:
        st.error("The local history database could not be opened.")
        return
    if not rows:
        st.info("No saved analyses yet. Analyse a resume and click **Save to history**.")
        return

    df = pd.DataFrame([{"ID": r["id"], "Date (UTC)": r["created_at"].replace("T", " ")[:16], "File": r["filename"],
                        "Quality": r["quality_score"], "Top role": r["top_role"], "Confidence": f"{r['top_probability']:.0%}"} for r in rows])
    st.dataframe(df, hide_index=True)
    if len(rows) >= 2:
        st.subheader("Quality score over time")
        st.line_chart(df.sort_values("ID").set_index("ID")["Quality"])

    st.subheader("Inspect or delete")
    pick = st.selectbox("Analysis", [r["id"] for r in rows], key="hist_pick")
    rec = db.get_analysis(pick)
    if rec:
        st.markdown(f"**{rec['filename']}** - quality {rec['quality_score']:.0f}/100 - best fit **{rec['top_role']}**")
        st.write("Top 3 roles:", ", ".join(f"{x['role']} ({x['probability']:.0%})" for x in rec["recommendations"]))
        st.write("Technical skills:", ", ".join(rec["skills"]["technical"]) or "-")
        g = rec["gap"]
        st.write(f"Readiness for {g['role']}: {g['readiness']:.0f}%. Missing core: {', '.join(g['core_missing']) or '-'}")
        st.write("Top jobs:", "; ".join(f"{j['title']} @ {j['company']} ({j['score']:.0f}%)" for j in rec["top_jobs"]))
        if st.button("Delete this analysis", key="hist_delete"):
            db.delete_analysis(pick)
            st.rerun()
    with st.expander("Danger zone"):
        sure = st.checkbox("I understand this deletes ALL history", key="hist_sure")
        if st.button("Clear all history", key="hist_clear", disabled=not sure):
            db.clear_all()
            st.rerun()
