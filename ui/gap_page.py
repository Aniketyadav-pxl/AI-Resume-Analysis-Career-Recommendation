import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.knowledge_base import ROLES, display_name
from src.skill_gap import analyze_gap
from ui import common


def render():
    st.title("Skill Gap Analysis")
    r = common.require_analysis()
    if r is None:
        return
    roles = list(ROLES)
    default = roles.index(r.recommendations[0].role)
    role = st.selectbox("Target role", roles, index=default, key="gap_role")
    gap = analyze_gap(role, r.skills)

    c1, c2, c3 = st.columns(3)
    c1.metric("Readiness", f"{gap.readiness:.0f}%")
    c2.metric("Core skills", f"{len(gap.core_have)}/{len(gap.core_have) + len(gap.core_missing)}")
    c3.metric("Nice-to-have", f"{len(gap.secondary_have)}/{len(gap.secondary_have) + len(gap.secondary_missing)}")
    st.progress(min(1.0, gap.readiness / 100), text=f"Readiness for {role}")
    st.caption(ROLES[role]["description"])

    fmt = lambda xs: " ".join(f"`{display_name(s)}`" for s in xs) or "-"
    left, right = st.columns(2)
    left.markdown(f"**Core skills you have**\n\n{fmt(gap.core_have)}")
    right.markdown(f"**Core skills missing**\n\n{fmt(gap.core_missing)}")
    left.markdown(f"**Nice-to-have you have**\n\n{fmt(gap.secondary_have)}")
    right.markdown(f"**Nice-to-have missing**\n\n{fmt(gap.secondary_missing)}")

    fig = go.Figure(go.Bar(
        x=["Core", "Nice-to-have"],
        y=[len(gap.core_have) / (len(gap.core_have) + len(gap.core_missing)),
           len(gap.secondary_have) / (len(gap.secondary_have) + len(gap.secondary_missing))],
        marker_color=["#4F46E5", "#A5B4FC"]))
    fig.update_layout(height=240, yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)

    st.subheader("Suggested learning order")
    if not gap.priorities:
        st.success("You cover every skill listed for this role.")
        return
    st.dataframe(pd.DataFrame([{"Priority": i, "Skill": display_name(p["skill"]), "Type": p["tier"],
                                "Category": p["category"], "What to do": p["suggestion"]}
                               for i, p in enumerate(gap.priorities, 1)]), hide_index=True)
    st.caption("Order: core skills first; within each group, skills useful for more roles come first.")
