import streamlit as st


def render(title: str, module_id: str, description: str):
    st.title(title)
    st.info(f"Module {module_id} is not built yet. {description}")
