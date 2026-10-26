"""Headless UI tests using Streamlit's AppTest (no browser needed)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Home", "Resume Analyzer", "Career Recommendations", "Job Matching", "Skill Gap",
         "History", "Model Insights", "Dataset Explorer"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_AI_DB", str(tmp_path / "ui.db"))
    return AppTest.from_file(APP, default_timeout=60).run()


def goto(at, page):
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    return at


def test_every_page_renders_before_any_analysis(app):
    assert not app.exception
    for page in PAGES:
        goto(app, page)
    goto(app, "Career Recommendations")
    assert any("No resume analysed" in i.value for i in app.info)


def test_full_user_journey(app):
    goto(app, "Resume Analyzer")
    box = app.selectbox(key="sample_choice")
    idx = [i for i, o in enumerate(box.options) if "handwritten_devops" in str(o)][0]
    box.select_index(idx)
    app.button(key="analyze_sample").click().run()
    assert not app.exception and not app.error
    assert app.session_state["analysis"].recommendations[0].role == "DevOps Engineer"

    for page in ["Career Recommendations", "Job Matching", "Skill Gap", "Model Insights"]:
        goto(app, page)
        assert not app.error, page

    goto(app, "Resume Analyzer")
    app.button(key="save_history").click().run()
    assert not app.exception and app.session_state["saved_id"]
    goto(app, "History")
    assert len(app.dataframe) >= 1


def test_paste_text_validation_and_success(app):
    goto(app, "Resume Analyzer")
    app.button(key="analyze_paste").click().run()
    assert any("paste at least" in e.value for e in app.error)
    app.text_area(key="paste_text").set_value(
        "SKILLS\nPython, Pandas, scikit-learn, SQL, Statistics\nEXPERIENCE\n- Built models with Python.\nEDUCATION\nBCA").run()
    app.button(key="analyze_paste").click().run()
    assert not app.exception and app.session_state["analysis"] is not None
