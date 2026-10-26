import pandas as pd

import config
from src.preprocessing import normalize_text, prepare_many, prepare_model_text, protect_skills
from src.sections import split_sections
from src.skill_extractor import evaluate_extractor, extract_skills, slug


def test_sections_detected_including_inline_headers():
    text = "Name\n\nSUMMARY\nHi\n\nSkills: Python, SQL\nmore\n\nWork Experience\n- did things\n\nEDUCATION\nBCA"
    names = [s.name for s in split_sections(text)]
    assert names == ["header", "summary", "skills", "experience", "education"]


def test_aliases_and_case_insensitivity():
    r = extract_skills("SKILLS\nPython, JS, Sklearn, K8s, node, MS Excel, CI/CD, C++, Node.js, React.js")
    assert {"python", "javascript", "scikit-learn", "kubernetes", "node.js", "excel", "ci/cd", "c++", "react"} <= set(r.technical)


def test_longest_match_wins_and_no_substring_hits():
    r = extract_skills("SKILLS\nJavaScript, GitHub Actions, Spring Boot, Java")
    assert "java" in r.technical and "javascript" in r.technical
    assert "github actions" in r.technical and "git" not in r.technical
    assert extract_skills("SKILLS\nJavaScript").technical == ["javascript"]
    assert extract_skills("SKILLS\nPythonic reactor").technical == []


def test_ambiguous_terms_need_skills_section():
    prose = "SUMMARY\nI excel in teamwork and will express interest. Updated my CV and ML pipeline."
    assert extract_skills(prose).technical == []
    assert set(extract_skills("SKILLS\nExcel, ML, R, Express").technical) == {"excel", "machine learning", "r", "express"}


def test_headerless_text_uses_capitalisation_rule():
    r = extract_skills("I use Excel daily and also python for scripts. I excel at maths.")
    assert "excel" in r.technical and "python" in r.technical
    assert extract_skills("I excel at maths").technical == []


def test_soft_skills():
    assert {"teamwork", "communication"} <= set(extract_skills("Good communication and teamwork").soft)


def test_extractor_accuracy_on_synthetic_ground_truth():
    df = pd.read_csv(config.SYNTHETIC_RESUMES_CSV).head(400)
    m = evaluate_extractor(df)
    print(m)
    assert m["recall"] >= 0.97 and m["precision"] >= 0.95


def test_protect_and_lemmatise():
    out = prepare_model_text("SKILLS\nPandas, scikit-learn, C++\nEXPERIENCE\n- Developed models and analysing datasets. Email: a@example.com")
    toks = out.split()
    assert {"sk_pandas", "sk_scikit_learn", "sk_cpp"} <= set(toks)
    assert "panda" not in toks and "example" not in out
    assert "develop" in toks or "developed" in toks


def test_slug_and_normalize():
    assert slug("c#") == "sk_csharp" and slug("node.js") == "sk_node_js" and slug("ci/cd") == "sk_ci_cd"
    assert normalize_text("\u2022 Item\u200b  here") == "- Item here"
    assert "sk_python" in protect_skills("SKILLS\nPython")


def test_prepare_many_matches_single():
    texts = ["SKILLS\nPython, Docker", "EXPERIENCE\nBuilt APIs with Flask"]
    assert prepare_many(texts) == [prepare_model_text(t) for t in texts]


def test_category_label_is_not_a_skill():
    assert extract_skills("SKILLS\nData/ML: Pandas, NumPy").technical == ["numpy", "pandas"]
