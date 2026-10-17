# AI-Based Resume Analysis and Career Recommendation System

Semester V Machine Learning project. Runs fully locally; uses synthetic resumes only.

## Setup (Windows / macOS / Linux)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.data_generator --n 1500     # creates data/synthetic_resumes.csv
python -m pytest -q                       # all tests should pass
streamlit run app.py                      # opens http://localhost:8501
```

## Structure
```
resume_career_ai/
├── app.py                  Streamlit entry point + navigation
├── config.py               paths, constants, module status registry
├── requirements.txt
├── data/                   synthetic_resumes.csv, jobs.csv (M6), app.db (M8)
├── models/                 trained .joblib + metrics.json (M4)
├── sample_resumes/         synthetic PDF/DOCX samples for demo (M1)
├── src/
│   ├── knowledge_base.py   94 skills + aliases, 10 roles (core/secondary skills)
│   ├── data_generator.py   synthetic dataset generator
│   ├── text_extractor.py   (M1) PDF/DOCX/TXT -> text
│   ├── preprocessing.py    (M2) spaCy cleaning, lemmatisation
│   ├── skill_extractor.py  (M2) taxonomy + alias matching
│   ├── analyzer.py         (M3) sections, completeness score
│   ├── train.py            (M4) training + evaluation + joblib export
│   ├── recommender.py      (M5) predict_proba -> top-N roles
│   ├── job_matcher.py      (M6) TF-IDF cosine similarity
│   ├── skill_gap.py        (M7) missing core/secondary skills
│   └── database.py         (M8) SQLite
├── ui/                     one file per Streamlit page
└── tests/                  pytest per module
```
