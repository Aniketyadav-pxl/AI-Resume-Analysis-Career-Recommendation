# AI-Based Resume Analysis and Career Recommendation System

Semester V Machine Learning project. Upload a resume (PDF / DOCX / TXT) and get: extracted skills, a quality score
with improvement tips, **top-3 career recommendations from a trained ML model**, matching jobs, and a skill-gap
learning plan. Everything runs locally; the model is trained on **synthetic** resumes only (no real personal data).

## Quick start

**Requirements:** Python **3.11 or 3.12** (3.10 and older are not supported by the pinned scikit-learn).

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements-dev.txt
python -m pytest -q               # optional: 75 tests
streamlit run app.py              # opens http://localhost:8501
```
macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```
The spaCy model (`en_core_web_sm`) is installed by `requirements.txt` as a wheel, so there is no separate download step.
The trained model, dataset and sample resumes are committed, so the app works immediately.

### Regenerate everything from scratch (optional)
```
python -m src.data_generator --n 1500   # synthetic resumes  -> data/synthetic_resumes.csv
python -m src.job_generator             # synthetic jobs     -> data/jobs.csv
python -m src.make_samples              # TXT/DOCX/PDF samples -> sample_resumes/
python -m src.train                     # train + evaluate   -> models/*.joblib, models/metrics.json
```

## Project structure
```
app.py                  Streamlit entry point (sidebar navigation, global error guard)
config.py               paths, constants, module status
requirements.txt        runtime deps (deployment)      requirements-dev.txt  + pytest
.github/workflows/      CI: tests on Windows + Linux, Python 3.11 + 3.12
.streamlit/config.toml  theme, 5 MB upload limit
data/                   synthetic_resumes.csv, jobs.csv, app.db (runtime, git-ignored)
models/                 career_classifier.joblib, metrics.json
sample_resumes/         10 synthetic (TXT/DOCX/PDF) + 3 hand-written fictional resumes
src/
  knowledge_base.py     94 skills + aliases, 10 roles (core / secondary skills)
  text_extractor.py     M1  PDF/DOCX/TXT -> text, friendly errors
  sections.py           resume section detection
  skill_extractor.py    M2  alias-aware, section-aware skill extraction
  preprocessing.py      M2  spaCy cleaning + lemmatisation, protected skill tokens
  analyzer.py           M3  rule-based quality score (0-100) + suggestions
  train.py              M4  model comparison, evaluation, export
  model_store.py        M4  safe model loading / auto-retrain fallback
  recommender.py        M5  top-3 careers via predict_proba
  job_matcher.py        M6  TF-IDF cosine similarity + skill coverage
  skill_gap.py          M7  gap analysis + learning order
  database.py           M8  SQLite history (no resume text stored)
  pipeline.py           orchestrates M1-M8 for the UI and tests
  data_generator.py, job_generator.py, make_samples.py   synthetic data tools
ui/                     one file per Streamlit page
tests/                  75 pytest tests (per module + headless UI journey)
```

## How it works (workflow)
1. **Extract text** (PyMuPDF / python-docx / TXT with encoding fallback). Detects file type from magic bytes; rejects empty, oversized, password-protected, scanned (no text) and corrupt files with clear messages.
2. **Preprocess** with spaCy: skills are replaced by protected tokens (`sk_scikit_learn`) *before* lemmatisation so a lemmatiser can't turn "pandas" into "panda"; remaining words are lower-cased, lemmatised and stop-word filtered.
3. **Extract skills** by matching a 94-skill taxonomy with aliases (`sklearn`, `k8s`, `JS`). Ambiguous words (`excel`, `ml`, `cv`, `r`) only count inside a Skills section so "I excel in teamwork" is not a skill.
4. **Quality score** (rule-based, not ML): structure 25, skills 20, experience and impact 20, contact 10, length 10, breadth 5, formatting 10.
5. **Career classifier** (ML): TF-IDF (1-2 grams) -> classifier -> probability for each of 10 roles -> top 3.
6. **Job matching**: `score = 100 x (0.4 x normalised TF-IDF cosine + 0.6 x weighted skill coverage)`.
7. **Skill gap**: readiness = 70% core coverage + 30% nice-to-have coverage; missing skills ordered core-first, then by how many roles use them.
8. **History**: scores and skills saved to local SQLite (the resume text itself is never stored).

## Evaluation (real numbers from `models/metrics.json`)
Protocol: 1,500 synthetic resumes, stratified 80/20 split (1,200 train / 300 test). Four models compared with 5-fold CV on the training part; the winner by macro-F1 is evaluated once on the untouched test set and then refit on all data for deployment.

| Model | CV accuracy | CV macro-F1 |
|---|---|---|
| **Multinomial Naive Bayes (selected)** | 0.952 +/- 0.010 | 0.952 +/- 0.010 |
| Logistic Regression | 0.941 +/- 0.014 | 0.941 +/- 0.013 |
| Linear SVM (calibrated) | 0.938 +/- 0.012 | 0.938 +/- 0.011 |
| Random Forest | 0.893 +/- 0.016 | 0.892 +/- 0.016 |

| Test result | Value |
|---|---|
| Hold-out accuracy / macro-F1 | 0.947 / 0.946 |
| Top-3 accuracy | 1.000 |
| Independent synthetic set (500 new resumes, different seed) | accuracy 0.952, top-3 1.000 |
| Baseline: always predict one role | 0.100 |
| Baseline: rule-based core-skill overlap (no ML) | 0.713 |
| Skill extractor vs ground truth (400 resumes) | precision 0.997, recall 1.000 |

Most errors are between genuinely similar roles: DevOps vs Cybersecurity, Full Stack vs Frontend, Data Scientist vs Data Analyst. Open the **Model Insights** page for the confusion matrix, per-role scores and top terms per role.

### Limitations of synthetic data (be upfront about these)
- **The data is generated from the same skill lists the model learns from**, so scores measure in-distribution performance, not real-world accuracy. Real resumes are messier (layouts, typos, vague wording) and the true accuracy will be lower. The 3 hand-written resumes in `sample_resumes/` are a small, informal check of style shift (the right role appears in the top 3 for all of them); they are *not* a benchmark.
- **A development lesson:** the first version of the generator produced 100% accuracy, and diagnostics showed that project nouns alone *and* the skills section alone each predicted the role almost perfectly (data leakage by construction). The generator was changed to add career switchers, hybrids, noise skills and shared project themes; the numbers above are from the fixed version. `tests/test_train_and_recommender.py` fails if the macro-F1 is >= 0.995 as a guard.
- Only 10 roles exist, so every resume is forced into one of them. Probabilities are relative, not an "out of scope" detector.
- The quality score and job-match weights are transparent heuristics chosen by the developer, not learned.
- No fairness audit is possible with synthetic data. Do not use this tool for real hiring decisions.
- OCR is not supported (scanned PDFs are rejected with a message).

## Model loading strategy
`src/model_store.py` loads `models/career_classifier.joblib`. If the file is missing, corrupt, or was built with a different scikit-learn minor version or preprocessing version, it automatically retrains a fast logistic-regression model from the synthetic CSV (a few seconds) and tries to save it (read-only hosts just keep it in memory). `requirements.txt` pins `scikit-learn==1.8.0` so the committed model loads without retraining. If you retrain locally, commit the new `models/` files.

## Deploy to GitHub + Streamlit Community Cloud
1. Create a GitHub repo and push this folder (the `.gitignore` already keeps `models/`, `data/*.csv` and `sample_resumes/` tracked):
   ```
   git init && git add . && git commit -m "Resume AI project"
   git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
   ```
2. Go to https://share.streamlit.io -> **New app** -> pick the repo, branch `main`, main file `app.py`.
3. Open **Advanced settings** and select **Python 3.11 or 3.12** (Community Cloud's default may differ; the Python version is chosen there, not in `requirements.txt`).
4. Click **Deploy**. First build takes a few minutes (spaCy, scikit-learn). Keep only one dependency file in the repo root that Cloud might pick up (`requirements.txt`); `requirements-dev.txt` is ignored by Cloud.

Notes: Community Cloud's file system is ephemeral, so SQLite history resets when the app restarts. That is fine for a demo; locally the history persists in `data/app.db`.
GitHub Actions (`.github/workflows/tests.yml`) runs the test-suite on **Windows and Linux** for Python 3.11 and 3.12 on every push.

## Troubleshooting
- `ModuleNotFoundError: src` - run commands from the project root (the folder containing `app.py`).
- PowerShell refuses to activate the venv - `Set-ExecutionPolicy -Scope Process Bypass`, or use `.venv\Scripts\activate.bat` in cmd.
- Warning "spaCy model not found" - run `pip install -r requirements.txt` again (it installs the model wheel).
- Port already in use - `streamlit run app.py --server.port 8502`.

## Viva cheat sheet
**Algorithms**
- *TF-IDF*: term frequency x inverse document frequency. Words frequent in one resume but rare overall get high weight; sublinear TF (1+log tf) damps repeated words; 1-2 grams capture phrases like "machine learning".
- *Multinomial Naive Bayes*: applies Bayes' theorem assuming word counts are conditionally independent given the role; P(role | text) is proportional to P(role) x product of P(word | role). Fast, works well on small text data, gives probabilities.
- *Logistic Regression / Linear SVM*: linear models on TF-IDF; SVM is calibrated to output probabilities. *Random Forest*: many decision trees voting; weaker on sparse high-dimensional text.
- *Cosine similarity* (job matching): the angle between the resume vector and a job vector, independent of text length.
- *Cross-validation*: split training data into 5 folds, train on 4, validate on 1, rotate; the mean +/- std shows stability and is used for model selection so the test set stays untouched.

**Metrics**: *accuracy* = correct / total. *Precision* = of those predicted as a role, how many were right. *Recall* = of the true members of a role, how many were found. *F1* = harmonic mean of precision and recall; *macro* averages over roles equally. *Top-3 accuracy* = true role is among the 3 highest probabilities (matches how we present results). *Log loss* penalises confident wrong probabilities. *Confusion matrix* shows which roles get mixed up.

**Likely questions**
1. *Why is accuracy so high?* Synthetic data from the same skill lists; see limitations. Real-world accuracy would be lower, which is why I compare against a rule-based baseline (0.71) and report an independent set.
2. *Why ML instead of keyword rules?* Rules gave 71%; the model learns weights from many co-occurring skills and gives calibrated probabilities (+23 points on the same test data).
3. *Why Naive Bayes won?* Highest CV macro-F1 on short, keyword-driven text with small data; differences from Logistic Regression are about 1 point, within noise.
4. *What is data leakage and did you have it?* Yes, in the first generator version (100% accuracy); I diagnosed it by training on single sections and fixed it.
5. *How do you avoid training on the test set?* Split first; model selection by CV on training data only; test set used once.
6. *How would you improve it with real data?* Collect consented, anonymised resumes with human-verified labels, add embeddings (e.g. sentence transformers), fairness audits, OCR.
