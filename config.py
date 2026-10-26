"""Central configuration: paths, constants, and the module build-status registry."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_resumes"

SYNTHETIC_RESUMES_CSV = DATA_DIR / "synthetic_resumes.csv"
JOBS_CSV = DATA_DIR / "jobs.csv"
DB_PATH = DATA_DIR / "app.db"
MODEL_PATH = MODELS_DIR / "career_classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
SPACY_MODEL = "en_core_web_sm"
ALLOWED_UPLOAD_TYPES = ["pdf", "docx", "txt"]

# Build-status registry shown on the dashboard. Update status as modules are completed.
MODULES = [
    ("M0", "Project scaffold, knowledge base, synthetic dataset", "done"),
    ("M1", "Resume text extraction (PDF / DOCX / TXT)", "done"),
    ("M2", "NLP preprocessing and skill extraction", "done"),
    ("M3", "Resume analysis and quality score", "done"),
    ("M4", "ML training pipeline (TF-IDF + classifiers)", "done"),
    ("M5", "Career recommendation engine", "done"),
    ("M6", "Job matching (cosine similarity)", "done"),
    ("M7", "Skill gap analysis", "done"),
    ("M8", "SQLite storage and history", "done"),
    ("M9", "Final dashboard and polish", "done"),
]
