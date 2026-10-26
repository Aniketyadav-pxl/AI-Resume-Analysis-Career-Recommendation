"""M2 - Text normalisation and spaCy preprocessing for the ML model.

Skills are first replaced by protected tokens (e.g. 'sk_scikit_learn') so lemmatisation can
never damage them (a naive lemmatiser turns 'pandas' into 'panda'). Remaining words are
lower-cased, lemmatised and stop-word filtered with spaCy.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from functools import lru_cache

import config
from src.skill_extractor import find_skill_spans, slug

PREP_VERSION = "v1"  # bump when preprocessing changes -> forces model retraining
_EMAIL = re.compile(r"\S+@\S+")
_URL = re.compile(r"(https?://\S+|www\.\S+|(linkedin|github)\.com/\S+)", re.I)
_ID = re.compile(r"\b(candidate\s+)?r\d{4}\b", re.I)
_BULLETS = re.compile(r"[\u2022\u25CF\u25AA\u25E6\u2023\u2043\u25A0\u2013\u2014]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "")
    text = _BULLETS.sub("-", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy once. Falls back to a blank English pipeline (no lemmas) if the model is missing."""
    import spacy

    try:
        return spacy.load(config.SPACY_MODEL, disable=["parser", "ner"])
    except OSError:
        warnings.warn(
            f"spaCy model '{config.SPACY_MODEL}' not found; using a basic tokenizer without lemmatisation. "
            f"Install it with: python -m spacy download {config.SPACY_MODEL}", stacklevel=2)
        return spacy.blank("en")


def protect_skills(text: str) -> str:
    """Replace every detected skill mention with its protected token."""
    out, last = [], 0
    for start, end, canon in find_skill_spans(text):
        out.append(text[last:start])
        out.append(f" {slug(canon)} ")
        last = end
    out.append(text[last:])
    return "".join(out)


def _pre(text: str) -> str:
    text = normalize_text(text)
    text = protect_skills(text)  # before URL/email removal changes offsets
    return _ID.sub(" ", _URL.sub(" ", _EMAIL.sub(" ", text)))


def _tokens(doc) -> list:
    out = []
    for tok in doc:
        t = tok.text.lower()
        if t.startswith("sk_"):
            out.append(t)
        elif tok.is_alpha and not tok.is_stop and len(t) > 1:
            lemma = (tok.lemma_ or t).lower()
            out.append(lemma if lemma.isalpha() else t)
    return out


def prepare_model_text(text: str) -> str:
    return " ".join(_tokens(get_nlp()(_pre(text))))


def prepare_many(texts, batch_size: int = 64) -> list:
    nlp = get_nlp()
    return [" ".join(_tokens(d)) for d in nlp.pipe((_pre(t) for t in texts), batch_size=batch_size)]
