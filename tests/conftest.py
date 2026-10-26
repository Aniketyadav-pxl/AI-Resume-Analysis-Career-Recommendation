import pytest

from src.model_store import load_bundle


@pytest.fixture(scope="session")
def bundle():
    b, _status = load_bundle()
    return b
