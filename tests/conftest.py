"""Test setup: fake API clients, a throwaway database, and no retry delays.

This runs before the app is imported, so the real .env keys and the real
team2_evidence.db are never used.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "team2_app.py"
sys.path.insert(0, str(Path(__file__).parent))

_TMP = tempfile.mkdtemp(prefix="team2_tests_")
os.environ["TAVILY_API_KEY"] = "test"
os.environ["DEEPSEEK_API_KEY"] = "test"
os.environ["TEAM2_EVIDENCE_DB"] = os.path.join(_TMP, "test_evidence.db")
os.environ["TEAM2_RETRY_BASE_SECONDS"] = "0"

import openai  # noqa: E402
import tavily  # noqa: E402

import fakes  # noqa: E402

openai.OpenAI = fakes.FakeOpenAI
tavily.TavilyClient = fakes.FakeTavily


@pytest.fixture(autouse=True)
def fake_mode():
    fakes.reset()
    yield fakes.MODE


@pytest.fixture(scope="session")
def app():
    """The app imported as a module (Streamlit UI code runs harmlessly in bare mode)."""
    spec = importlib.util.spec_from_file_location("team2_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
