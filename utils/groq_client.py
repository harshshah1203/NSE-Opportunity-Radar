"""
Shared Groq client + environment loading helpers.
"""

from pathlib import Path
import os

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _clean_secret(value: str) -> str:
    return value.strip().strip('"').strip("'")


def get_groq_api_key() -> str:
    """
    Load and validate GROQ_API_KEY from project .env.

    Uses override=True so stale shell/system values do not shadow .env.
    """
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    api_key = _clean_secret(os.getenv("GROQ_API_KEY", ""))

    if not api_key:
        raise ValueError(f"GROQ_API_KEY missing. Add it to {ENV_PATH}.")

    # Fast sanity checks to catch common placeholder mistakes early.
    key_lower = api_key.lower()
    if key_lower.endswith("your_key_here") or key_lower.endswith("api_key_here") or key_lower.endswith("key_here"):
        raise ValueError("GROQ_API_KEY looks like placeholder text. Paste your real Groq key.")

    if not api_key.startswith("gsk_"):
        raise ValueError("GROQ_API_KEY format looks invalid (expected prefix 'gsk_').")

    return api_key


def get_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())
