"""
Project configuration: paths, model registry, constants, API key loading.
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
CORPUS_PATH = CORPUS_DIR / "blood_meridian_clean.txt"
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPERS_DIR = PROJECT_ROOT / "papers"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

FINGERPRINT_PATH = REFERENCE_DIR / "mccarthy_fingerprint.json"

# Ensure output dirs exist
for d in [REFERENCE_DIR, RESULTS_DIR / "contamination",
          RESULTS_DIR / "benchmark", RESULTS_DIR / "internals"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_CI = 0.95
BONFERRONI_K = 8  # number of models
ALPHA = 0.05 / BONFERRONI_K

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 600  # ~300 words
DEFAULT_TEMPERATURE = 0.8
MIN_WORDS_FOR_DELTA = 5000  # Eder 2015
CONTINUATION_PREFIX_WORDS = 150
GENERATION_TARGET_WORDS = 300

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    # Provider API models
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "label": "GPT-4o",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "label": "GPT-4o-mini",
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "label": "Claude Sonnet 4.5",
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5",
    },
    "gemini-pro": {
        "provider": "google",
        "model_id": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
    },
    "gemini-flash": {
        "provider": "google",
        "model_id": "gemini-2.0-flash",
        "label": "Gemini 2.0 Flash",
    },
    # Local models (Ollama)
    "llama-3.2-3b": {
        "provider": "ollama",
        "model_id": "llama3.2:3b",
        "label": "Llama 3.2 3B",
    },
    "mistral-7b": {
        "provider": "ollama",
        "model_id": "mistral:7b",
        "label": "Mistral 7B",
    },
}

MODEL_ORDER = [
    "gpt-4o", "gpt-4o-mini",
    "claude-sonnet", "claude-haiku",
    "gemini-pro", "gemini-flash",
    "llama-3.2-3b", "mistral-7b",
]

# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def get_api_key(provider: str) -> str:
    """Load API key from environment. Raises if missing."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    var = env_map.get(provider)
    if var is None:
        return ""  # ollama doesn't need a key
    key = os.environ.get(var, "")
    if not key:
        raise EnvironmentError(
            f"Missing {var}. Set it in your shell or .env file."
        )
    return key


# ---------------------------------------------------------------------------
# McCarthy baseline targets (from FINDINGS.md)
# ---------------------------------------------------------------------------

MCCARTHY_TARGETS = {
    "monosyllable_pct": 81.6,
    "avg_syllables_per_word": 1.22,
    "and_frequency_pct": 5.82,
    "quotation_marks": 0,
    "avg_sentence_length": 15.7,
    "median_sentence_length": 11,
    "hapax_ratio": 0.525,
    "type_token_ratio": 0.0879,
}

# Sentence length distribution from FINDINGS.md (for JSD comparison)
MCCARTHY_SENTENCE_DIST = {
    "1-10": 48.1,
    "11-20": 25.4,
    "21-30": 13.8,
    "31-50": 9.8,
    "51-75": 2.2,
    "76-100": 0.4,
    "101-200": 0.3,
    "201+": 0.0,
}

# Syllable distribution from FINDINGS.md
MCCARTHY_SYLLABLE_DIST = {
    1: 81.6,
    2: 15.0,
    3: 2.8,
    4: 0.4,
    5: 0.1,
}
