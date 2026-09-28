"""
All prompt templates for LLM benchmarking.

Four task types:
1. Passage continuation
2. Zero-shot scene generation
3. Few-shot scene generation
4. Style transfer
"""

import json
from pathlib import Path



# ---------------------------------------------------------------------------
# System prompt (shared across all tasks)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a literary writing assistant. Follow the instructions precisely. "
    "Do not add meta-commentary, titles, or explanations. "
    "Output only the requested prose."
)


# ---------------------------------------------------------------------------
# Task 1: Passage Continuation
# ---------------------------------------------------------------------------

CONTINUATION_TEMPLATE = """\
Continue this passage in the same style. Do not use quotation marks. Write exactly 300 words.

{passage}"""


# ---------------------------------------------------------------------------
# Task 2: Zero-Shot Scene Generation
# ---------------------------------------------------------------------------

ZERO_SHOT_TEMPLATE = """\
Write a 300-word passage in the style of Cormac McCarthy's Blood Meridian. \
The scene: {description}. Do not use quotation marks."""


# ---------------------------------------------------------------------------
# Task 3: Few-Shot Scene Generation
# ---------------------------------------------------------------------------

FEW_SHOT_TEMPLATE = """\
Here are three passages from Cormac McCarthy's Blood Meridian:

---
{example_1}
---
{example_2}
---
{example_3}
---

Now write a new 300-word passage in this same style. \
The scene: {description}. Do not use quotation marks."""


# ---------------------------------------------------------------------------
# Task 4: Style Transfer
# ---------------------------------------------------------------------------

TRANSFER_TEMPLATE = """\
Rewrite the following passage in the style of Cormac McCarthy's Blood Meridian. \
Preserve the narrative content but transform the prose style. \
Do not use quotation marks. Write approximately 300 words.

Original:
{passage}"""


# ---------------------------------------------------------------------------
# Contamination probe prompt
# ---------------------------------------------------------------------------

CONTAMINATION_TEMPLATE = """\
Continue this text. Write the next 200 words exactly as they appear in the original source. \
Do not paraphrase or add new content. Reproduce the original text as faithfully as possible.

{passage}"""


# ---------------------------------------------------------------------------
# LLM-as-judge prompt (pairwise)
# ---------------------------------------------------------------------------

JUDGE_PAIRWISE_TEMPLATE = """\
You are judging which of two passages better captures the prose style of \
Cormac McCarthy's Blood Meridian.

Consider: sentence rhythm (mix of short and long), monosyllabic vocabulary, \
polysyndeton ("and...and...and"), absence of quotation marks, dark/biblical imagery, \
landscape descriptions, motion verbs.

Passage A:
{passage_a}

Passage B:
{passage_b}

Which passage better captures McCarthy's style? Answer with only "A" or "B"."""


# ---------------------------------------------------------------------------
# Scene descriptions (for zero-shot and few-shot tasks)
# ---------------------------------------------------------------------------

SCENE_DESCRIPTIONS = [
    "A group of riders crosses a salt flat at dawn. The light is pale and the horizon empty.",
    "Two men sit by a fire in the desert at night. One of them speaks about God.",
    "A lone figure walks through the ruins of an abandoned Mexican village after a massacre.",
    "Riders encounter a dust storm on the plains. They cannot see. The horses refuse to move.",
    "A man stands at the edge of a river at dusk, watching the water carry something downstream.",
]


# ---------------------------------------------------------------------------
# Style transfer source passages
# ---------------------------------------------------------------------------

# TRANSFER_SOURCES (Hemingway, Faulkner, DeLillo excerpts) live in data/benchmark/passages.json,
# which is gitignored: the passages are copyrighted. Loaded on first access; see _passages().


# ---------------------------------------------------------------------------
# Few-shot examples (Blood Meridian excerpts)
# ---------------------------------------------------------------------------

# FEW_SHOT_EXAMPLES (three Blood Meridian excerpts) live in data/benchmark/passages.json,
# which is gitignored for the same reason. Loaded on first access; see _passages().

_PASSAGES_FILE = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "passages.json"
_PASSAGES_CACHE = None


def _passages() -> dict:
    global _PASSAGES_CACHE
    if _PASSAGES_CACHE is None:
        if not _PASSAGES_FILE.exists():
            raise FileNotFoundError(
                f"{_PASSAGES_FILE} is missing. It holds copyrighted excerpts and is not in git; "
                "restore it locally to run the few-shot and style-transfer tasks."
            )
        _PASSAGES_CACHE = json.loads(_PASSAGES_FILE.read_text(encoding="utf-8"))
    return _PASSAGES_CACHE


def __getattr__(name):
    if name in ("FEW_SHOT_EXAMPLES", "TRANSFER_SOURCES"):
        return _passages()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_continuation_prompt(passage: str) -> str:
    return CONTINUATION_TEMPLATE.format(passage=passage)


def build_zero_shot_prompt(description: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(description=description)


def build_few_shot_prompt(description: str, examples: list[str] = None) -> str:
    exs = examples or _passages()["FEW_SHOT_EXAMPLES"]
    return FEW_SHOT_TEMPLATE.format(
        example_1=exs[0],
        example_2=exs[1],
        example_3=exs[2],
        description=description,
    )


def build_transfer_prompt(passage: str) -> str:
    return TRANSFER_TEMPLATE.format(passage=passage)


def build_contamination_prompt(passage: str) -> str:
    return CONTAMINATION_TEMPLATE.format(passage=passage)


def build_judge_prompt(passage_a: str, passage_b: str) -> str:
    return JUDGE_PAIRWISE_TEMPLATE.format(
        passage_a=passage_a,
        passage_b=passage_b,
    )
