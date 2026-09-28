"""
McCarthy-specific stylometric metrics.

Refactored from evaluate.py + corpus_analysis.py, extended with academic metrics.
All functions take plain text and return dicts or scalars.
"""

import re
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Syllable counting (heuristic, fast)
# ---------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    """Estimate syllable count via vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(1, count)


# ---------------------------------------------------------------------------
# Core McCarthy metrics (from evaluate.py)
# ---------------------------------------------------------------------------

def extract_words(text: str) -> list[str]:
    """Extract word tokens from text."""
    return re.findall(r"[a-zA-Z']+", text)


def extract_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, filter empties."""
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


def mccarthy_metrics(text: str) -> dict:
    """
    Compute McCarthy-specific metrics for a text sample.

    Returns dict with:
        total_words, monosyllable_pct, avg_syllables_per_word,
        and_frequency_pct, and_count, quotation_marks,
        avg_sentence_length, median_sentence_length,
        min_sentence_length, max_sentence_length, num_sentences,
        sentence_length_dist, syllable_dist,
        type_token_ratio, unique_words, top_words
    """
    words = extract_words(text)
    n = len(words)
    if n == 0:
        return {"error": "No words found"}

    result = {"total_words": n}

    # Syllable analysis
    syls = [count_syllables(w) for w in words]
    mono = sum(1 for s in syls if s == 1)
    result["monosyllable_pct"] = 100.0 * mono / n
    result["avg_syllables_per_word"] = sum(syls) / n

    syl_dist = Counter(syls)
    result["syllable_dist"] = {k: 100.0 * v / n for k, v in sorted(syl_dist.items())}

    # Polysyndeton
    and_count = sum(1 for w in words if w.lower() == "and")
    result["and_frequency_pct"] = 100.0 * and_count / n
    result["and_count"] = and_count

    # Punctuation
    result["quotation_marks"] = (
        text.count('"') + text.count('\u201c') + text.count('\u201d')
    )
    result["periods"] = text.count(".")
    result["commas"] = text.count(",")

    # Sentences
    sentences = extract_sentences(text)
    sent_lens = [len(extract_words(s)) for s in sentences]
    sent_lens = [l for l in sent_lens if l > 0]

    if sent_lens:
        result["num_sentences"] = len(sent_lens)
        result["avg_sentence_length"] = np.mean(sent_lens)
        result["median_sentence_length"] = np.median(sent_lens)
        result["min_sentence_length"] = min(sent_lens)
        result["max_sentence_length"] = max(sent_lens)
        result["sentence_lengths"] = sent_lens  # raw list for distribution tests

        # Bucketed distribution
        buckets = {"1-10": 0, "11-20": 0, "21-30": 0, "31-50": 0,
                   "51-75": 0, "76-100": 0, "101-200": 0, "201+": 0}
        for l in sent_lens:
            if l <= 10:      buckets["1-10"] += 1
            elif l <= 20:    buckets["11-20"] += 1
            elif l <= 30:    buckets["21-30"] += 1
            elif l <= 50:    buckets["31-50"] += 1
            elif l <= 75:    buckets["51-75"] += 1
            elif l <= 100:   buckets["76-100"] += 1
            elif l <= 200:   buckets["101-200"] += 1
            else:            buckets["201+"] += 1
        result["sentence_length_dist"] = {
            k: 100.0 * v / len(sent_lens) for k, v in buckets.items()
        }

    # Vocabulary
    lower_words = [w.lower() for w in words]
    unique = set(lower_words)
    result["unique_words"] = len(unique)
    result["type_token_ratio"] = len(unique) / n

    freq = Counter(lower_words)
    result["top_words"] = freq.most_common(10)

    # Hapax legomena
    hapax = sum(1 for w, c in freq.items() if c == 1)
    result["hapax_count"] = hapax
    result["hapax_ratio"] = hapax / len(unique) if unique else 0

    return result


# ---------------------------------------------------------------------------
# Comparison to baseline
# ---------------------------------------------------------------------------

def compare_to_baseline(metrics: dict, targets: dict = None) -> dict:
    """
    Compare computed metrics to McCarthy baseline targets.

    Returns dict of {metric: {target, actual, diff, pct_diff, status}}.
    """
    if targets is None:
        from lib.config import MCCARTHY_TARGETS
        targets = MCCARTHY_TARGETS

    comparisons = {}
    for key, target in targets.items():
        actual = metrics.get(key)
        if actual is None:
            continue
        diff = actual - target
        pct_diff = 100.0 * diff / target if target != 0 else float("inf")
        if abs(pct_diff) < 10:
            status = "CLOSE"
        elif abs(pct_diff) < 25:
            status = "OKAY"
        else:
            status = "OFF"
        comparisons[key] = {
            "target": target,
            "actual": actual,
            "diff": diff,
            "pct_diff": pct_diff,
            "status": status,
        }
    return comparisons


# ---------------------------------------------------------------------------
# Distinct-N (Li 2016)
# ---------------------------------------------------------------------------

def distinct_n(text: str, n: int = 1) -> float:
    """
    Ratio of unique n-grams to total n-grams.
    Higher = more lexically diverse.
    """
    words = [w.lower() for w in extract_words(text)]
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


# ---------------------------------------------------------------------------
# Sentence length distribution vector
# ---------------------------------------------------------------------------

SENTENCE_BINS = ["1-10", "11-20", "21-30", "31-50", "51-75", "76-100", "101-200", "201+"]

def sentence_length_distribution(text: str) -> np.ndarray:
    """Return normalized sentence length distribution as array matching SENTENCE_BINS."""
    m = mccarthy_metrics(text)
    dist = m.get("sentence_length_dist", {})
    return np.array([dist.get(b, 0.0) for b in SENTENCE_BINS]) / 100.0


# ---------------------------------------------------------------------------
# POS tag distributions (requires spaCy)
# ---------------------------------------------------------------------------

def pos_distribution(text: str, nlp=None) -> dict:
    """
    Compute POS unigram and bigram distributions.

    Args:
        text: Input text
        nlp: spaCy Language object (loaded externally to avoid repeated loading)

    Returns:
        {"unigram": Counter, "bigram": Counter}
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    tags = [tok.pos_ for tok in doc if not tok.is_space]

    unigrams = Counter(tags)
    bigrams = Counter(zip(tags[:-1], tags[1:]))

    # Normalize
    uni_total = sum(unigrams.values())
    bi_total = sum(bigrams.values())

    uni_norm = {k: v / uni_total for k, v in unigrams.items()} if uni_total else {}
    bi_norm = {k: v / bi_total for k, v in bigrams.items()} if bi_total else {}

    return {"unigram": uni_norm, "bigram": bi_norm}


# ---------------------------------------------------------------------------
# Fingerprint I/O
# ---------------------------------------------------------------------------

def save_fingerprint(metrics: dict, path: str | Path):
    """Save fingerprint to JSON (numpy-safe serialization)."""
    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=default))


def load_fingerprint(path: str | Path) -> dict:
    """Load fingerprint from JSON."""
    return json.loads(Path(path).read_text())
