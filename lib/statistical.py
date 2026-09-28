"""
Statistical methods: bootstrap CIs, JSD, significance tests.
"""

import numpy as np
from scipy import stats
from collections import Counter

from lib.config import BOOTSTRAP_ITERATIONS, BOOTSTRAP_CI, ALPHA, SEED


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci: float = BOOTSTRAP_CI,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    point = statistic(data)

    boot_stats = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic(sample)

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return float(point), float(lower), float(upper)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> float:
    """
    Paired bootstrap test: is the mean of scores_a significantly different from scores_b?

    Returns p-value (two-sided).
    """
    rng = np.random.default_rng(seed)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    assert len(scores_a) == len(scores_b), "Arrays must be same length for paired test"

    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    diffs = scores_a - scores_b

    count = 0
    for _ in range(n_iterations):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        if abs(np.mean(sample)) >= abs(observed_diff):
            count += 1

    return count / n_iterations


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def jsd(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    Jensen-Shannon Divergence between two probability distributions.
    Bounded [0, 1] when base=2.

    Both arrays must be valid probability distributions (non-negative, sum to ~1).
    Zeros are handled via the convention 0 * log(0) = 0.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize in case of floating point drift
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    # KL divergence with zero handling
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]) / np.log(base))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def jsd_from_counters(counter_a: Counter, counter_b: Counter) -> float:
    """
    JSD between two Counter distributions.
    Aligns keys and converts to probability vectors.
    """
    all_keys = sorted(set(counter_a.keys()) | set(counter_b.keys()))
    p = np.array([counter_a.get(k, 0) for k in all_keys], dtype=np.float64)
    q = np.array([counter_b.get(k, 0) for k in all_keys], dtype=np.float64)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 1.0  # maximum divergence

    p = p / p_sum
    q = q / q_sum
    return jsd(p, q)


# ---------------------------------------------------------------------------
# Distribution alignment helpers
# ---------------------------------------------------------------------------

def align_distributions(
    dist_a: dict[str, float],
    dist_b: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two {label: proportion} dicts to matching numpy arrays.
    Missing keys get 0.
    """
    all_keys = sorted(set(dist_a.keys()) | set(dist_b.keys()))
    a = np.array([dist_a.get(k, 0.0) for k in all_keys])
    b = np.array([dist_b.get(k, 0.0) for k in all_keys])
    return a, b


# ---------------------------------------------------------------------------
# Hartigan's dip test for bimodality
# ---------------------------------------------------------------------------

def dip_test(data: np.ndarray) -> tuple[float, float]:
    """
    Hartigan's dip test for unimodality.

    Returns (dip_statistic, p_value).
    Uses diptest package if available, otherwise falls back to a
    simple bimodality coefficient.
    """
    data = np.sort(np.asarray(data, dtype=np.float64))
    try:
        import diptest
        dip, pval = diptest.diptest(data)
        return float(dip), float(pval)
    except ImportError:
        # Fallback: bimodality coefficient (BC)
        # BC > 5/9 ~ 0.555 suggests bimodality
        n = len(data)
        if n < 3:
            return 0.0, 1.0
        skew = float(stats.skew(data))
        kurt = float(stats.kurtosis(data, fisher=True))  # excess kurtosis
        bc = (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
        # Approximate: BC > 0.555 is "bimodal"
        pval = 0.01 if bc > 0.555 else 0.5
        return float(bc), float(pval)


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)
