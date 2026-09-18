"""Enrichment of a feature (fragment or fragment combination) in binders.

Two estimates are produced.

The *pooled* enrichment factor is the fraction of binders carrying the feature
over the fraction of non-binders carrying it, so EF = 4 means the feature turns
up four times as often among binders.  It is easy to read but it silently
compares compounds across sub-libraries, and sub-libraries differ enormously in
how many binders they contain - so a feature can score highly just for living in
a productive sub-library.

The *Mantel-Haenszel* enrichment fixes that by comparing compounds only against
others from the same sub-library and pooling those per-stratum comparisons.  It
estimates an odds ratio rather than a risk ratio, but when binders are a small
fraction of the library - the usual case, and more so for large ones - the two
coincide, so it reads on the same scale as the pooled factor.

Counts are per compound: a feature occurring twice in one molecule counts once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, hypergeom

DEFAULT_ALPHA = 1.0

# Sums that a Mantel-Haenszel estimate needs, accumulated over strata.
MH_TERMS = ("mh_a", "mh_r", "mh_s", "mh_pr", "mh_psqr", "mh_qs", "mh_e", "mh_v")


def stratum_terms(
    n_binder: np.ndarray,
    n_nonbinder: np.ndarray,
    total_binder: np.ndarray,
    total_nonbinder: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-stratum Mantel-Haenszel contributions, ready to be summed by feature.

    Each row is one feature in one stratum: ``a`` binders and ``c`` non-binders
    carry it, out of ``total_binder``/``total_nonbinder`` compounds in that
    stratum.  Strata where the feature is absent contribute zero to every term,
    so they can simply be left out of the table.
    """
    a = np.asarray(n_binder, dtype=np.float64)
    c = np.asarray(n_nonbinder, dtype=np.float64)
    b = np.asarray(total_binder, dtype=np.float64) - a
    d = np.asarray(total_nonbinder, dtype=np.float64) - c
    n = a + b + c + d

    with np.errstate(divide="ignore", invalid="ignore"):
        r = a * d / n
        s = b * c / n
        p = (a + d) / n
        q = (b + c) / n
        e = (a + b) * (a + c) / n
        v = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1.0))

    out = {
        "mh_a": a,
        "mh_r": r,
        "mh_s": s,
        "mh_pr": p * r,
        "mh_psqr": p * s + q * r,
        "mh_qs": q * s,
        "mh_e": e,
        "mh_v": v,
    }
    return {key: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in out.items()}


def add_mantel_haenszel(summed: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Turn summed per-stratum terms into an adjusted enrichment and its CI.

    The variance follows Robins-Breslow-Greenland.  When no non-binder anywhere
    carries the feature the odds ratio is infinite, so ``alpha`` stands in for
    the empty denominator to keep the confidence bound - the value used for
    ranking - finite.
    """
    out = summed.copy()
    sum_r = out["mh_r"].to_numpy(dtype=np.float64)
    sum_s = out["mh_s"].to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["enrichment_mh"] = sum_r / sum_s

    guard_s = np.where(sum_s > 0, sum_s, alpha)
    guard_r = np.where(sum_r > 0, sum_r, alpha)
    ratio = guard_r / guard_s
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (
            out["mh_pr"].to_numpy() / (2.0 * guard_r**2)
            + out["mh_psqr"].to_numpy() / (2.0 * guard_r * guard_s)
            + out["mh_qs"].to_numpy() / (2.0 * guard_s**2)
        )
    se = np.sqrt(np.clip(np.nan_to_num(variance, nan=np.inf), 0.0, None))
    log_ratio = np.log(ratio)
    out["enrichment_mh_lo95"] = np.exp(log_ratio - 1.96 * se)
    out["enrichment_mh_hi95"] = np.exp(log_ratio + 1.96 * se)

    # Mantel-Haenszel test, continuity-corrected.
    observed = out["mh_a"].to_numpy(dtype=np.float64)
    expected = out["mh_e"].to_numpy(dtype=np.float64)
    var = out["mh_v"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.clip(np.abs(observed - expected) - 0.5, 0.0, None) ** 2 / var
    statistic = np.nan_to_num(statistic, nan=0.0, posinf=0.0)
    # One-sided: only over-representation among binders is of interest.
    p_two_sided = chi2.sf(statistic, df=1)
    out["p_value_mh"] = np.where(observed >= expected, p_two_sided / 2.0, 1.0 - p_two_sided / 2.0)
    return out


def add_enrichment(
    counts: pd.DataFrame,
    n_binders: int,
    n_nonbinders: int,
    alpha: float = DEFAULT_ALPHA,
    binder_col: str = "n_binder",
    nonbinder_col: str = "n_nonbinder",
    tested: np.ndarray | None = None,
    with_test: bool = True,
) -> pd.DataFrame:
    """Append the pooled enrichment factor to a table of per-feature counts.

    ``alpha`` is a pseudocount that keeps the factor finite for features absent
    from one class; the unsmoothed ratio is kept alongside it.  ``tested``
    restricts the multiple-testing correction to the features worth testing, so
    that a long tail of features seen in a handful of compounds does not inflate
    every q-value.
    """
    out = counts.copy()
    n_bind = out[binder_col].to_numpy(dtype=np.float64)
    n_non = out[nonbinder_col].to_numpy(dtype=np.float64)
    total_b = float(n_binders)
    total_n = float(n_nonbinders)

    out["n_total"] = out[binder_col] + out[nonbinder_col]
    out["frac_binder"] = n_bind / total_b
    out["frac_nonbinder"] = n_non / total_n

    with np.errstate(divide="ignore", invalid="ignore"):
        out["enrichment_factor"] = out["frac_binder"] / out["frac_nonbinder"]
    out["enrichment_factor_smoothed"] = ((n_bind + alpha) / (total_b + alpha)) / (
        (n_non + alpha) / (total_n + alpha)
    )
    out["log2_enrichment"] = np.log2(out["enrichment_factor_smoothed"])

    # 95% lower bound on the enrichment factor, from the standard error of the
    # log odds ratio - a conservative ranking key that discounts rare features.
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(
            1.0 / np.maximum(n_bind, 0.5)
            + 1.0 / np.maximum(n_non, 0.5)
            - 1.0 / total_b
            - 1.0 / total_n
        )
    se = np.nan_to_num(se, nan=np.inf, posinf=np.inf)
    log_ef = np.log(out["enrichment_factor_smoothed"].to_numpy())
    out["enrichment_factor_lo95"] = np.exp(log_ef - 1.96 * se)

    if with_test:
        # One-sided Fisher exact test (hypergeometric upper tail) for over-
        # representation among binders; vectorised, unlike scipy's fisher_exact.
        total = total_b + total_n
        drawn = n_bind + n_non
        out["p_value"] = hypergeom.sf(n_bind - 1, total, total_b, drawn)
        if tested is None:
            tested = np.ones(len(out), dtype=bool)
        out["tested"] = tested
        q = np.full(len(out), np.nan)
        q[tested] = benjamini_hochberg(out["p_value"].to_numpy()[tested])
        out["q_value"] = q
    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    n = p_values.size
    if n == 0:
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out
