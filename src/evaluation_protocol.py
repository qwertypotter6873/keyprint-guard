"""Shared, deterministic sampling and paired uncertainty for verification trials."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def balanced_impostor_indices(index, target, word, per_person=2, seed=42):
    """Equal quota from every eligible other person, independent of dict ordering.

    index maps (person, session, occurrence) to tensor index. A local keyed RNG
    makes the same trials reusable across model/mode evaluators.
    """
    grouped = defaultdict(list)
    for key, idx in sorted(index.items()):
        if key[0] != target:
            grouped[key[0]].append(idx)
    if not grouped:
        return []
    quota = min(per_person, min(map(len, grouped.values())))
    selected = []
    for person, values in sorted(grouped.items()):
        digest = hashlib.sha256(f"{seed}|{word}|{target}|{person}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        selected.extend(sorted(rng.choice(values, size=quota, replace=False).tolist()))
    return selected


def paired_person_auc_ci(y, score_a, score_b, targets, queries, n_boot=1000, seed=0):
    """Paired participant bootstrap, conditional on observed words/sessions.

    Resample identities once per replicate and retain their full trial blocks.
    A genuine row receives its person's multiplicity; an impostor comparison
    receives the product of target/query multiplicities. Thus both roles of a
    repeatedly used identity are retained, rather than resampling trial rows.
    """
    y, score_a, score_b = map(np.asarray, (y, score_a, score_b))
    targets, queries = np.asarray(targets), np.asarray(queries)
    if not (len(y) == len(targets) == len(queries)):
        raise ValueError("Every score needs target and query identity")
    people = sorted(set(targets) | set(queries))
    lookup = {p: i for i, p in enumerate(people)}
    ti = np.array([lookup[p] for p in targets])
    qi = np.array([lookup[p] for p in queries])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        weights = mult[ti] * np.where(ti == qi, 1, mult[qi])
        if not weights[y == 0].sum() or not weights[y == 1].sum():
            continue
        diffs.append(roc_auc_score(y, score_a, sample_weight=weights) -
                     roc_auc_score(y, score_b, sample_weight=weights))
    if not diffs:
        raise ValueError("Too few identity clusters for a two-class bootstrap")
    return {
        "point_difference": round(float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)), 4),
        "mean": round(float(np.mean(diffs)), 4),
        "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
        "method": "paired_participant_bootstrap_target_query_weights",
        "num_people": len(people),
        "replicates": len(diffs),
        "conditional_on": "observed words, sessions and enrollment protocol",
    }
