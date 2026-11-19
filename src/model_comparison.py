"""Paired uncertainty for the proposed encoder versus the augmentation-matched control."""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def comparison_with_control(trials, y, proposed_scores, control_scores, proposed_threshold,
                            control_threshold, seed, n_resamples=1000):
    """Resample enrolled identities, keeping each person's sessions and paired queries together.

    Intervals condition on the observed human queries and the fixed, independent Test bot bank.
    The two decision thresholds stay at their respective Validation operating points.
    """
    targets = np.concatenate([
        np.repeat(g["subject"], len(g["gen_idx"]) + len(g["imp_idx"]) + len(g["bot_idx"]))
        for g in trials["groups"]
    ])
    y = np.asarray(y)
    if not (len(targets) == len(y) == len(proposed_scores) == len(control_scores)):
        raise ValueError("Scores and enrollment-group trial counts must agree")
    subjects, groups = np.unique(targets, return_inverse=True)
    n_subjects = len(subjects)
    models = [(np.asarray(proposed_scores), proposed_threshold),
              (np.asarray(control_scores), control_threshold)]

    def difference(weights):
        result = []
        for scores, threshold in models:
            fpr, tpr, _ = roc_curve(y, scores, sample_weight=weights)
            nearest = np.argmin(np.abs(fpr - (1 - tpr)))
            eer = (fpr[nearest] + 1 - tpr[nearest]) / 2
            result.append(np.array([
                roc_auc_score(y, scores, sample_weight=weights), eer,
                f1_score(y, scores >= threshold, average="macro", sample_weight=weights),
            ]))
        return result[0] - result[1]

    point = difference(None)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_subjects, np.full(n_subjects, 1 / n_subjects), size=n_resamples)
    sampled = np.stack([difference(row[groups]) for row in counts])
    lo, hi = np.quantile(sampled, [0.025, 0.975], axis=0)
    return {
        "contrast": "KeyPrintNet minus Vanilla1DCNN_Aug_ASNorm",
        "resampling_unit": "enrolled subject (queries and bot bank held fixed)",
        "num_subjects": n_subjects, "n_resamples": n_resamples, "seed": seed,
        "metrics": {
            k: {"difference": round(float(d), 5), "ci95": [round(float(a), 5), round(float(b), 5)]}
            for k, d, a, b in zip(("roc_auc", "eer", "macro_f1"), point, lo, hi)
        },
    }
