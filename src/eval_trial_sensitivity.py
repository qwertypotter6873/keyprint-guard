"""Frozen-checkpoint sensitivity to verification-trial sampling on CMU Test.

Writes only models/trial_sensitivity.json. No training, threshold calibration,
model choice, or modification of the official evaluation report occurs here.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from models import KeyPrintNet, Vanilla1DCNN
from preprocessing import SEED, prepare_full_pipeline
from train_and_evaluate import (
    TEST_BOT_SEED, build_verification_trials, compute_cohort_centroids,
    encode_sequences, score_embedding_trials, summarize_metrics,
)

TRIAL_SEEDS = tuple(range(42, 62))
METRICS = ("macro_f1", "roc_auc", "eer", "far", "frr", "balanced_accuracy")


def exhaustive_human_trials(split, base_trials):
    """All non-enrollment-session genuine rows and all other-person rows per template.

    Enrollment stays K=5 at sessions 1,3,5,7. Enrollment samples can be queries
    against OTHER people's templates; no own-session genuine query is included.
    """
    groups = []
    for original in base_trials["groups"]:
        group = dict(original)
        own = split.subject_ids == original["uid"]
        group["gen_idx"] = np.where(own & (split.sessions != original["enroll_session"]))[0]
        group["imp_idx"] = np.where(~own)[0]
        group["bot_idx"] = np.array([], dtype=np.int64)
        groups.append(group)
    return {**base_trials, "groups": groups}


def trial_counts(trials):
    return {
        "enrollment_templates": len(trials["groups"]),
        "genuine_comparisons": sum(len(g["gen_idx"]) for g in trials["groups"]),
        "human_impostor_comparisons": sum(len(g["imp_idx"]) for g in trials["groups"]),
        "bot_comparisons": sum(len(g["bot_idx"]) for g in trials["groups"]),
    }


def summarize_draws(draws, key):
    """Descriptive spread across sampled trial sets, NOT a population confidence interval."""
    result = {}
    for metric in METRICS:
        values = np.array([row[key][metric] for row in draws])
        result[metric] = {
            "mean": round(float(values.mean()), 6),
            "median": round(float(np.median(values)), 6),
            "std_across_draws": round(float(values.std(ddof=1)), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
            "seed_at_min": draws[int(values.argmin())]["trial_seed"],
            "seed_at_max": draws[int(values.argmax())]["trial_seed"],
        }
        if metric == "macro_f1" and key in ("proposed", "control"):
            result[metric]["goal"] = 0.85
            result[metric]["seeds_meeting_goal"] = [
                row["trial_seed"] for row in draws if row[key][metric] >= 0.85
            ]
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    out_root = Path(os.environ.get("KEYPRINT_OUTPUT_ROOT", root)).resolve()
    official_path = out_root / "models/evaluation_report.json"
    official = json.loads(official_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        splits, meta = prepare_full_pipeline(root / "data/raw/DSL-StrongPasswordData.csv", Path(tmp), seed=SEED)
    mean = np.asarray(meta["channel_mean"], np.float32)
    std = np.asarray(meta["channel_std"], np.float32)
    base = build_verification_trials(splits["test"], mean, std, k_shot=5, seed=SEED, bot_seed=TEST_BOT_SEED)
    device = torch.device("cpu")
    models = {}
    weights = {}
    for name, klass, weight_path in (
        ("proposed", KeyPrintNet, "models/keyprint_net.pt"),
        ("control", Vanilla1DCNN, "models/vanilla_1dcnn_aug.pt"),
    ):
        model = klass(in_channels=17)
        path = out_root / weight_path
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        cohort = compute_cohort_centroids(encode_sequences(model, splits["train"].seq_scaled, device), splits["train"].subject_ids)
        models[name] = (encode_sequences(model, splits["test"].seq_scaled, device),
                        encode_sequences(model, base["bot_seq"], device), cohort)
        weights[weight_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    thresholds = {"proposed": official["models"]["KeyPrintNet"]["threshold"],
                  "control": official["models"]["Vanilla1DCNN_Aug_ASNorm"]["threshold"]}

    def evaluate(trials):
        result, labels = {}, None
        for name, (embedding, bots, cohort) in models.items():
            y, score, _ = score_embedding_trials(embedding, bots, trials, cohort=cohort)
            if labels is not None:
                np.testing.assert_array_equal(labels, y)
            labels = y
            values = summarize_metrics(y, score, thresholds[name])
            values["balanced_accuracy"] = round(0.5 * (
                values["tp"] / (values["tp"] + values["fn"]) +
                values["tn"] / (values["tn"] + values["fp"])), 6)
            result[name] = values
        result["proposed_minus_control"] = {k: round(result["proposed"][k] - result["control"][k], 6) for k in METRICS}
        return result

    full = exhaustive_human_trials(splits["test"], base)
    counts = trial_counts(full)
    assert counts["genuine_comparisons"] == 14000
    assert counts["human_impostor_comparisons"] == 144000
    exhaustive = {"counts": counts, **evaluate(full)}
    print("Exhaustive human comparison:", json.dumps(exhaustive), flush=True)

    draws = []
    for seed in TRIAL_SEEDS:
        trials = build_verification_trials(splits["test"], mean, std, k_shot=5, seed=seed, bot_seed=TEST_BOT_SEED)
        row = {"trial_seed": seed, "counts": trial_counts(trials), **evaluate(trials)}
        draws.append(row)
        print("Trial seed", seed, row["proposed_minus_control"], flush=True)
    official_check = {}
    for label, report_name in (("proposed", "KeyPrintNet"), ("control", "Vanilla1DCNN_Aug_ASNorm")):
        official_check[label] = {key: {"official": official["models"][report_name][key], "recomputed": draws[0][label][key]}
                                 for key in METRICS if key != "balanced_accuracy" and
                                 official["models"][report_name][key] != draws[0][label][key]}
    report = {
        "protocol": {
            "split": "CMU subject-disjoint Test; same 10 previously observed benchmark people",
            "subjects": splits["test"].subjects,
            "training_split_seed": SEED,
            "enrollment_sessions": [1, 3, 5, 7], "k_enroll": 5,
            "trial_seeds": list(TRIAL_SEEDS), "bot_pool_seed": TEST_BOT_SEED,
            "thresholds": thresholds,
            "threshold_source": "saved Validation operating points (4-decimal deployed precision); no re-calibration",
            "varied": "Only which genuine/human-impostor/bot queries are drawn per fixed template; weights, people, enrollment, cohort, bot pool and thresholds fixed.",
            "paired": "Both models receive the identical enrollment and query indices for every comparison.",
            "exhaustive_counts_note": "40 templates x 350 cross-session genuine = 14,000; 40 x 3,600 other-person rows = 144,000; no bots. Repeated comparisons are not independent raw inputs.",
            "sampled_counts_note": "40 templates x (45 genuine + 40 human impostor + 5 fixed-pool bot) = 3,600 comparisons per seed.",
            "interpretation": "Sampling spread is conditional on these Test people, not a fresh population confidence interval. Full human and sampled bot-mixed evaluations have different class ratios; their Macro F1 values are separate aggregates. EER/AUC describe Test curves, not deployed threshold selection.",
            "heldout_test_tuning": False,
        },
        "exhaustive_human": exhaustive,
        "sampled_20_draws": {"draws": draws, "spread": {k: summarize_draws(draws, k) for k in ("proposed", "control", "proposed_minus_control")}},
        "official_seed_42_metric_differences": official_check,
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": {f"src/{name}": hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
                              for name in ("preprocessing.py", "train_and_evaluate.py", "models.py")},
            "checkpoint_sha256": weights,
            "official_report_sha256": hashlib.sha256(official_path.read_bytes()).hexdigest(),
            "torch_version": torch.__version__, "numpy_version": np.__version__,
        },
    }
    output = out_root / "models/trial_sensitivity.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("Wrote", output, flush=True)


if __name__ == "__main__":
    main()
