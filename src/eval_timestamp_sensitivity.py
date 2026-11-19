"""Audit the Train-derived fixed-text interpretation and separate inference and training effects."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
import torch

from feature_eval import eval_password_transfer, evaluate_clean_sentence_cross_session
from train_and_evaluate import (
    build_verification_trials,
    compute_cohort_centroids,
    encode_sequences,
    find_optimal_val_threshold,
    score_embedding_trials,
    summarize_metrics,
)
from models import KeyPrintNet
from preprocessing import N_CHANNELS, SEED, prepare_full_pipeline

NOMINAL_WEIGHT = "models/legacy/keyprint_net_17ch_published_dd.pt"
NOMINAL_MANIFEST = "models/legacy/keyprint_net_17ch_published_dd.json"


def column_identity_audit(path):
    """Check published DD/UD/DU/UU against adjacent measured holds, without CSV quote guessing."""
    rows = []
    with path.open() as stream:
        next(stream)
        for line in stream:
            ps1, ps2, tail = line.rstrip("\r\n").split(",", 2)
            fields = tail.rstrip(",").rsplit(",", 5)
            # Session-end rows have blank timing cells; do not join across them.
            if len(fields) != 6:
                rows.append(None)
                continue
            try:
                h, dd, du, ud, uu = map(float, fields[1:])
            except ValueError:
                rows.append(None)
                continue
            rows.append(((ps1, ps2), h, dd, du, ud, uu))
    errors = {"UD_equals_DD_minus_Hcurr": [], "DU_equals_DD_plus_Hnext": [], "UU_equals_DD_plus_Hnext_minus_Hcurr": []}
    for a, b in zip(rows, rows[1:]):
        if a is None or b is None or a[0] != b[0]:
            continue
        _, h, dd, du, ud, uu = a
        hn = b[1]
        errors["UD_equals_DD_minus_Hcurr"].append(abs(ud - (dd-h)))
        errors["DU_equals_DD_plus_Hnext"].append(abs(du - (dd+hn)))
        errors["UU_equals_DD_plus_Hnext_minus_Hcurr"].append(abs(uu - (dd+hn-h)))
    return {k: {"n": len(v), "median_absolute_error_sec": float(np.median(v)),
                "max_absolute_error_sec": float(max(v)),
                "fraction_within_2ms": float(np.mean(np.asarray(v) <= .0020001))}
            for k, v in errors.items()}


def checkpoint_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_and_cohort(path: Path, channels: int, train, device):
    model = KeyPrintNet(in_channels=channels)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    cohort = compute_cohort_centroids(encode_sequences(model, train.seq_scaled[:, :channels], device), train.subject_ids)
    return model, cohort


def compare_training_interpretations(out, official, splits, sentence_splits, correct_model, correct_cohort, mean, std, device):
    """Compare 17ch training runs using identical CMU and clean sentence Validation inputs."""
    nominal_path, manifest_path = out / NOMINAL_WEIGHT, out / NOMINAL_MANIFEST
    recipe = official["provenance"]["training_recipe"]
    if not nominal_path.is_file() and not manifest_path.is_file():
        if recipe["keyprint_epochs"] == 35:
            raise FileNotFoundError("Run src/train_timestamp_control.py to create the published-DD reference and manifest")
        return {"reference_available": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nominal_recipe = {**recipe, "keyrecs_fixed_interpretation": "published_labels"}
    if (manifest["training_recipe"] != nominal_recipe or manifest["sha256"] != checkpoint_sha(nominal_path)
            or manifest["train_subjects"] != sentence_splits["train"].subjects):
        raise ValueError("The published-DD reference must match the 17ch training recipe and checkpoint hash")
    nominal_model, nominal_cohort = load_model_and_cohort(nominal_path, N_CHANNELS, splits["train"], device)
    val = splits["val"]
    trials = build_verification_trials(val, mean, std, seed=SEED)
    metrics = {}
    for interpretation, model, cmu_cohort in (
        ("published_labels", nominal_model, nominal_cohort),
        (recipe["keyrecs_fixed_interpretation"], correct_model, correct_cohort),
    ):
        encs = encode_sequences(model, val.seq_scaled, device)
        bots = encode_sequences(model, trials["bot_seq"], device)
        y, s, _ = score_embedding_trials(encs, bots, trials, cohort=cmu_cohort)
        cmu = summarize_metrics(y, s, find_optimal_val_threshold(y, s))
        train = sentence_splits["train"]
        sent_cohort = compute_cohort_centroids(encode_sequences(model, train.seq_scaled, device), train.subject_ids)
        metrics[interpretation] = {
            "cmu_canonical_password": {k: cmu[k] for k in ("roc_auc", "eer", "macro_f1", "threshold")},
            "clean_25_char_cross_session": evaluate_clean_sentence_cross_session(sentence_splits["val"], model, device, sent_cohort),
        }
    return {
        "reference_available": True,
        "evaluation_split": "validation_only",
        "controlled_factor": "KeyRecs fixed-text DD interpretation in the Train pool",
        "protocol": "Same seed, epochs, trainer, augmentation, scaler and Validation examples. Each run has its own Train cohorts and CMU Validation threshold.",
        "reference_provenance": manifest,
        "corrected_checkpoint_sha256": checkpoint_sha(out / "models/keyprint_net.pt"),
        "validation": metrics,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ.get("KEYPRINT_OUTPUT_ROOT", root)).resolve()
    config = json.loads((out / "web/public/models/model_config.json").read_text(encoding="utf-8"))
    official = json.loads((out / "models/evaluation_report.json").read_text(encoding="utf-8"))
    fixed_path = root / "data/raw/KeyRecs-fixed-text.csv"
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        splits, meta = prepare_full_pipeline(root / "data/raw/DSL-StrongPasswordData.csv", Path(tmp), seed=SEED)
    people = meta["_keyrecs_splits_subjs"]["test"]
    audit = meta["keyrecs_fixed_timestamps"]
    mean, std = (np.asarray(config[k], np.float32) for k in ("channel_mean", "channel_std"))
    encoders, hashes = {}, {}
    for name, channels, weight in (("legacy_9ch", 9, "models/legacy/keyprint_net_9ch.pt"),
                                    ("keyprint_17ch", 17, "models/keyprint_net.pt")):
        path = out / weight
        model, cohort = load_model_and_cohort(path, channels, splits["train"], device)
        hashes[name] = checkpoint_sha(path)
        threshold = config["feature_metrics"]["password_1to1"][name]["threshold"]
        encoders[name] = (model, channels, cohort, threshold)
    model, _, cohort, _ = encoders["keyprint_17ch"]
    training_comparison = compare_training_interpretations(
        out, official, splits, meta["_sentence_splits"], model, cohort, mean, std, device,
    )
    results = {
        interpretation: eval_password_transfer(
            fixed_path, {}, people, config["outlier_bounds"], mean, std, encoders, device,
            fixed_interpretation=interpretation, seed=SEED,
        )["keyrecs_fixed_cross_session"]
        for interpretation in audit["interpretations"]
    }
    report = {"fixed_text_column_identity_audit": audit,
              "free_text_column_identity_audit": column_identity_audit(root / "data/raw/KeyRecs-free-text.csv"),
              "fixed_text_frozen_model_sensitivity": results,
              "frozen_checkpoint_sha256": hashes,
              "matched_17ch_training_comparison": training_comparison,
              "protocol": "Test fixed text: 18 people, first 4 Session-1 enrolment, Session-2 genuine and 10 Session-2 queries per other person. Fixed weights, paired trials, scaler, CMU cohorts and Validation thresholds under both input mappings.",
              "interpretation": "Train timing identities identify the physical DD field as the published UU field. Training and fixed-text evaluation use DD from this field and derive UD = DD − Hcurr. The held-out input-swap diagnostic changes input interpretation alone. The 17ch training comparison is separately scored on Validation."}
    path = out / "models/timestamp_sensitivity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
