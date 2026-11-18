"""Read-only DD/UU hypothesis sensitivity with frozen models and calibration.

The release-clock hypothesis is a diagnostic, not a correction to training data.
Outputs models/timestamp_sensitivity.json; does not change any model/config.
"""
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch

import pandas as pd
from feature_eval import eval_password_transfer
from train_and_evaluate import compute_cohort_centroids, encode_sequences
from models import KeyPrintNet
from preprocessing import prepare_full_pipeline, KEYRECS_DD_COLS, KEYRECS_HOLD_COLS


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


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ.get("KEYPRINT_OUTPUT_ROOT", root)).resolve()
    config = json.loads((out / "web/public/models/model_config.json").read_text())
    fixed_path = root / "data/raw/KeyRecs-fixed-text.csv"
    df = pd.read_csv(fixed_path)
    df.columns = [c.strip() for c in df.columns]
    dd_cols = KEYRECS_DD_COLS
    uu_cols = [c.replace("DD.", "UU.") for c in dd_cols]
    du_cols = [c.replace("DD.", "DU.") for c in dd_cols]
    ud_cols = [c.replace("DD.", "UD.") for c in dd_cols]
    h = df[KEYRECS_HOLD_COLS].to_numpy()
    dd, uu = df[dd_cols].to_numpy(), df[uu_cols].to_numpy()
    du, ud = df[du_cols].to_numpy(), df[ud_cols].to_numpy()
    def describe(values):
        values = np.abs(values[np.isfinite(values)])
        return {"n": len(values), "median_absolute_error_sec": float(np.median(values)),
                "max_absolute_error_sec": float(values.max()),
                "fraction_within_2ms": float(np.mean(values <= .0020001))}
    audit = {
        "published_labels": {
            "UD_equals_DD_minus_Hcurr": describe(ud - (dd-h[:, :-1])),
            "DU_equals_DD_plus_Hnext": describe(du - (dd+h[:, 1:])),
            "UU_equals_DD_plus_Hnext_minus_Hcurr": describe(uu - (dd+h[:, 1:]-h[:, :-1])),
        },
        "hypothesis_DD_UU_and_DU_UD_labels_swapped": {
            "UD_equals_DD_minus_Hcurr": describe(du - (uu-h[:, :-1])),
            "DU_equals_DD_plus_Hnext": describe(ud - (uu+h[:, 1:])),
            "UU_equals_DD_plus_Hnext_minus_Hcurr": describe(dd - (uu+h[:, 1:]-h[:, :-1])),
        },
        "DD_vs_UU_absolute_difference_sec": describe(dd-uu),
    }
    with tempfile.TemporaryDirectory() as tmp:
        splits, meta = prepare_full_pipeline(root / "data/raw/DSL-StrongPasswordData.csv", Path(tmp))
        people = meta["_keyrecs_splits_subjs"]["test"]
        mean, std = (np.asarray(config[k], np.float32) for k in ("channel_mean", "channel_std"))
        encoders = {}
        for name, channels, weight in (("legacy_9ch", 9, "models/legacy/keyprint_net_9ch.pt"),
                                        ("keyprint_17ch", 17, "models/keyprint_net.pt")):
            model = KeyPrintNet(in_channels=channels)
            model.load_state_dict(torch.load(out / weight, weights_only=True, map_location="cpu"))
            model.eval()
            cohort = compute_cohort_centroids(encode_sequences(model, splits["train"].seq_scaled[:, :channels], torch.device("cpu")),
                                              splits["train"].subject_ids)
            threshold = config["feature_metrics"]["password_1to1"][name]["threshold"]
            encoders[name] = (model, channels, cohort, threshold)
        results = {}
        for label, use_uu in (("published_DD", False), ("hypothesis_use_UU_as_DD", True)):
            path = fixed_path
            if use_uu:
                candidate = df.copy()
                candidate[dd_cols] = candidate[uu_cols].to_numpy()
                path = Path(tmp) / "fixed_uu_hypothesis.csv"
                candidate.to_csv(path, index=False)
            results[label] = eval_password_transfer(path, {}, people, config["outlier_bounds"], mean, std,
                                                    encoders, torch.device("cpu"), seed=42)["keyrecs_fixed_cross_session"]
    report = {"fixed_text_column_identity_audit": audit,
              "free_text_column_identity_audit": column_identity_audit(root / "data/raw/KeyRecs-free-text.csv"),
              "fixed_text_frozen_model_sensitivity": results,
              "protocol": "Test-only 18-person KeyRecs fixed text; first 4 Session-1 enrollment, Session-2 genuine; 10 Session-2 queries per other person. Same paired trials, original frozen 9ch/17ch weights, scaler, CMU cohorts and Validation thresholds.",
              "interpretation": "A DD/UU label swap is an arithmetic hypothesis, not confirmed logger semantics. This input sensitivity is not retraining under the hypothesis. Neither hypothesis nor a model was selected on Test; training/calibration remain unchanged pending source confirmation."}
    path = out / "models/timestamp_sensitivity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
