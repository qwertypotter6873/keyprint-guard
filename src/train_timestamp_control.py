"""Retrain the matched 17-channel timestamp control with the published KeyRecs column names."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from preprocessing import N_CHANNELS, SEED, TOUCH_PROB, load_keyrecs_fixed_split, prepare_full_pipeline
from train_and_evaluate import (
    KEYPRINT_DEFAULT_EPOCHS, build_verification_trials, checkpoint_training_recipe,
    get_device, train_keyprint_net,
)


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("KEYPRINT_OUTPUT_ROOT", root)) / "models" / "legacy"
    output.mkdir(parents=True, exist_ok=True)
    epochs = int(os.environ.get("KEYPRINT_SMOKE_EPOCHS", KEYPRINT_DEFAULT_EPOCHS))
    device = torch.device(os.environ["KEYPRINT_DEVICE"]) if "KEYPRINT_DEVICE" in os.environ else get_device()
    raw = root / "data" / "raw"
    with tempfile.TemporaryDirectory() as work:
        splits, meta = prepare_full_pipeline(raw / "DSL-StrongPasswordData.csv", Path(work), seed=SEED)
    mean = np.asarray(meta["channel_mean"], dtype=np.float32)
    std = np.asarray(meta["channel_std"], dtype=np.float32)
    val_trials = build_verification_trials(splits["val"], mean, std, k_shot=5, seed=SEED)
    people = meta["_keyrecs_splits_subjs"]["train"]
    fixed = load_keyrecs_fixed_split(raw / "KeyRecs-fixed-text.csv", people, meta["outlier_bounds"],
                                     mean, std, interpretation="published_labels")
    sentences = meta["_sentence_splits"]
    model, history = train_keyprint_net(
        splits["train"], splits["val"], val_trials, meta["outlier_bounds"], mean, std,
        device, epochs=epochs, batch_size=256, use_augmentation=True, raw_channels_only=False,
        sentence_train=sentences["train"], kr_fixed_train=fixed, sentence_val=sentences["val"],
        tag="KeyPrintNet_17ch_published_DD", in_channels=N_CHANNELS, touch_prob=TOUCH_PROB,
        touch_hold=meta["touch_hold_impute_sec"],
    )
    weights = output / "keyprint_net_17ch_published_dd.pt"
    torch.save(model.cpu().state_dict(), weights)
    record = {
        "training_recipe": checkpoint_training_recipe("published_labels", epochs, epochs),
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "epochs": epochs,
        "device": str(device), "train_subjects": people, "validation_history": history,
    }
    weights.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved matched timestamp control {weights.name}", flush=True)


if __name__ == "__main__":
    main()
