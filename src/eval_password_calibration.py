"""
Password 1:1 — is the deployed decision threshold still right on a password the user chose?

The web scores a password attempt with AS-Norm against cohort centroids of CMU subjects typing
`.tie5Roanl`, and accepts it above a threshold calibrated on CMU validation people. A user's own
password is different content, so this script checks how well that calibration transfers, for
the deployed 9-channel password encoder, on keyboard and simulated touch (holds unknown):

  * cmu_*            the CMU password itself (val / test people, cross-session genuine queries)
  * krfixed_*        KeyRecs fixed text `vpwjkeurkb`, enrolled on day 1, verified on day 2
  * words_same_day_* / words_cross_day_*
                     words unseen people typed repeatedly in free text (`things`, `people`,
                     `because`): content never trained as a password; genuine attempts split
                     into the enrollment day and the following day

Two decision rules:
  * deployed  — AS-Norm score >= the deployed threshold (web/public/models/model_config.json)
  * raw       — template cosine similarity (no cohort) >= a threshold tuned on CMU val
                (max macro-F1, the pipeline's rule). Nothing else is used for tuning.

Prints a table and writes models/password_calibration.json.

    python src/eval_password_calibration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_profile import kind_of, load_keyrecs_sessions  # noqa: E402
from capture_sim import make_template, ms_rounded  # noqa: E402
from feature_eval import encode_variable, flat_to_hdu  # noqa: E402
from keyboard_layout import code_for_label, key_channel_matrix, pairs_from_codes  # noqa: E402
from models import KeyPrintNet  # noqa: E402
from preprocessing import (  # noqa: E402
    KEYRECS_DD_COLS,
    KEYRECS_FIXED_KEY_BLOCK,
    KEYRECS_HOLD_COLS,
    N_TIMING_CHANNELS,
    SEED,
    apply_channel_scaler,
    extract_sequences,
    get_feature_columns,
    prepare_full_pipeline,
    touchify_timings,
)
from train_and_evaluate import build_verification_trials, compute_cohort_centroids, encode_sequences  # noqa: E402

L9 = N_TIMING_CHANNELS
TOP_M = 12
MODES = ("kb", "touch")


def build_scenarios(root: Path, device) -> tuple[dict, dict]:
    """Trial groups (enroll embeddings, genuine embeddings, impostor embeddings) per scenario."""
    with tempfile.TemporaryDirectory() as tmp:  # keep data/processed untouched
        splits, meta = prepare_full_pipeline(root / "data" / "raw" / "DSL-StrongPasswordData.csv", Path(tmp), seed=SEED)
    kr = meta["_keyrecs_splits_subjs"]
    mean = np.array(meta["channel_mean"], dtype=np.float32)
    std = np.array(meta["channel_std"], dtype=np.float32)
    bounds = {k: tuple(v) for k, v in meta["outlier_bounds"].items()}
    touch_hold = float(meta["touch_hold_impute_sec"])

    model = KeyPrintNet(in_channels=L9, hidden_channels=80, embed_dim=64, dropout=0.18)
    model.load_state_dict(torch.load(root / "models" / "legacy" / "keyprint_net_9ch.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()

    def seqs(h, d, kb, touch):
        if touch:
            h2, u2 = touchify_timings(h, d, touch_hold, bounds)
            return apply_channel_scaler(extract_sequences(h2, d, u2, kb, np.zeros(len(d), dtype=np.float32)), mean, std)
        u = np.clip(d - h[:, :-1], *bounds["UD"])
        return apply_channel_scaler(extract_sequences(h, d, u, kb), mean, std)

    _, hold_cols, dd_cols, _ = get_feature_columns()

    def cmu_hd(split):
        return split.df[hold_cols].values.astype(np.float32), split.df[dd_cols].values.astype(np.float32)

    tr_h, tr_d = cmu_hd(splits["train"])
    cohorts = {
        "kb": compute_cohort_centroids(encode_sequences(model, splits["train"].seq_scaled[:, :L9], device), splits["train"].subject_ids),
        "touch": compute_cohort_centroids(
            encode_sequences(model, seqs(tr_h, tr_d, splits["train"].key_block, True)[:, :L9], device), splits["train"].subject_ids
        ),
    }
    scen: dict = {}

    # 1) CMU password: the pipeline's trials, first 4 enrollment reps like the web
    for part in ("val", "test"):
        trials = build_verification_trials(splits[part], mean, std, k_shot=5, seed=SEED)
        h, d = cmu_hd(splits[part])
        bh, bd = flat_to_hdu(trials["bot_flat"])
        for mode in MODES:
            if mode == "kb":
                emb = encode_sequences(model, splits[part].seq_scaled[:, :L9], device)
                bemb = encode_sequences(model, trials["bot_seq"][:, :L9], device)
            else:
                emb = encode_sequences(model, seqs(h, d, splits[part].key_block, True)[:, :L9], device)
                bemb = encode_sequences(model, seqs(bh.astype(np.float32), bd.astype(np.float32), splits[part].key_block, True)[:, :L9], device)
            scen[f"cmu_{part}_{mode}"] = [
                (emb[g["enroll_idx"][:4]], emb[g["gen_idx"]], np.concatenate([emb[g["imp_idx"]], bemb[g["bot_idx"]]]))
                for g in trials["groups"]
            ]

    # 2) KeyRecs fixed text: enroll 4 reps on day 1, genuine = day 2, impostors = 10 day-2 reps of each other person
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(root / "data" / "raw" / "KeyRecs-fixed-text.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=KEYRECS_HOLD_COLS + KEYRECS_DD_COLS).reset_index(drop=True)
    for part in ("val", "test"):
        sub = df[df["participant"].isin(kr[part])].reset_index(drop=True)
        h = np.clip(sub[KEYRECS_HOLD_COLS].values.astype(np.float32), *bounds["H"])
        d = np.clip(sub[KEYRECS_DD_COLS].values.astype(np.float32), *bounds["DD"])
        P, S = sub["participant"].values, sub["session"].values
        imp_pick = {}
        for p in np.unique(P):
            imp = []
            for q in np.unique(P):
                if q != p:
                    cand = np.where((P == q) & (S == 2))[0]
                    imp.extend(rng.choice(cand, size=min(10, len(cand)), replace=False).tolist())
            imp_pick[p] = imp
        for mode in MODES:
            emb = encode_variable(model, list(seqs(h, d, KEYRECS_FIXED_KEY_BLOCK, mode == "touch")), device, L9)
            groups = []
            for p in np.unique(P):
                own = np.where(P == p)[0]
                enroll, gen = own[S[own] == 1][:4], own[S[own] == 2]
                if len(enroll) == 4 and len(gen):
                    groups.append((emb[enroll], emb[gen], emb[imp_pick[p]]))
            scen[f"krfixed_{part}_{mode}"] = groups

    # 3) Repeated words of unseen people (val + test), same selection as feature_eval.eval_password_transfer
    raw_sessions = {k: ms_rounded(v) for k, v in load_keyrecs_sessions(root / "data" / "raw" / "KeyRecs-free-text.csv").items()}
    occ = defaultdict(lambda: defaultdict(lambda: {1: [], 2: []}))
    for p in kr["val"] + kr["test"]:
        for s in (1, 2):
            cur, ok = [], True
            for label, t, hold in list(raw_sessions.get((p, s), [])) + [("Space", 1e12, 0.1)]:
                kind, _ = kind_of(label)
                if kind == "space":
                    if ok and len(cur) >= 5:
                        ts = np.array([c[1] for c in cur])
                        hs = np.array([c[2] for c in cur])
                        dd = np.diff(ts)
                        if not ((dd <= 0.006).any() or (dd > 2.2).any() or (hs > 1.2).any() or (hs < 0.008).any()):
                            occ["".join(c[0] for c in cur)][p][s].append((hs, dd))
                    cur, ok = [], True
                elif kind == "char" and len(label) == 1 and label.isalpha():
                    cur.append((label.lower(), t, hold))
                elif kind in ("shift", "caps"):
                    continue
                else:
                    ok = False
    words = []
    for mode in MODES:
        same, cross = [], []
        for w, per in occ.items():
            enrolled = [p for p, v in per.items() if len(v[1]) >= 3 and len(v[1]) + len(v[2]) >= 4]
            if len(enrolled) < 3 or len(per) < 4:
                continue
            kbm = key_channel_matrix(pairs_from_codes([code_for_label(c) for c in w]))
            tensors, index = [], {}
            for p, v in per.items():
                for s in (1, 2):
                    for k, (hs, dd) in enumerate(v[s]):
                        hh = np.clip(hs, *bounds["H"]).astype(np.float32)[None]
                        dc = np.clip(dd, *bounds["DD"]).astype(np.float32)[None]
                        tensors.append(seqs(hh, dc, kbm[None], mode == "touch")[0])
                        index[(p, s, k)] = len(tensors) - 1
            emb = encode_variable(model, tensors, device, L9)
            for p in enrolled:
                s1 = [index[(p, 1, k)] for k in range(len(per[p][1]))]
                s2 = [index[(p, 2, k)] for k in range(len(per[p][2]))]
                imp = [i for (q, _, _), i in index.items() if q != p][:12]
                if s1[3:]:
                    same.append((emb[s1[:3]], emb[s1[3:]], emb[imp]))
                if s2:
                    cross.append((emb[s1[:3]], emb[s2], emb[imp]))
            if mode == "kb":
                words.append(w)
        scen[f"words_same_day_{mode}"] = same
        scen[f"words_cross_day_{mode}"] = cross
    return scen, {"cohorts": cohorts, "words": words}


def score(groups, cohort):
    """Labels, deployed AS-Norm scores and raw template similarities (same formula as the web)."""
    y, a, r = [], [], []
    for E, G, I in groups:
        tmpl = make_template(np.asarray(E, dtype=np.float64))
        u = np.sort(cohort @ tmpl["centroid"])[-TOP_M:]
        for Q, lab in ((G, 1), (I, 0)):
            Q = np.asarray(Q, dtype=np.float64)
            raw = 0.65 * (Q @ tmpl["centroid"]) + 0.35 * np.max(Q @ tmpl["exemplars"].T, axis=1)
            q = np.sort(Q @ cohort.T, axis=1)[:, -TOP_M:]
            z = (raw - 0.5 * (u.mean() + q.mean(1))) / (0.5 * (u.std() + 0.05 + q.std(1) + 0.05))
            a += list(1.0 / (1.0 + np.exp(-0.85 * (z - 0.15))))
            r += list(raw)
            y += [lab] * len(Q)
    return np.asarray(y), np.asarray(a), np.asarray(r)


def f1_threshold(y, s):
    cand = np.linspace(np.percentile(s, 5), np.percentile(s, 95), 300)
    return float(cand[int(np.argmax([f1_score(y, (s >= t).astype(int), average="macro") for t in cand]))])


def rates(y, s, thr):
    far = float((s[y == 0] >= thr).mean())
    frr = float((s[y == 1] < thr).mean())
    return {"far": round(far, 4), "frr": round(frr, 4), "hter": round((far + frr) / 2, 4)}


def eer_point(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return round(float((fpr[i] + 1 - tpr[i]) / 2), 4), round(float(thr[i]), 4)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cpu")
    cfg = json.loads((root / "web" / "public" / "models" / "model_config.json").read_text(encoding="utf-8"))
    deployed = {"kb": float(cfg["calibrated_threshold"]), "touch": float(cfg["calibrated_threshold_touch"])}

    scen, info = build_scenarios(root, device)
    scored = {k: score(v, info["cohorts"]["touch" if k.endswith("touch") else "kb"]) for k, v in scen.items()}
    raw_thr = {m: f1_threshold(scored[f"cmu_val_{m}"][0], scored[f"cmu_val_{m}"][2]) for m in MODES}

    out = {
        "deployed_threshold": deployed,
        "raw_threshold_cmu_val": {m: round(v, 4) for m, v in raw_thr.items()},
        "words": info["words"],
        "scenarios": {},
    }
    print(f"deployed AS-Norm thresholds {deployed} | raw thresholds tuned on CMU val {out['raw_threshold_cmu_val']}")
    print(f"{'scenario':24s} {'gen/imp':>10s} {'AUC':>6s} {'EER':>6s} {'EER thr':>8s} | {'deployed FAR / FRR':>18s} | {'raw FAR / FRR':>15s}")
    for k, (y, a, r) in scored.items():
        m = "touch" if k.endswith("touch") else "kb"
        eer, eer_thr = eer_point(y, a)
        dep, raw = rates(y, a, deployed[m]), rates(y, r, raw_thr[m])
        out["scenarios"][k] = {
            "num_genuine": int((y == 1).sum()),
            "num_impostor": int((y == 0).sum()),
            "as_norm_auc": round(float(roc_auc_score(y, a)), 4),
            "as_norm_eer": eer,
            "as_norm_eer_threshold": eer_thr,
            "raw_auc": round(float(roc_auc_score(y, r)), 4),
            "deployed": dep,
            "raw_cmu_val_threshold": raw,
        }
        print(
            f"{k:24s} {int((y == 1).sum()):4d}/{int((y == 0).sum()):<5d} {out['scenarios'][k]['as_norm_auc']:6.3f} {eer * 100:5.1f}% {eer_thr:8.3f} | "
            f"{dep['far'] * 100:7.1f} / {dep['frr'] * 100:5.1f}% | {raw['far'] * 100:5.1f} / {raw['frr'] * 100:5.1f}%"
        )
    path = root / "models" / "password_calibration.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
