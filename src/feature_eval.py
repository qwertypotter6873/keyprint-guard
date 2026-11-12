"""
Evaluation helpers for the key-identity channels, the touch (hold-less) input path and the
behavioural-profile fusion. Everything here scores captures exactly the way the browser does
(see capture_sim.py), so the reported numbers describe the deployed web app.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from behavior_profile import BehaviorModel, Event, merge, summarize
from capture_sim import (
    as_norm_similarity,
    build_input_tensor,
    fuse,
    make_template,
    simulate_captures,
)
from keyboard_layout import CMU_PASSWORD_CODES, pairs_from_codes


# ---------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def encode_variable(model, tensors: Sequence[np.ndarray], device, in_ch: int) -> np.ndarray:
    """Encode variable-length (C, M) tensors, batching rows of equal length."""
    model.eval()
    out = np.zeros((len(tensors), 64), dtype=np.float32)
    by_len: Dict[int, List[int]] = defaultdict(list)
    for i, t in enumerate(tensors):
        by_len[t.shape[1]].append(i)
    for _, idx in by_len.items():
        x = np.stack([tensors[i][:in_ch] for i in idx], axis=0).astype(np.float32)
        for s in range(0, len(idx), 256):
            xb = torch.from_numpy(x[s : s + 256]).to(device)
            out[idx[s : s + 256]] = model(xb).cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------
# CMU password strokes -> behavioural statistics (only pause rhythm is observable here)
# ---------------------------------------------------------------------------------------
def stroke_events(holds: Sequence[float], dds: Sequence[float], codes: Sequence[str]) -> List[Event]:
    t = np.concatenate([[0.0], np.cumsum(dds)])
    evs = []
    for code, tt, h in zip(codes, t, holds):
        if code == "Enter":
            continue  # Enter submits the form in the web app and is never recorded
        kind = "space" if code == "Space" else "char"
        evs.append(Event(kind=kind, t=float(tt), hold=float(h)))
    return evs


def cmu_behavior_stats(holds: np.ndarray, dds: np.ndarray) -> List[Dict]:
    return [summarize(stroke_events(h, d, CMU_PASSWORD_CODES)) for h, d in zip(holds, dds)]


def flat_to_hdu(flat: np.ndarray, m: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    holds = np.stack([flat[:, 3 * k] for k in range(m)] + [flat[:, 3 * m]], axis=1)
    dds = np.stack([flat[:, 3 * k + 1] for k in range(m)], axis=1)
    return holds, dds


def score_trials(
    embeddings: np.ndarray,
    bot_embeddings: np.ndarray,
    trials: Dict,
    cohort: np.ndarray,
    stats: Optional[List[Dict]] = None,
    bot_stats: Optional[List[Dict]] = None,
    behavior_model: Optional[BehaviorModel] = None,
    beta: float = 0.0,
    top_m: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """1:1 verification exactly as the web scores it (AS-Norm + optional behaviour fusion)."""
    y_true, y_score = [], []
    for g in trials["groups"]:
        tmpl = make_template(embeddings[g["enroll_idx"]])
        if stats is not None:
            beh = None
            for i in g["enroll_idx"]:
                beh = merge(beh, stats[i])
            tmpl["behavior"] = beh

        def _s(emb, st):
            s = as_norm_similarity(emb, tmpl, cohort, top_m)
            if behavior_model is not None and beta > 0 and st is not None:
                b, _ = behavior_model.llr(st, tmpl["behavior"])
                s = fuse(s, b, beta)
            return s

        for i in g["gen_idx"]:
            y_true.append(1)
            y_score.append(_s(embeddings[i], stats[i] if stats else None))
        for i in g["imp_idx"]:
            y_true.append(0)
            y_score.append(_s(embeddings[i], stats[i] if stats else None))
        for i in g["bot_idx"]:
            y_true.append(0)
            y_score.append(_s(bot_embeddings[i], bot_stats[i] if bot_stats else None))
    return np.asarray(y_true), np.asarray(y_score)


# ---------------------------------------------------------------------------------------
# Web-faithful capture banks (KeyRecs raw streams replayed through the browser recorder)
# ---------------------------------------------------------------------------------------
def build_capture_bank(
    raw_sessions: Dict[Tuple[str, int], Sequence],
    participants: Sequence[str],
    touch: bool,
    target_chars: int,
) -> Dict[str, Dict[int, List[Dict]]]:
    bank: Dict[str, Dict[int, List[Dict]]] = {}
    for p in participants:
        if (p, 1) not in raw_sessions or (p, 2) not in raw_sessions:
            continue
        bank[p] = {s: simulate_captures(raw_sessions[(p, s)], target_chars=target_chars, touch=touch) for s in (1, 2)}
    return bank


def attach_tensors(bank, bounds, impute_hold, mean, std) -> None:
    for p in bank:
        for s in bank[p]:
            for cap in bank[p][s]:
                cap["tensor"] = build_input_tensor(cap, bounds, impute_hold, mean, std)


def embed_bank(bank, model, device, in_ch: int, key: str) -> None:
    flat = [(p, s, i) for p in bank for s in bank[p] for i in range(len(bank[p][s]))]
    embs = encode_variable(model, [bank[p][s][i]["tensor"] for p, s, i in flat], device, in_ch)
    for (p, s, i), e in zip(flat, embs):
        bank[p][s][i][key] = e


def eval_web_identification(
    bank,
    emb_key: str,
    cohort: np.ndarray,
    top_m: int = 16,
    behavior_model: Optional[BehaviorModel] = None,
    beta: float = 0.0,
    k_enroll: int = 3,
    behavior_only: bool = False,
) -> Dict[str, Dict[str, float]]:
    """1:N "Who is typing?" with K web captures enrolled from Session 1 per person."""
    people = [p for p in sorted(bank) if len(bank[p][1]) > k_enroll and len(bank[p][2]) > 0]
    gallery = []
    for p in people:
        enroll = bank[p][1][:k_enroll]
        beh = None
        for c in enroll:
            beh = merge(beh, c["behavior"])
        t = make_template(np.stack([c[emb_key] for c in enroll]) if emb_key else np.zeros((1, 64)), beh)
        gallery.append(t)

    res = {}
    for split, sel in (("same_session", lambda p: bank[p][1][k_enroll:]), ("cross_session", lambda p: bank[p][2])):
        y_true, y_pred, top3 = [], [], []
        for ti, p in enumerate(people):
            for cap in sel(p):
                scores = []
                for tmpl in gallery:
                    if behavior_only:
                        s, _ = behavior_model.llr(cap["behavior"], tmpl["behavior"])
                    else:
                        s = as_norm_similarity(cap[emb_key], tmpl, cohort, top_m)
                        if behavior_model is not None and beta > 0:
                            b, _ = behavior_model.llr(cap["behavior"], tmpl["behavior"])
                            s = fuse(s, b, beta)
                    scores.append(s)
                order = np.argsort(-np.asarray(scores))
                y_true.append(ti)
                y_pred.append(int(order[0]))
                top3.append(ti in order[:3])
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        res[split] = {
            "num_classes": len(people),
            "num_queries": int(len(y_true)),
            "top1_accuracy": round(float(np.mean(y_true == y_pred)), 4),
            "top3_accuracy": round(float(np.mean(top3)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        }
    return res


def calibrate_open_set_threshold(
    bank,
    emb_key: str,
    cohort: np.ndarray,
    behavior_model: Optional[BehaviorModel],
    beta: float,
    top_m: int = 16,
    k_enroll: int = 3,
    n_outsiders: int = 3,
) -> Dict[str, float]:
    """Pick the Arena open-set threshold on Validation people: the first people are enrolled,
    the last `n_outsiders` act as unregistered typists who must be rejected."""
    people = [p for p in sorted(bank) if len(bank[p][1]) > k_enroll and len(bank[p][2]) > 0]
    enrolled, outsiders = people[:-n_outsiders], people[-n_outsiders:]
    gallery = []
    for p in enrolled:
        enroll = bank[p][1][:k_enroll]
        beh = None
        for c in enroll:
            beh = merge(beh, c["behavior"])
        gallery.append(make_template(np.stack([c[emb_key] for c in enroll]), beh))

    rows = []  # (max_score, is_enrolled, top1_correct)
    for p in people:
        caps = bank[p][2] + (bank[p][1][k_enroll:] if p in outsiders else [])
        for cap in caps:
            scores = []
            for tmpl in gallery:
                s = as_norm_similarity(cap[emb_key], tmpl, cohort, top_m)
                if behavior_model is not None and beta > 0:
                    b, _ = behavior_model.llr(cap["behavior"], tmpl["behavior"])
                    s = fuse(s, b, beta)
                scores.append(s)
            best = int(np.argmax(scores))
            is_enr = p in enrolled
            rows.append((max(scores), is_enr, is_enr and enrolled[best] == p))
    scores = np.asarray([r[0] for r in rows])
    is_enr = np.asarray([r[1] for r in rows])
    correct = np.asarray([r[2] for r in rows])
    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(np.percentile(scores, 2), np.percentile(scores, 98), 400):
        acc_enr = np.mean(correct[is_enr] & (scores[is_enr] >= t)) if is_enr.any() else 0.0
        rej_out = np.mean(scores[~is_enr] < t) if (~is_enr).any() else 0.0
        bal = 0.5 * (acc_enr + rej_out)
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return {"threshold": round(best_t, 4), "balanced_open_set_accuracy": round(float(best_bal), 4)}


BETA_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def tune_beta_identification(
    val_bank, emb_key: str, cohort: np.ndarray, behavior_model: BehaviorModel, grid: Sequence[float] = BETA_GRID
) -> Tuple[float, List[Dict]]:
    """Fusion weight for sentence typing, chosen on Validation people (same + cross session top-1)."""
    rows = []
    for beta in grid:
        ident = eval_web_identification(val_bank, emb_key, cohort, 16, behavior_model, beta)
        rows.append(
            {
                "beta": beta,
                "val_cross_top1": ident["cross_session"]["top1_accuracy"],
                "val_same_top1": ident["same_session"]["top1_accuracy"],
            }
        )
    best = max(rows, key=lambda r: (round(r["val_cross_top1"] + r["val_same_top1"], 4), -r["beta"]))
    return float(best["beta"]), rows


def tune_beta_password(
    cmu_args: Dict, behavior_model: BehaviorModel, grid: Sequence[float] = BETA_GRID
) -> Tuple[float, List[Dict]]:
    """Fusion weight for passwords, chosen on Validation subjects (1:1 ROC-AUC)."""
    rows = []
    for beta in grid:
        y, s = score_trials(behavior_model=behavior_model, beta=beta, **cmu_args)
        rows.append({"beta": beta, "val_password_auc": round(float(roc_auc_score(y, s)), 4)})
    best = max(rows, key=lambda r: (r["val_password_auc"], -r["beta"]))
    return float(best["beta"]), rows


# ---------------------------------------------------------------------------------------
# Web payload helpers
# ---------------------------------------------------------------------------------------
def capture_stats(cap: Dict) -> Dict[str, int]:
    dds = cap["dds"]
    holds = [h for h in cap["holds"] if h is not None]
    uds = [u for u in cap["uds"] if u is not None]
    return {
        "cpm": int(round(60.0 / (sum(dds) / len(dds)))) if dds else 0,
        "holdMs": int(round(1000.0 * sum(holds) / len(holds))) if holds else 0,
        "rolloverPct": int(round(100.0 * sum(1 for u in uds if u < 0) / len(uds))) if uds else 0,
    }


def stroke_payload(cap: Dict) -> Dict:
    r5 = lambda v: None if v is None else round(float(v), 5)  # noqa: E731
    return {
        "holds": [r5(h) for h in cap["holds"]],
        "holdValid": [bool(v) for v in cap["hold_valid"]],
        "dds": [r5(d) for d in cap["dds"]],
        "uds": [r5(u) for u in cap["uds"]],
        "stitchFlags": [bool(v) for v in cap.get("stitch_flags", [False] * len(cap["dds"]))],
        "pairKeys": cap["pair_keys"],
    }


def fixed_stroke_capture(holds: Sequence[float], dds: Sequence[float], uds: Sequence[float], codes: Sequence[str]) -> Dict:
    return {
        "holds": [float(h) for h in holds],
        "hold_valid": [True] * len(holds),
        "dds": [float(d) for d in dds],
        "uds": [float(u) for u in uds],
        "stitch_flags": [False] * len(dds),
        "pair_keys": [list(p) for p in pairs_from_codes(list(codes))],
    }


def template_payload(embs: np.ndarray, caps: Sequence[Dict], behavior: Dict, input_mode: str = "keyboard") -> Dict:
    tmpl = make_template(embs)
    stats = [capture_stats(c) for c in caps]
    avg = lambda k: int(round(sum(s[k] for s in stats) / max(1, len(stats))))  # noqa: E731
    return {
        "centroid": [round(float(v), 5) for v in tmpl["centroid"]],
        "exemplars": [[round(float(v), 5) for v in e] for e in tmpl["exemplars"]],
        "count": int(len(embs)),
        "cpm": avg("cpm"),
        "holdMs": avg("holdMs"),
        "rolloverPct": avg("rolloverPct"),
        "behavior": {k: [round(v[0], 6), v[1]] for k, v in behavior.items()},
        "inputMode": input_mode,
    }


def dd_cv(caps: Sequence[Dict]) -> float:
    dds = np.concatenate([np.asarray(c["dds"]) for c in caps])
    return float(np.std(dds) / (np.mean(dds) + 1e-8))


# ---------------------------------------------------------------------------------------
# Password encoder choice: does the score transfer to passwords the encoder never trained on?
# ---------------------------------------------------------------------------------------
def _verification_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> Dict[str, float]:
    from train_and_evaluate import compute_eer  # local import: train_and_evaluate is heavier

    far = float(((s >= threshold) & (y == 0)).sum() / max(1, (y == 0).sum()))
    frr = float(((s < threshold) & (y == 1)).sum() / max(1, (y == 1).sum()))
    return {
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "eer": round(float(compute_eer(y, s)[0]), 4),
        "far_at_deployed_threshold": round(far, 4),
        "frr_at_deployed_threshold": round(frr, 4),
        "impostor_mean_score": round(float(s[y == 0].mean()), 4),
        "genuine_mean_score": round(float(s[y == 1].mean()), 4),
        "num_genuine": int(y.sum()),
        "num_impostor": int((y == 0).sum()),
    }


def _paired_auc_diff_ci(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    return {
        "mean": round(float(np.mean(diffs)), 4),
        "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
    }


def eval_password_transfer(
    fixed_csv,
    raw_sessions: Dict[Tuple[str, int], Sequence],
    participants: Sequence[str],
    bounds: Dict[str, Tuple[float, float]],
    mean: np.ndarray,
    std: np.ndarray,
    encoders: Dict[str, Tuple[torch.nn.Module, int, np.ndarray, float]],
    device,
    seed: int = 0,
    min_word_len: int = 5,
) -> Dict:
    """
    Web-faithful 1:1 password verification on content other than the CMU cohort password, for each
    encoder (model, in_channels, CMU password cohort, CMU-calibrated threshold):

    * keyrecs_fixed_cross_session — KeyRecs fixed text ('vpwjkeurkb'); enroll = 4 session-1 reps,
      genuine = session-2 reps, impostors = 10 session-2 reps of every other unseen person. Both
      encoders saw this string from Train participants during training.
    * repeated_words_unseen — words (>= min_word_len letters) an unseen person typed cleanly
      >= 4 times in free text: a fixed string never trained as a password, like a user's own
      password. Enroll = first 3 session-1 occurrences, genuine = session-2 (else later) ones,
      impostors = up to 12 occurrences of the same word by other people.
    """
    import pandas as pd

    from behavior_profile import kind_of
    from keyboard_layout import code_for_label, key_channel_matrix
    from preprocessing import (
        KEYRECS_DD_COLS,
        KEYRECS_FIXED_KEY_BLOCK,
        KEYRECS_HOLD_COLS,
        apply_channel_scaler,
        extract_sequences,
    )

    rng = np.random.default_rng(seed)
    names = list(encoders)

    def _enc(name: str, seq: np.ndarray) -> np.ndarray:
        model, in_ch, _, _ = encoders[name]
        return encode_variable(model, list(seq), device, in_ch)

    def _score_groups(seq, groups: List[Tuple[List[int], List[int], List[int]]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        embs = {n: _enc(n, seq) for n in names}
        y, scores = [], {n: [] for n in names}
        for enroll, gen, imp in groups:
            tmpls = {n: make_template(embs[n][enroll]) for n in names}
            for idxs, lab in ((gen, 1), (imp, 0)):
                for i in idxs:
                    y.append(lab)
                    for n in names:
                        scores[n].append(as_norm_similarity(embs[n][i], tmpls[n], encoders[n][2], 12))
        return np.asarray(y), {n: np.asarray(v) for n, v in scores.items()}

    out: Dict[str, Dict] = {}

    # 1) KeyRecs fixed text, cross-session
    df = pd.read_csv(fixed_csv)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["participant"].isin(participants)].dropna(subset=KEYRECS_HOLD_COLS + KEYRECS_DD_COLS).reset_index(drop=True)
    h = np.clip(df[KEYRECS_HOLD_COLS].values.astype(np.float32), *bounds["H"])
    d = np.clip(df[KEYRECS_DD_COLS].values.astype(np.float32), *bounds["DD"])
    u = np.clip(d - h[:, :-1], *bounds["UD"])
    seq = apply_channel_scaler(extract_sequences(h, d, u, KEYRECS_FIXED_KEY_BLOCK), mean, std)
    P, S = df["participant"].values, df["session"].values
    groups = []
    for p in np.unique(P):
        own = np.where(P == p)[0]
        enroll = own[S[own] == 1][:4].tolist()
        gen = own[S[own] == 2].tolist()
        if len(enroll) < 4 or not gen:
            continue
        imp = []
        for q in np.unique(P):
            if q != p:
                cand = np.where((P == q) & (S == 2))[0]
                imp.extend(rng.choice(cand, size=min(10, len(cand)), replace=False).tolist())
        groups.append((enroll, gen, imp))
    y, sc = _score_groups(seq, groups)
    out["keyrecs_fixed_cross_session"] = {n: _verification_metrics(y, sc[n], encoders[n][3]) for n in names}
    if len(names) >= 2:
        out["keyrecs_fixed_cross_session"]["auc_diff_first_minus_second"] = _paired_auc_diff_ci(y, sc[names[0]], sc[names[1]], seed=seed)

    # 2) Repeated words of unseen people (content never trained as a fixed password)
    occ: Dict[str, Dict[str, Dict[int, List]]] = defaultdict(lambda: defaultdict(lambda: {1: [], 2: []}))
    for p in participants:
        for s in (1, 2):
            cur, ok = [], True
            for label, t, hold in list(raw_sessions.get((p, s), [])) + [("Space", 1e12, 0.1)]:
                kind, _ = kind_of(label)
                if kind == "space":
                    if ok and len(cur) >= min_word_len:
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
    seqs, groups, words = [], [], []
    for w, per in occ.items():
        enrolled = [p for p, dd_ in per.items() if len(dd_[1]) >= 3 and len(dd_[1]) + len(dd_[2]) >= 4]
        if len(enrolled) < 3 or len(per) < 4:
            continue
        kb = key_channel_matrix(pairs_from_codes([code_for_label(c) for c in w]))
        index = {}
        for p, dd_ in per.items():
            for s in (1, 2):
                for k, (hs, dd) in enumerate(dd_[s]):
                    hh = np.clip(hs, *bounds["H"]).astype(np.float32)
                    dc = np.clip(dd, *bounds["DD"]).astype(np.float32)
                    uu = np.clip(dc - hh[:-1], *bounds["UD"]).astype(np.float32)
                    seqs.append(apply_channel_scaler(extract_sequences(hh[None], dc[None], uu[None], kb[None]), mean, std)[0])
                    index[(p, s, k)] = len(seqs) - 1
        used = 0
        for p in enrolled:
            s1 = [index[(p, 1, k)] for k in range(len(per[p][1]))]
            gen = [index[(p, 2, k)] for k in range(len(per[p][2]))] or s1[3:]
            if not gen:
                continue
            imp = [i for (q, _, _), i in index.items() if q != p][:12]
            groups.append((s1[:3], gen, imp))
            used += 1
        if used:
            words.append({"word": w, "enrolled_people": used})
    if groups:
        y, sc = _score_groups(seqs, groups)
        res = {n: _verification_metrics(y, sc[n], encoders[n][3]) for n in names}
        if len(names) >= 2:
            res["auc_diff_first_minus_second"] = _paired_auc_diff_ci(y, sc[names[0]], sc[names[1]], seed=seed)
        res["words"] = words
        out["repeated_words_unseen"] = res
    return out
