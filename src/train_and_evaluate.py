"""
End-to-end training, 1:1 biometric hacker-blocking verification, 1:N multi-user
typing-practice identification ("Who Is Typing?"), ablation study, and ONNX export
for KeyPrint Guard & Arena.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocessing import (
    CMU_KEY_BLOCK,
    KEYRECS_FIXED_KEY_BLOCK,
    N_CHANNELS,
    SEED,
    TOUCH_PROB,
    SentenceSplitData,
    SplitData,
    apply_channel_scaler,
    augment_keystroke_batch,
    extract_sequences,
    get_feature_columns,
)
from models import (
    KeyPrintNet,
    ScaledManhattanDetector,
    SubCenterCosFaceHead,
    SupervisedContrastiveLoss,
    Vanilla1DCNN,
)


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_korean_matplotlib() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for cand in [
        "Apple SD Gothic Neo",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
    ]:
        if cand in available:
            chosen = cand
            break
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 200
    return chosen


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_sequences(
    model: nn.Module, seq_scaled: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(seq_scaled), batch_size):
        xb = torch.from_numpy(seq_scaled[i : i + batch_size]).to(device)
        emb = model(xb).cpu().numpy()
        outs.append(emb)
    return np.concatenate(outs, axis=0)


def generate_bot_sequences(
    n_bots: int,
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate automated attack traffic (Macro Bots) typing the exact password:
    1) Uniform Constant-Delay Macro Bot (fixed hold/flight + tiny OS timer jitter)
    2) Random Gaussian Human-Mimic Bot (i.i.d. per-key delays ignoring digraph phrasing)
    """
    rng = np.random.RandomState(seed + 999)
    half = n_bots // 2

    base_h_a = rng.uniform(0.055, 0.110, size=(half, 1))
    base_dd_a = rng.uniform(0.090, 0.220, size=(half, 1))
    holds_a = np.clip(base_h_a + rng.normal(0, 0.003, size=(half, 11)), 0.015, 0.40)
    dds_a = np.clip(base_dd_a + rng.normal(0, 0.004, size=(half, 10)), 0.025, 1.00)
    uds_a = dds_a - holds_a[:, :-1]

    rem = n_bots - half
    holds_b = np.clip(rng.uniform(0.045, 0.165, size=(rem, 11)), 0.015, 0.40)
    dds_b = np.clip(rng.uniform(0.080, 0.380, size=(rem, 10)), 0.025, 1.00)
    uds_b = dds_b - holds_b[:, :-1]

    holds = np.concatenate([holds_a, holds_b], axis=0).astype(np.float32)
    dds = np.concatenate([dds_a, dds_b], axis=0).astype(np.float32)
    uds = np.concatenate([uds_a, uds_b], axis=0).astype(np.float32)

    flat_parts = []
    for k in range(10):
        flat_parts.extend([holds[:, k : k + 1], dds[:, k : k + 1], uds[:, k : k + 1]])
    flat_parts.append(holds[:, 10:11])
    flat = np.concatenate(flat_parts, axis=1).astype(np.float32)

    seq_unscaled = extract_sequences(holds, dds, uds, CMU_KEY_BLOCK)
    seq_scaled = apply_channel_scaler(seq_unscaled, chan_mean, chan_std)
    return flat, seq_scaled


def build_verification_trials(
    split: SplitData,
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
    k_shot: int = 5,
    seed: int = SEED,
    session1_only_enroll: bool = False,
) -> Dict:
    """
    Construct open-set 1:1 biometric verification trials on unseen subjects:
    - Genuine queries strictly from DIFFERENT sessions than enrollment (cross-session)
    - Skilled human impostors typing the exact same password
    - Automated macro-bot impostors
    """
    rng = np.random.RandomState(seed)
    n_subjs = len(split.subjects)
    bot_flat, bot_seq = generate_bot_sequences(240, chan_mean, chan_std, seed=seed)

    enroll_groups = []
    sessions_to_enroll = [1] if session1_only_enroll else [1, 3, 5, 7]

    for uid in range(n_subjs):
        u_mask = split.subject_ids == uid
        for sess in sessions_to_enroll:
            enroll_idx = np.where(u_mask & (split.sessions == sess) & (split.reps <= k_shot))[0]
            gen_pool = np.where(u_mask & (split.sessions != sess))[0]
            gen_q = rng.choice(gen_pool, size=min(45, len(gen_pool)), replace=False)

            imp_pool = np.where(split.subject_ids != uid)[0]
            imp_q = rng.choice(imp_pool, size=40, replace=False)
            bot_q = rng.choice(len(bot_flat), size=5, replace=False)

            enroll_groups.append(
                {
                    "uid": uid,
                    "subject": split.subjects[uid],
                    "enroll_session": sess,
                    "enroll_idx": enroll_idx,
                    "gen_idx": gen_q,
                    "imp_idx": imp_q,
                    "bot_idx": bot_q,
                }
            )

    return {
        "groups": enroll_groups,
        "bot_flat": bot_flat,
        "bot_seq": bot_seq,
    }


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    eer_thresh = float(thresholds[idx])
    return eer, eer_thresh, fpr, fnr, thresholds


def compute_cohort_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute unit-normalized background cohort centroids on Train split for AS-Norm."""
    centroids = []
    for uid in np.unique(labels):
        c = embeddings[labels == uid].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids.append(c)
    return np.stack(centroids, axis=0).astype(np.float32)


def score_embedding_trials(
    embeddings: np.ndarray,
    bot_embeddings: np.ndarray,
    trials: Dict,
    cohort: Optional[np.ndarray] = None,
    top_m: int = 12,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Score 1:1 verification trials using hybrid centroid + nearest-exemplar cosine similarity,
    calibrated with Adaptive Symmetric Cohort Normalization (AS-Norm) mapped to (0, 1).
    """
    y_true_list, y_score_list = [], []
    detailed_records: List[Dict] = []

    for g in trials["groups"]:
        e_vecs = embeddings[g["enroll_idx"]]
        centroid = e_vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        if cohort is not None:
            k_top = min(top_m, len(cohort))
            c_coh = np.sort(cohort @ centroid)[-k_top:]
            mu_u = float(c_coh.mean())
            sd_u = float(c_coh.std() + 0.05)

        def _sim(q_vecs: np.ndarray) -> np.ndarray:
            raw = 0.65 * (q_vecs @ centroid) + 0.35 * np.max(q_vecs @ e_vecs.T, axis=1)
            if cohort is None:
                return raw
            k_top = min(top_m, len(cohort))
            q_coh = np.sort(q_vecs @ cohort.T, axis=1)[:, -k_top:]
            mu_q = q_coh.mean(axis=1)
            sd_q = q_coh.std(axis=1) + 0.05
            z_as = (raw - 0.5 * (mu_u + mu_q)) / (0.5 * (sd_u + sd_q))
            return 1.0 / (1.0 + np.exp(-0.85 * (z_as - 0.15)))

        gen_scores = _sim(embeddings[g["gen_idx"]])
        imp_scores = _sim(embeddings[g["imp_idx"]])
        bot_scores = _sim(bot_embeddings[g["bot_idx"]])

        y_true_list.append(np.ones(len(gen_scores), dtype=np.int64))
        y_score_list.append(gen_scores)

        y_true_list.append(np.zeros(len(imp_scores) + len(bot_scores), dtype=np.int64))
        y_score_list.append(np.concatenate([imp_scores, bot_scores]))

        for idx_sample, sc in zip(g["gen_idx"], gen_scores):
            detailed_records.append(
                {
                    "type": "genuine",
                    "target_subject": g["subject"],
                    "enroll_session": g["enroll_session"],
                    "sample_idx": int(idx_sample),
                    "score": float(sc),
                    "label": 1,
                }
            )
        for idx_sample, sc in zip(g["imp_idx"], imp_scores):
            detailed_records.append(
                {
                    "type": "human_impostor",
                    "target_subject": g["subject"],
                    "enroll_session": g["enroll_session"],
                    "sample_idx": int(idx_sample),
                    "score": float(sc),
                    "label": 0,
                }
            )

    return np.concatenate(y_true_list), np.concatenate(y_score_list), detailed_records


def score_manhattan_trials(
    detector: ScaledManhattanDetector,
    split: SplitData,
    trials: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_list, y_score_list = [], []
    for g in trials["groups"]:
        tmpl = detector.enroll(split.raw_flat[g["enroll_idx"]])
        gen_scores = detector.score(split.raw_flat[g["gen_idx"]], tmpl)
        imp_scores = detector.score(split.raw_flat[g["imp_idx"]], tmpl)
        bot_scores = detector.score(trials["bot_flat"][g["bot_idx"]], tmpl)

        y_true_list.append(np.ones(len(gen_scores), dtype=np.int64))
        y_score_list.append(gen_scores)
        y_true_list.append(np.zeros(len(imp_scores) + len(bot_scores), dtype=np.int64))
        y_score_list.append(np.concatenate([imp_scores, bot_scores]))

    return np.concatenate(y_true_list), np.concatenate(y_score_list)


def evaluate_1toN_identification_embeddings(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    sessions: np.ndarray,
    reps_or_chunks: np.ndarray,
    subjects: List[str],
    k_shot: int = 5,
    cross_session: bool = True,
    cohort: Optional[np.ndarray] = None,
    top_m: int = 12,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Task 2: 1:N Multi-User Typing Identification ("Who Is Typing?")
    - Each of the N unseen subjects registers `k_shot` typing strokes in Session 1.
    - All remaining strokes (strictly Session >= 2 if cross_session=True, or held-out chunks)
      are queried blindly against all N gallery templates.
    - Returns Top-1 Accuracy, Top-3 Accuracy, N-way Macro F1, and (N, N) confusion matrix.
    """
    n_classes = len(subjects)
    gallery_centroids = []
    gallery_exemplars = []
    gallery_coh_stats = []

    for uid in range(n_classes):
        e_idx = np.where((subject_ids == uid) & (sessions == 1) & (reps_or_chunks < k_shot + (1 if reps_or_chunks.min() == 1 else 0)))[0][:k_shot]
        e_vecs = embeddings[e_idx]
        c = e_vecs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        gallery_centroids.append(c)
        gallery_exemplars.append(e_vecs)
        if cohort is not None:
            k_top = min(top_m, len(cohort))
            c_coh = np.sort(cohort @ c)[-k_top:]
            gallery_coh_stats.append((float(c_coh.mean()), float(c_coh.std() + 0.05)))

    if cross_session:
        q_idx = np.where(sessions != 1)[0]
    else:
        min_idx = int(reps_or_chunks.min())
        q_idx = np.where((sessions == 1) & (reps_or_chunks >= min_idx + k_shot))[0]

    q_vecs = embeddings[q_idx]
    y_true = subject_ids[q_idx]

    if cohort is not None:
        k_top = min(top_m, len(cohort))
        q_coh = np.sort(q_vecs @ cohort.T, axis=1)[:, -k_top:]
        mu_q = q_coh.mean(axis=1)
        sd_q = q_coh.std(axis=1) + 0.05

    score_mat = np.zeros((len(q_idx), n_classes), dtype=np.float32)
    for uid in range(n_classes):
        c = gallery_centroids[uid]
        ex = gallery_exemplars[uid]
        raw = 0.65 * (q_vecs @ c) + 0.35 * np.max(q_vecs @ ex.T, axis=1)
        if cohort is not None:
            mu_u, sd_u = gallery_coh_stats[uid]
            z_as = (raw - 0.5 * (mu_u + mu_q)) / (0.5 * (sd_u + sd_q))
            score_mat[:, uid] = 1.0 / (1.0 + np.exp(-0.85 * (z_as - 0.15)))
        else:
            score_mat[:, uid] = raw

    y_pred = np.argmax(score_mat, axis=1)
    top3 = np.argsort(score_mat, axis=1)[:, -min(3, n_classes) :]
    top3_acc = float(np.mean([yt in top3[i] for i, yt in enumerate(y_true)]))
    top1_acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "num_classes": int(n_classes),
        "num_queries": int(len(q_idx)),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
    }, cm


def evaluate_1toN_identification_manhattan(
    detector: ScaledManhattanDetector,
    raw_flat: np.ndarray,
    subject_ids: np.ndarray,
    sessions: np.ndarray,
    reps_or_chunks: np.ndarray,
    subjects: List[str],
    k_shot: int = 5,
    cross_session: bool = True,
) -> Dict[str, float]:
    n_classes = len(subjects)
    templates = []
    min_idx = int(reps_or_chunks.min())
    for uid in range(n_classes):
        e_idx = np.where((subject_ids == uid) & (sessions == 1) & (reps_or_chunks < min_idx + k_shot))[0][:k_shot]
        templates.append(detector.enroll(raw_flat[e_idx]))

    q_idx = np.where(sessions != 1)[0] if cross_session else np.where((sessions == 1) & (reps_or_chunks >= min_idx + k_shot))[0]
    q_flat = raw_flat[q_idx]
    y_true = subject_ids[q_idx]

    score_mat = np.stack([detector.score(q_flat, tmpl) for tmpl in templates], axis=1)
    y_pred = np.argmax(score_mat, axis=1)
    top3 = np.argsort(score_mat, axis=1)[:, -min(3, n_classes) :]
    top3_acc = float(np.mean([yt in top3[i] for i, yt in enumerate(y_true)]))
    return {
        "num_classes": int(n_classes),
        "num_queries": int(len(q_idx)),
        "top1_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def summarize_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, float]:
    eer, eer_thresh, fpr, fnr, _ = compute_eer(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(np.int64)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    pos_f1 = float(f1_score(y_true, y_pred, pos_label=1))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = float(fp / max(1, fp + tn))
    frr = float(fn / max(1, fn + tp))

    valid_mask = fnr <= 0.05
    far_at_frr5 = float(np.min(fpr[valid_mask])) if np.any(valid_mask) else 1.0

    return {
        "macro_f1": round(macro_f1, 4),
        "genuine_f1": round(pos_f1, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "eer": round(eer, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "far_at_frr5": round(far_at_frr5, 4),
        "threshold": round(float(threshold), 4),
        "eer_threshold": round(float(eer_thresh), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_optimal_val_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Calibrate decision threshold strictly on Validation split (maximizing Macro F1)."""
    candidates = np.linspace(np.percentile(s_val, 5), np.percentile(s_val, 95), 300)
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        f1 = f1_score(y_val, (s_val >= t).astype(np.int64), average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def train_vanilla_baseline(
    train_split: SplitData,
    val_split: SplitData,
    val_trials: Dict,
    device: torch.device,
    epochs: int = 45,
    batch_size: int = 256,
    sentence_train: Optional[SentenceSplitData] = None,
    kr_fixed_train: Optional[Tuple] = None,
) -> Tuple[Vanilla1DCNN, List[Dict]]:
    """Train Scratch Vanilla1DCNN with standard Softmax Cross-Entropy across multi-length pools."""
    set_all_seeds(SEED)
    n_cmu = len(train_split.subjects)
    n_kr_fixed = (int(kr_fixed_train[4].max()) + 1) if kr_fixed_train is not None else 0
    n_kr = len(sentence_train.subjects) if sentence_train is not None else 0
    num_classes = n_cmu + n_kr_fixed + n_kr

    model = Vanilla1DCNN(in_channels=train_split.seq_scaled.shape[1], embed_dim=64).to(device)
    clf_head = nn.Linear(64, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(clf_head.parameters()),
        lr=1.0e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pools = [(train_split.seq_scaled, train_split.subject_ids)]
    if kr_fixed_train is not None:
        pools.append((kr_fixed_train[3], kr_fixed_train[4] + n_cmu))
    if sentence_train is not None:
        pools.append((sentence_train.seq_scaled, sentence_train.subject_ids + n_cmu + n_kr_fixed))

    rng = np.random.RandomState(SEED)
    history: List[Dict] = []
    best_state = None
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        clf_head.train()
        batches = []
        for x_p, y_p in pools:
            perm = rng.permutation(len(y_p))
            for i in range(0, len(y_p), batch_size):
                batches.append((x_p[perm[i : i + batch_size]], y_p[perm[i : i + batch_size]]))
        rng.shuffle(batches)

        total_loss = 0.0
        for xb_np, yb_np in batches:
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(yb_np).to(device)
            optimizer.zero_grad()
            emb = model(xb)
            logits = clf_head(emb * 10.0)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        val_emb = encode_sequences(model, val_split.seq_scaled, device)
        bot_emb = encode_sequences(model, val_trials["bot_seq"], device)
        yv, sv, _ = score_embedding_trials(val_emb, bot_emb, val_trials)
        eer, _, _, _, _ = compute_eer(yv, sv)
        opt_t = find_optimal_val_threshold(yv, sv)
        val_f1 = float(f1_score(yv, (sv >= opt_t).astype(np.int64), average="macro"))
        val_auc = float(roc_auc_score(yv, sv))

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, len(batches)), 4),
            "val_eer": round(eer, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_roc_auc": round(val_auc, 4),
        }
        history.append(row)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch in (1, 5, 10, 20, 30, 40, epochs):
            print(
                f"[Vanilla1DCNN] Epoch {epoch:02d}/{epochs} | "
                f"Loss={row['train_loss']:.4f} | Val F1={val_f1:.4f} | "
                f"Val EER={eer*100:.2f}% | Val AUC={val_auc:.4f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def train_keyprint_net(
    train_split: SplitData,
    val_split: SplitData,
    val_trials: Dict,
    bounds: Dict[str, Tuple[float, float]],
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
    device: torch.device,
    epochs: int = 45,
    batch_size: int = 256,
    use_augmentation: bool = True,
    raw_channels_only: bool = False,
    sentence_train: Optional[SentenceSplitData] = None,
    kr_fixed_train: Optional[Tuple] = None,
    sentence_val: Optional[SentenceSplitData] = None,
    tag: str = "KeyPrintNet",
    in_channels: int = N_CHANNELS,
    touch_prob: float = TOUCH_PROB,
    touch_hold: float = 0.1,
) -> Tuple[KeyPrintNet, List[Dict]]:
    """
    Train Proposed KeyPrintNet across multi-corpus / multi-length typing sequences
    (CMU L=10 passwords + KeyRecs L=9 fixed-text + KeyRecs L=24 practice sentences)
    with Smoothly Ramped Sub-Center CosFace + Supervised Contrastive Loss.
    """
    set_all_seeds(SEED)
    n_cmu = len(train_split.subjects)
    n_kr_fixed = (int(kr_fixed_train[4].max()) + 1) if kr_fixed_train is not None else 0
    n_kr = len(sentence_train.subjects) if sentence_train is not None else 0
    num_classes = n_cmu + n_kr_fixed + n_kr

    in_ch = 4 if raw_channels_only else in_channels
    if in_ch < N_CHANNELS:
        touch_prob = 0.0  # legacy / ablation inputs carry no Hold_Valid flag
    model = KeyPrintNet(in_channels=in_ch, hidden_channels=80, embed_dim=64, dropout=0.18).to(device)
    cosface_head = SubCenterCosFaceHead(
        embed_dim=64, num_classes=num_classes, sub_centers=2, scale=16.0, max_margin=0.08
    ).to(device)
    supcon_fn = SupervisedContrastiveLoss(temperature=0.15)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cosface_head.parameters()),
        lr=8.5e-4,
        weight_decay=3e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    _, hold_cols, dd_cols, ud_cols = get_feature_columns()
    pools = [
        {
            "holds": train_split.df[hold_cols].values.astype(np.float32),
            "dds": train_split.df[dd_cols].values.astype(np.float32),
            "uds": train_split.df[ud_cols].values.astype(np.float32),
            "x_clean": train_split.seq_scaled[:, :in_ch, :],
            "kb": train_split.key_block,
            "y": train_split.subject_ids,
        }
    ]
    if kr_fixed_train is not None:
        pools.append(
            {
                "holds": kr_fixed_train[0],
                "dds": kr_fixed_train[1],
                "uds": kr_fixed_train[2],
                "x_clean": kr_fixed_train[3][:, :in_ch, :],
                "kb": KEYRECS_FIXED_KEY_BLOCK,
                "y": kr_fixed_train[4] + n_cmu,
            }
        )
    if sentence_train is not None:
        pools.append(
            {
                "holds": sentence_train.holds,
                "dds": sentence_train.dds,
                "uds": sentence_train.uds,
                "x_clean": sentence_train.seq_scaled[:, :in_ch, :],
                "kb": sentence_train.key_blocks,
                "y": sentence_train.subject_ids + n_cmu + n_kr_fixed,
            }
        )

    rng = np.random.RandomState(SEED)
    history: List[Dict] = []
    best_state = None
    best_val_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        cosface_head.train()
        ramp = min(1.0, (epoch - 1) / 20.0)

        batch_specs = []
        for p_idx, pool in enumerate(pools):
            perm = rng.permutation(len(pool["y"]))
            for i in range(0, len(pool["y"]), batch_size):
                batch_specs.append((p_idx, perm[i : i + batch_size]))
        rng.shuffle(batch_specs)

        total_loss = 0.0
        for p_idx, idx in batch_specs:
            pool = pools[p_idx]
            kb = pool["kb"] if pool["kb"].ndim == 2 else pool["kb"][idx]
            warp = use_augmentation and rng.rand() < 0.45
            if warp or touch_prob > 0:
                xb_np = augment_keystroke_batch(
                    pool["holds"][idx],
                    pool["dds"][idx],
                    pool["uds"][idx],
                    bounds,
                    chan_mean,
                    chan_std,
                    rng,
                    key_block=kb,
                    warp=warp,
                    touch_prob=touch_prob,
                    touch_hold=touch_hold,
                )[:, :in_ch, :]
            else:
                xb_np = pool["x_clean"][idx]

            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(pool["y"][idx]).to(device)

            optimizer.zero_grad()
            emb = model(xb)
            loss_cls = cosface_head(emb, yb, ramp_ratio=ramp)
            loss_con = supcon_fn(emb, yb)
            loss = 0.88 * loss_cls + 0.12 * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            total_loss += float(loss.item())

        scheduler.step()

        tr_emb_cmu = encode_sequences(model, pools[0]["x_clean"], device)
        cohort = compute_cohort_centroids(tr_emb_cmu, pools[0]["y"])
        val_emb = encode_sequences(model, val_split.seq_scaled[:, :in_ch, :], device)
        bot_emb_v = encode_sequences(model, val_trials["bot_seq"][:, :in_ch, :], device)
        yv, sv, _ = score_embedding_trials(val_emb, bot_emb_v, val_trials, cohort=cohort)
        eer, _, _, _, _ = compute_eer(yv, sv)
        opt_t = find_optimal_val_threshold(yv, sv)
        val_f1 = float(f1_score(yv, (sv >= opt_t).astype(np.int64), average="macro"))
        val_auc = float(roc_auc_score(yv, sv))

        sent_val_acc = 0.0
        if sentence_val is not None:
            sve = encode_sequences(model, sentence_val.seq_scaled[:, :in_ch, :], device)
            id_m, _ = evaluate_1toN_identification_embeddings(
                sve,
                sentence_val.subject_ids,
                sentence_val.sessions,
                sentence_val.chunk_ids,
                sentence_val.subjects,
                k_shot=5,
                cross_session=False,
            )
            sent_val_acc = id_m["top1_accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, len(batch_specs)), 4),
            "val_eer": round(eer, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_roc_auc": round(val_auc, 4),
            "val_sentence_top1": round(sent_val_acc, 4),
        }
        history.append(row)
        combo_score = 0.45 * val_f1 + 0.45 * val_auc + (0.10 * sent_val_acc if sentence_val is not None else 0.0)
        if combo_score > best_val_score:
            best_val_score = combo_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch in (1, 5, 10, 20, 30, epochs):
            extra = f" | Sent-1:N={sent_val_acc*100:.1f}%" if sentence_val is not None else ""
            print(
                f"[{tag}] Epoch {epoch:02d}/{epochs} | "
                f"Loss={row['train_loss']:.4f} | Val F1={val_f1:.4f} | "
                f"Val EER={eer*100:.2f}% | Val AUC={val_auc:.4f}{extra}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history
