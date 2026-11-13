"""
Master execution script for KeyPrint Guard & Arena:
1. Runs 4-stage preprocessing across CMU StrongPassword + KeyRecs Sentence Transcription corpora
2. Trains Statistical Baseline, Scratch Vanilla1DCNN, and Proposed KeyPrintNet (CUDA L4 / MPS / CPU)
3. Evaluates BOTH:
   - Task 1 (1:1 Open-Set Password Intrusion Blocking on Unseen Subjects)
   - Task 2 (1:N Multi-User Typing-Practice Identification "Who Is Typing?" on Unseen Subjects)
4. Runs Session-1 Speed Drift Ablation + Sentence-Length Identification Sensitivity
5. Generates 7 publication-grade Korean analytical figures + exports ONNX & Web Arena bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, silhouette_score

from behavior_profile import fit_behavior_model, load_keyrecs_sessions, merge, to_events
from capture_sim import (
    as_norm_similarity,
    build_input_tensor,
    export_parity_fixture,
    extract_naive_unrepaired_stroke,
    fuse,
    make_template,
    ms_rounded,
)
from feature_eval import (
    attach_tensors,
    build_capture_bank,
    calibrate_open_set_threshold,
    capture_stats,
    cmu_behavior_stats,
    dd_cv,
    embed_bank,
    encode_variable,
    eval_password_transfer,
    eval_web_identification,
    fixed_stroke_capture,
    flat_to_hdu,
    open_set_eval,
    open_set_matrix,
    score_trials,
    sentence_add_check,
    stroke_payload,
    template_payload,
    tune_beta_identification,
    tune_beta_password,
    tune_open_set_rule,
)
from keyboard_layout import CMU_PASSWORD_CODES, export_layout_for_web, pairs_from_codes
from models import KeyPrintNet, ScaledManhattanDetector, Vanilla1DCNN
from password_synth import (
    CALIB_LENGTHS,
    DISPLAY_SLOPE,
    adaptation_eval,
    calibrate_by_length,
    calibrate_dual,
    capital_fusion,
    capital_llr,
    capital_method_rates,
    capital_trials,
    choose_strings,
    digraph_tables,
    evaluate_lengths,
    evaluate_strings,
    params_for_length,
    person_speed,
    string_trials,
    strings_of_length,
    synthesis_metadata,
)
from preprocessing import (
    CHANNEL_NAMES,
    DIGRAPH_LABELS,
    KEYRECS_DD_COLS,
    KEYRECS_HOLD_COLS,
    N_CHANNELS,
    N_TIMING_CHANNELS,
    SEED,
    TOUCH_PROB,
    SentenceSplitData,
    apply_channel_scaler,
    extract_sequences,
    get_feature_columns,
    prepare_full_pipeline,
    touchify_timings,
)
from train_and_evaluate import (
    KEYPRINT_DEFAULT_EPOCHS,
    VANILLA_DEFAULT_EPOCHS,
    build_verification_trials,
    compute_cohort_centroids,
    compute_eer,
    configure_korean_matplotlib,
    encode_sequences,
    evaluate_1toN_identification_embeddings,
    evaluate_1toN_identification_manhattan,
    find_optimal_val_threshold,
    get_device,
    impostor_breakdown_embedding,
    impostor_breakdown_manhattan,
    score_embedding_trials,
    score_manhattan_trials,
    set_all_seeds,
    summarize_metrics,
    train_keyprint_net,
    train_vanilla_baseline,
)

MODEL_VERSION = "kp17-touch-behavior-v4"
SCHEMA_VERSION = 4
PREPROCESS_VERSION = "kp-preproc-v4"
SAMPLE_PROFILE_TIMESTAMP = 1700000000000
WEB_CAPTURE_CHARS = 68  # average length of the web app's practice sentences
ARENA_ENROLL_SENTENCES = 5  # typing practice: sentences per profile
ARENA_QUERY_SENTENCES = 2  # typing practice: sentences typed before "who is typing?" is answered
PASSWORD_FRR_TARGET = 0.10  # dual password score: tau = Validation genuine quantile
PASSWORD_MIN_LENGTH = 12  # new accounts: longer passwords carry more rhythm evidence (README section 5.7)
REUSE_WEIGHTS = os.environ.get("KEYPRINT_REUSE_WEIGHTS") == "1"
_SMOKE_EPOCHS = os.environ.get("KEYPRINT_SMOKE_EPOCHS")
VANILLA_EPOCHS = int(_SMOKE_EPOCHS) if _SMOKE_EPOCHS else VANILLA_DEFAULT_EPOCHS
KEYPRINT_EPOCHS = int(_SMOKE_EPOCHS) if _SMOKE_EPOCHS else KEYPRINT_DEFAULT_EPOCHS


def safe_export_onnx(model: KeyPrintNet, onnx_path: Path, web_onnx_path: Path, in_channels: int = N_CHANNELS) -> Dict:
    cpu_model = KeyPrintNet(in_channels=in_channels, hidden_channels=80, embed_dim=64, dropout=0.0)
    cpu_model.load_state_dict(model.cpu().state_dict())
    cpu_model.eval()

    dummy = torch.randn(1, in_channels, 10, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    web_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["keystroke_seq"],
        output_names=["biometric_embedding"],
        dynamic_axes={
            "keystroke_seq": {0: "batch", 2: "seq_len"},
            "biometric_embedding": {0: "batch"},
        },
    )
    try:
        torch.onnx.export(cpu_model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(cpu_model, dummy, str(onnx_path), **export_kwargs)

    loaded = onnx.load(str(onnx_path))
    onnx.checker.check_model(loaded)
    onnx.save_model(loaded, str(onnx_path), save_as_external_data=False)
    shutil.copy2(onnx_path, web_onnx_path)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diffs = {}
    for seq_len in [5, 10, 18, 24, 32]:
        x_test = np.random.RandomState(seq_len).randn(2, in_channels, seq_len).astype(np.float32)
        with torch.no_grad():
            pt_out = cpu_model(torch.from_numpy(x_test)).numpy()
        ort_out = sess.run(["biometric_embedding"], {"keystroke_seq": x_test})[0]
        max_diffs[f"L={seq_len}"] = float(np.max(np.abs(pt_out - ort_out)))

    size_kb = round(onnx_path.stat().st_size / 1024.0, 2)
    print(f"[ONNX] Exported {onnx_path.name} ({size_kb} KB) | Parity={max_diffs}")
    return {"onnx_size_kb": size_kb, "max_abs_diff_by_length": max_diffs}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_git_head(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        out = res.stdout.strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def compute_troubleshooting_ablations(
    test_split,
    test_trials: Dict,
    bounds: Dict[str, Tuple[float, float]],
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
    touch_hold: float,
    legacy_model: KeyPrintNet,
    keyprint_model: KeyPrintNet,
    cohort_l: np.ndarray,
    sent_cohort: np.ndarray,
    behavior_model,
    beta_kb: float,
    test_bank: Dict[str, Dict[int, List[Dict]]],
    device: torch.device,
) -> Dict:
    """Reproducible quantitative evidence for README §9.2 Case 2 (tempo normalization) and
    Case 4 (Backspace stitch repair vs naive unrepaired keystroke capture)."""
    # 1) Tempo Normalization Ablation (Unseen CMU Test Subjects, 1.25x Speed Warp + Cross-Session)
    warped_raw_h = (test_split.raw_holds * 0.80).astype(np.float32)
    warped_raw_d = (test_split.raw_dds * 0.80).astype(np.float32)
    warped_raw_u = (warped_raw_d - warped_raw_h[:, :-1]).astype(np.float32)
    warped_c_h = np.clip(warped_raw_h, *bounds["H"]).astype(np.float32)
    warped_c_d = np.clip(warped_raw_d, *bounds["DD"]).astype(np.float32)
    warped_c_u = np.clip(warped_raw_u, *bounds["UD"]).astype(np.float32)
    warped_seq = apply_channel_scaler(
        extract_sequences(warped_c_h, warped_c_d, warped_c_u, test_split.key_block), chan_mean, chan_std
    )

    def _unit(mat: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return (mat / n).astype(np.float32)

    L9 = N_TIMING_CHANNELS
    reps_clean = {
        "raw_4ch_scaled": _unit(test_split.seq_scaled[:, :4, :].reshape(len(test_split.df), -1)),
        "relative_tempo_5ch_scaled": _unit(test_split.seq_scaled[:, 4:9, :].reshape(len(test_split.df), -1)),
        "full_bio_9ch_scaled": _unit(test_split.seq_scaled[:, :9, :].reshape(len(test_split.df), -1)),
        "keyprint_net_9ch_embedding": encode_sequences(legacy_model, test_split.seq_scaled[:, :L9, :], device),
    }
    reps_warped = {
        "raw_4ch_scaled": _unit(warped_seq[:, :4, :].reshape(len(test_split.df), -1)),
        "relative_tempo_5ch_scaled": _unit(warped_seq[:, 4:9, :].reshape(len(test_split.df), -1)),
        "full_bio_9ch_scaled": _unit(warped_seq[:, :9, :].reshape(len(test_split.df), -1)),
        "keyprint_net_9ch_embedding": encode_sequences(legacy_model, warped_seq[:, :L9, :], device),
    }
    bots = {
        "raw_4ch_scaled": _unit(test_trials["bot_seq"][:, :4, :].reshape(len(test_trials["bot_seq"]), -1)),
        "relative_tempo_5ch_scaled": _unit(test_trials["bot_seq"][:, 4:9, :].reshape(len(test_trials["bot_seq"]), -1)),
        "full_bio_9ch_scaled": _unit(test_trials["bot_seq"][:, :9, :].reshape(len(test_trials["bot_seq"]), -1)),
        "keyprint_net_9ch_embedding": encode_sequences(legacy_model, test_trials["bot_seq"][:, :L9, :], device),
    }

    tempo_summary: Dict[str, Dict[str, float]] = {}
    for rname in reps_clean:
        x_clean = reps_clean[rname]
        x_warp = reps_warped[rname]
        cross_cos, warp_cos = [], []
        for g in test_trials["groups"]:
            c = x_clean[g["enroll_idx"]].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-8)
            cross_cos.append(x_clean[g["gen_idx"]] @ c)
            warp_cos.append(x_warp[g["gen_idx"]] @ c)
        use_coh = cohort_l if rname == "keyprint_net_9ch_embedding" else None
        is_single = rname != "keyprint_net_9ch_embedding"
        yt, st, _ = score_embedding_trials(
            x_clean, bots[rname], test_trials, cohort=use_coh, single_centroid=is_single
        )
        eer, _, _, _, _ = compute_eer(yt, st)
        tempo_summary[rname] = {
            "genuine_cross_session_cosine_mean": round(float(np.mean(np.concatenate(cross_cos))), 4),
            "speed_warped_1_25x_cosine_mean": round(float(np.mean(np.concatenate(warp_cos))), 4),
            "verification_eer": round(float(eer), 4),
            "verification_roc_auc": round(float(roc_auc_score(yt, st)), 4),
        }

    # 2) Backspace Stitch Subset Ablation (Unseen KeyRecs Test 18-Way Keyboard 1:N Gallery, k_enroll=3)
    k_enroll = 3
    people = [p for p in sorted(test_bank) if len(test_bank[p][1]) > k_enroll and len(test_bank[p][2]) > 0]
    gallery = []
    for p in people:
        enroll = test_bank[p][1][:k_enroll]
        beh = None
        for c in enroll:
            beh = merge(beh, c["behavior"])
        gallery.append(make_template(np.stack([c["emb"] for c in enroll]), beh))

    def _score_query(emb_vec: np.ndarray, beh_dict: Dict, use_beh: bool) -> Tuple[int, bool]:
        scores = []
        for tmpl in gallery:
            s = as_norm_similarity(emb_vec, tmpl, sent_cohort, 16)
            if use_beh and behavior_model is not None and beta_kb > 0:
                b, _ = behavior_model.llr(beh_dict, tmpl["behavior"])
                s = fuse(s, b, beta_kb)
            scores.append(s)
        order = np.argsort(-np.asarray(scores))
        return int(order[0]), order[:3].tolist()

    query_meta = []
    naive_tensors = []
    naive_map: Dict[int, int] = {}
    for ti, p in enumerate(people):
        for s_tag, caps in (("same_session", test_bank[p][1][k_enroll:]), ("cross_session", test_bank[p][2])):
            for cap in caps:
                q_idx = len(query_meta)
                bs_cnt = int(cap.get("backspace_count", 0))
                query_meta.append((ti, s_tag, bs_cnt, cap))
                if bs_cnt > 0 and "_raw_segment" in cap:
                    ncap = extract_naive_unrepaired_stroke(cap["_raw_segment"], touch=False)
                    nt = build_input_tensor(ncap, bounds, touch_hold, chan_mean, chan_std)
                    if nt.shape[1] > 0:
                        naive_map[q_idx] = len(naive_tensors)
                        naive_tensors.append((nt, ncap["behavior"]))

    naive_embs = (
        encode_variable(keyprint_model, [x[0] for x in naive_tensors], device, N_CHANNELS)
        if naive_tensors
        else np.zeros((0, 64), dtype=np.float32)
    )

    def _summarize_subset(indices: List[int], use_naive: bool, use_beh: bool) -> Dict[str, float]:
        if not indices:
            return {"num_queries": 0, "top1_accuracy": 0.0, "top3_accuracy": 0.0}
        yt, yp, t3 = [], [], []
        for qi in indices:
            ti, _, _, cap = query_meta[qi]
            if use_naive:
                if qi not in naive_map:
                    continue
                ni = naive_map[qi]
                emb_v, beh_v = naive_embs[ni], naive_tensors[ni][1]
            else:
                emb_v, beh_v = cap["emb"], cap["behavior"]
            pred, top3_list = _score_query(emb_v, beh_v, use_beh)
            yt.append(ti)
            yp.append(pred)
            t3.append(ti in top3_list)
        if not yt:
            return {"num_queries": 0, "top1_accuracy": 0.0, "top3_accuracy": 0.0}
        return {
            "num_queries": int(len(yt)),
            "top1_accuracy": round(float(accuracy_score(yt, yp)), 4),
            "top3_accuracy": round(float(np.mean(t3)), 4),
        }

    backspace_summary: Dict[str, Dict] = {"num_gallery_subjects": len(people), "k_enroll": k_enroll}
    for scope_name, pred_fn in (
        ("same_session", lambda s_tag: s_tag == "same_session"),
        ("cross_session", lambda s_tag: s_tag == "cross_session"),
        ("pooled", lambda _: True),
    ):
        clean_idx = [i for i, (_, st, bs, _) in enumerate(query_meta) if pred_fn(st) and bs == 0]
        bs_idx = [i for i, (_, st, bs, _) in enumerate(query_meta) if pred_fn(st) and bs > 0]
        heavy_bs_idx = [i for i, (_, st, bs, _) in enumerate(query_meta) if pred_fn(st) and bs >= 3]
        backspace_summary[scope_name] = {
            "clean_captures": _summarize_subset(clean_idx, use_naive=False, use_beh=True),
            "backspace_repaired_captures": _summarize_subset(bs_idx, use_naive=False, use_beh=True),
            "naive_unrepaired_backspace_captures": _summarize_subset(bs_idx, use_naive=True, use_beh=True),
            "heavy_backspace_ge3_repaired": _summarize_subset(heavy_bs_idx, use_naive=False, use_beh=True),
            "heavy_backspace_ge3_naive": _summarize_subset(heavy_bs_idx, use_naive=True, use_beh=True),
            "encoder_only_17ch": {
                "clean_captures": _summarize_subset(clean_idx, use_naive=False, use_beh=False),
                "backspace_repaired_captures": _summarize_subset(bs_idx, use_naive=False, use_beh=False),
                "naive_unrepaired_backspace_captures": _summarize_subset(bs_idx, use_naive=True, use_beh=False),
            },
        }

    return {
        "tempo_normalization": tempo_summary,
        "backspace_stitch_subset": backspace_summary,
    }


def plot_fig1_preprocessing(
    raw_df: pd.DataFrame,
    splits: Dict,
    meta: Dict,
    sil_raw4: float,
    sil_bio9: float,
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))
    palette = ["#2563EB", "#10B981", "#F59E0B"]

    ax = axes[0]
    kr_m = meta.get("keyrecs_sentence_corpus")
    sc = meta["split_counts"]
    if kr_m:
        n_tr_p = sc["train_subjects"] + kr_m.get("train_subjects", kr_m.get("train_participants", 0))
        n_va_p = sc["val_subjects"] + kr_m.get("val_subjects", kr_m.get("val_participants", 0))
        n_te_p = sc["test_subjects"] + kr_m.get("test_subjects", kr_m.get("test_participants", 0))
        labels = [
            f"Train\n({n_tr_p}명 · {sc['train_samples'] + kr_m['train_sentence_chunks'] + kr_m['train_fixed_strokes']:,}건)",
            f"Validation\n({n_va_p}명 · {sc['val_samples'] + kr_m['val_sentence_chunks']:,}건)",
            f"Unseen Test\n({n_te_p}명 · {sc['test_samples'] + kr_m['test_sentence_chunks']:,}건)",
        ]
        counts = [
            sc["train_samples"] + kr_m["train_sentence_chunks"] + kr_m["train_fixed_strokes"],
            sc["val_samples"] + kr_m["val_sentence_chunks"],
            sc["test_samples"] + kr_m["test_sentence_chunks"],
        ]
    else:
        labels = ["Train\n(35명 · 68.6%)", "Validation\n(6명 · 11.8%)", "Unseen Test\n(10명 · 19.6%)"]
        counts = [sc["train_samples"], sc["val_samples"], sc["test_samples"]]

    bars = ax.bar(labels, counts, color=palette, width=0.56, edgecolor="#1E293B", linewidth=1.0)
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2,
            c + max(counts) * 0.02,
            f"{c:,}건",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
        )
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_ylabel("타건 시퀀스 수 (암호 + 영문 타자연습)")
    ax.set_title("(a) 피험자 완전 분리(Subject-Disjoint) 데이터 분할", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax = axes[1]
    col = "DD.i.e"
    raw_train_df = raw_df[raw_df["subject"].isin(splits["train"].subjects)]
    raw_vals = raw_train_df[col].values
    clip_vals = splits["train"].df[col].values
    assert len(raw_vals) == len(clip_vals), "fig1(b) before/after populations must match identically"
    ax.hist(
        np.clip(raw_vals, 0, 2.5),
        bins=45,
        alpha=0.55,
        color="#EF4444",
        label=f"정제 전 (Train n={len(raw_vals):,}, 최대 {raw_vals.max():.1f}초, 왜도 {pd.Series(raw_vals).skew():.1f})",
        density=True,
    )
    ax.hist(
        clip_vals,
        bins=45,
        alpha=0.68,
        color="#2563EB",
        label=f"정제 후 (Train n={len(clip_vals):,}, 상한 {clip_vals.max():.2f}초, 왜도 {pd.Series(clip_vals).skew():.2f})",
        density=True,
    )
    ax.set_xlabel("키 간 전환 지연 시간 DD.i.e (초)")
    ax.set_ylabel("밀도 (Density)")
    ax.set_title("(b) 동일 Train 모집단(n=14,000) 이상치 정제 전·후 비교", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.4)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[2]
    categories = ["Train 컬럼 평균 왜도\n(낮을수록 정규성 우수)", "피험자 군집 응집도×10\n(Silhouette, 높을수록 우수)"]
    before_metrics = [meta["preprocessing_before_after"]["before_mean_skewness"], max(0.05, sil_raw4 * 10)]
    after_metrics = [meta["preprocessing_before_after"]["after_mean_skewness"], sil_bio9 * 10]

    x = np.arange(len(categories))
    w = 0.34
    b1 = ax.bar(x - w / 2, before_metrics, width=w, color="#94A3B8", label="전처리 전 (원시 타이밍)", edgecolor="#1E293B")
    b2 = ax.bar(x + w / 2, after_metrics, width=w, color="#0EA5E9", label="전처리 후 (9채널 리듬 정규화)", edgecolor="#1E293B")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.25, f"{h:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(before_metrics + after_metrics) * 1.22)
    ax.set_title("(c) 9채널 템포 정규화 전·후 정량 지표 개선", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_preprocessing_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_learning_curves(
    hist_vanilla: List[Dict],
    hist_keyprint: List[Dict],
    fig_dir: Path,
    hist_keyprint_9ch: Optional[List[Dict]] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ev = [r["epoch"] for r in hist_vanilla]
    ek = [r["epoch"] for r in hist_keyprint]

    axes[0].plot(ev, [r["train_loss"] for r in hist_vanilla], "--", color="#64748B", lw=1.9, label="Vanilla1DCNN (CE Loss)")
    if hist_keyprint_9ch:
        e9 = [r["epoch"] for r in hist_keyprint_9ch]
        axes[0].plot(e9, [r["train_loss"] for r in hist_keyprint_9ch], "-.", color="#059669", lw=2.0, label="KeyPrintNet 9ch (Password)")
    axes[0].plot(ek, [r["train_loss"] for r in hist_keyprint], "-", color="#2563EB", lw=2.3, label="KeyPrintNet 17ch (Sentence)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("(a) 에포크별 학습 손실(Loss) 수렴 곡선", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8.6)
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(ev, [r["val_eer"] * 100 for r in hist_vanilla], "--o", markersize=2.8, color="#EF4444", lw=1.8, label="Vanilla1DCNN (Baseline)")
    if hist_keyprint_9ch:
        e9 = [r["epoch"] for r in hist_keyprint_9ch]
        axes[1].plot(e9, [r["val_eer"] * 100 for r in hist_keyprint_9ch], "-.^", markersize=2.8, color="#059669", lw=1.9, label="KeyPrintNet 9ch")
    axes[1].plot(ek, [r["val_eer"] * 100 for r in hist_keyprint], "-o", markersize=2.8, color="#2563EB", lw=2.2, label="KeyPrintNet 17ch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation EER (%) ↓ (낮을수록 우수)")
    axes[1].set_title("(b) 검증셋 동일오류율(Val EER) 변화", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8.6)
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(ev, [r["val_macro_f1"] for r in hist_vanilla], "--s", markersize=2.8, color="#F59E0B", lw=1.8, label="Vanilla1DCNN (Baseline)")
    if hist_keyprint_9ch:
        e9 = [r["epoch"] for r in hist_keyprint_9ch]
        axes[2].plot(e9, [r["val_macro_f1"] for r in hist_keyprint_9ch], "-.^", markersize=2.8, color="#059669", lw=1.9, label="KeyPrintNet 9ch")
    axes[2].plot(ek, [r["val_macro_f1"] for r in hist_keyprint], "-s", markersize=2.8, color="#2563EB", lw=2.2, label="KeyPrintNet 17ch")
    axes[2].axhline(0.85, color="#DC2626", linestyle=":", lw=1.5, label="정량 목표 (Macro F1 ≥ 0.85)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation Macro F1 ↑ (높을수록 우수)")
    axes[2].set_title("(c) 검증셋 Macro F1-Score 향상 추이", fontsize=12, fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=8.4)
    axes[2].grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_roc_det_kshot(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]],
    kshot_results: List[Dict],
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))
    colors = {
        "ScaledManhattan (통계 베이스라인)": "#64748B",
        "Vanilla1DCNN (딥러닝 베이스라인)": "#F59E0B",
        "KeyPrintNet (제안 모델)": "#2563EB",
    }

    ax = axes[0]
    for name, (yt, ys, m) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, ys)
        ax.plot(
            fpr,
            tpr,
            lw=2.4 if "KeyPrintNet" in name else 1.8,
            color=colors[name],
            label=f"{name}\n(AUC={m['roc_auc']:.4f}, EER={m['eer']*100:.2f}%)",
        )
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("해커 오허용률 (FAR / False Positive Rate)")
    ax.set_ylabel("본인 정인식률 (TPR / True Positive Rate)")
    ax.set_title("(a) 1:1 해커 차단 Unseen Test ROC 곡선 비교", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.6)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[1]
    yt_kp, ys_kp, m_kp = list(roc_data.values())[2]
    fpr, tpr, thresh = roc_curve(yt_kp, ys_kp)
    fnr = 1.0 - tpr
    valid = np.isfinite(thresh)
    ax.plot(thresh[valid], fpr[valid] * 100, color="#EF4444", lw=2.2, label="FAR (해커 오허용률 %)")
    ax.plot(thresh[valid], fnr[valid] * 100, color="#2563EB", lw=2.2, label="FRR (본인 오차단율 %)")
    ax.axvline(m_kp["threshold"], color="#10B981", linestyle="--", lw=1.8, label=f"운용 임계값 τ={m_kp['threshold']:.3f}")
    ax.scatter([m_kp["eer_threshold"]], [m_kp["eer"] * 100], color="#111827", zorder=5, s=55, label=f"EER 교차점 ({m_kp['eer']*100:.2f}%)")
    x_lo = max(0.0, min(float(np.percentile(ys_kp, 2)), m_kp["threshold"], m_kp["eer_threshold"]) - 0.06)
    x_hi = min(1.0, max(float(np.percentile(ys_kp, 98)), m_kp["threshold"], m_kp["eer_threshold"]) + 0.06)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 45)
    ax.set_xlabel("교정된 생체 유사도 임계값 (Calibrated Score τ)")
    ax.set_ylabel("오류율 (%)")
    ax.set_title("(b) KeyPrintNet 임계값별 FAR·FRR 트레이드오프", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", fontsize=8.6)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[2]
    ks = [r["k_shot"] for r in kshot_results]
    f1s = [r["macro_f1"] for r in kshot_results]
    eers = [r["eer"] * 100 for r in kshot_results]
    ax2 = ax.twinx()
    l1 = ax.plot(ks, f1s, "-o", color="#2563EB", lw=2.3, markersize=7, label="Test Macro F1 ↑")
    l2 = ax2.plot(ks, eers, "--s", color="#EF4444", lw=2.1, markersize=6, label="Test EER (%) ↓")
    ax.axvline(5, color="#10B981", linestyle=":", lw=1.6)
    ax.text(5.15, min(f1s) + 0.005, "웹 기본 채택\n(K=5회 등록)", color="#047857", fontsize=9, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xlabel("초기 타건 지문 등록 횟수 (K-Shot Enrollment)")
    ax.set_ylabel("Macro F1-Score", color="#2563EB")
    ax2.set_ylabel("EER (%)", color="#EF4444")
    ax.set_title("(c) 등록 횟수(K)에 따른 인증 정확도 민감도", fontsize=12, fontweight="bold")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_roc_det_kshot.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_confusion_matrices(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]], fig_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
    cmaps = ["Greys", "Oranges", "Blues"]

    for ax, (name, (yt, ys, m)), cmap in zip(axes, roc_data.items(), cmaps):
        yp = (ys >= m["threshold"]).astype(np.int64)
        cm = confusion_matrix(yt, yp)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
        annot = np.array(
            [[f"{cm[i, j]:,}건\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
        )
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap=cmap,
            cbar=False,
            xticklabels=["침입 차단\n(Impostor 예측)", "접속 승인\n(Genuine 예측)"],
            yticklabels=["실제 침입자\n(해커·봇)", "실제 본인\n(정당 사용자)"],
            ax=ax,
            vmin=0,
            vmax=100,
            annot_kws={"fontsize": 10.5, "fontweight": "bold"},
        )
        ax.set_title(f"{name}\nMacro F1={m['macro_f1']:.4f} | Acc={m['accuracy']*100:.1f}%", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig5_tsne_embeddings(
    test_split,
    vanilla_emb: np.ndarray,
    keyprint_emb: np.ndarray,
    bot_vanilla_emb: np.ndarray,
    bot_keyprint_emb: np.ndarray,
    fig_dir: Path,
) -> None:
    rng = np.random.RandomState(SEED)
    sub_indices = []
    sub_labels = []
    for uid, sname in enumerate(test_split.subjects):
        idx = rng.choice(np.where(test_split.subject_ids == uid)[0], size=45, replace=False)
        sub_indices.append(idx)
        sub_labels.extend([f"미학습 사용자 {sname}"] * len(idx))
    sub_idx = np.concatenate(sub_indices)
    bot_idx = rng.choice(len(bot_keyprint_emb), size=50, replace=False)

    all_labels = sub_labels + ["매크로 봇 공격"] * len(bot_idx)
    v_all = np.concatenate([vanilla_emb[sub_idx], bot_vanilla_emb[bot_idx]], axis=0)
    k_all = np.concatenate([keyprint_emb[sub_idx], bot_keyprint_emb[bot_idx]], axis=0)

    sil_v = float(silhouette_score(vanilla_emb[sub_idx], test_split.subject_ids[sub_idx], metric="cosine"))
    sil_k = float(silhouette_score(keyprint_emb[sub_idx], test_split.subject_ids[sub_idx], metric="cosine"))

    tsne_v = TSNE(n_components=2, perplexity=28, random_state=SEED, init="pca", learning_rate="auto").fit_transform(v_all)
    tsne_k = TSNE(n_components=2, perplexity=28, random_state=SEED, init="pca", learning_rate="auto").fit_transform(k_all)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    palette = sns.color_palette("tab10", 10)

    for ax, coords, title in [
        (axes[0], tsne_v, f"(a) Vanilla1DCNN 임베딩 공간 (Unseen Silhouette = {sil_v:.3f})"),
        (axes[1], tsne_k, f"(b) KeyPrintNet 임베딩 공간 (Unseen Silhouette = {sil_k:.3f})"),
    ]:
        for uid, sname in enumerate(test_split.subjects):
            mask = np.array(all_labels) == f"미학습 사용자 {sname}"
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=28,
                alpha=0.80,
                color=palette[uid],
                label=sname,
            )
        b_mask = np.array(all_labels) == "매크로 봇 공격"
        ax.scatter(
            coords[b_mask, 0],
            coords[b_mask, 1],
            s=52,
            marker="X",
            color="#111827",
            alpha=0.9,
            label="매크로 봇 공격",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(alpha=0.2, linestyle="--")

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.8, title="Unseen Test 클래스")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_embedding_tsne.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig6_error_analysis(
    test_split,
    detailed_records: List[Dict],
    threshold: float,
    ablation_summary: Dict,
    fig_dir: Path,
) -> Dict:
    fn_cases = [r for r in detailed_records if r["label"] == 1 and r["score"] < threshold]
    fp_cases = [r for r in detailed_records if r["label"] == 0 and r["score"] >= threshold]

    fn_cases.sort(key=lambda r: r["score"])
    fp_cases.sort(key=lambda r: -r["score"])

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9))
    x_idx = np.arange(len(DIGRAPH_LABELS))

    ax = axes[0]
    if fn_cases:
        worst_fn = fn_cases[0]
        target_subj = worst_fn["target_subject"]
        uid = test_split.subjects.index(target_subj)
        enroll_mask = (
            (test_split.subject_ids == uid)
            & (test_split.sessions == worst_fn["enroll_session"])
            & (test_split.reps <= 5)
        )
        tmpl_dd = test_split.seq_unscaled[enroll_mask, 2, :].mean(axis=0) * 1000.0
        err_dd = test_split.seq_unscaled[worst_fn["sample_idx"], 2, :] * 1000.0
        err_sess = int(test_split.sessions[worst_fn["sample_idx"]])

        ax.plot(x_idx, tmpl_dd, "-o", color="#2563EB", lw=2.2, label=f"{target_subj} 등록 템플릿 (Session {worst_fn['enroll_session']})")
        ax.plot(x_idx, err_dd, "--s", color="#EF4444", lw=2.2, label=f"오차단 샘플 (Session {err_sess}, 유사도={worst_fn['score']:.3f})")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(DIGRAPH_LABELS, rotation=25)
        ax.set_ylabel("키 간 눌림 간격 DD (ms)")
        ax.set_title("(a) 오류 사례 1: 본인 오차단(FRR) 급격한 리듬 일탈", fontsize=11.5, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, linestyle="--")

    ax = axes[1]
    if fp_cases:
        worst_fp = fp_cases[0]
        target_subj = worst_fp["target_subject"]
        uid = test_split.subjects.index(target_subj)
        enroll_mask = (
            (test_split.subject_ids == uid)
            & (test_split.sessions == worst_fp["enroll_session"])
            & (test_split.reps <= 5)
        )
        tmpl_rel = test_split.seq_unscaled[enroll_mask, 5, :].mean(axis=0)
        imp_rel = test_split.seq_unscaled[worst_fp["sample_idx"], 5, :]
        imp_subj = test_split.subjects[int(test_split.subject_ids[worst_fp["sample_idx"]])]

        ax.plot(x_idx, tmpl_rel, "-o", color="#10B981", lw=2.2, label=f"본인 템플릿 ({target_subj})")
        ax.plot(x_idx, imp_rel, "--^", color="#F59E0B", lw=2.2, label=f"유사 타건 침입자 ({imp_subj}, 유사도={worst_fp['score']:.3f})")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(DIGRAPH_LABELS, rotation=25)
        ax.set_ylabel("템포 정규화 상대 리듬 비율 (Rel_DD)")
        ax.set_title("(b) 오류 사례 2: 유사 근육 리듬(Wolf/Lamb) 오허용", fontsize=11.5, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, linestyle="--")

    ax = axes[2]
    labels = [
        "시도 1: 절대 타이밍 거리\n(ScaledManhattan)",
        "시도 2: 위치 비인지 CNN\n(Vanilla1DCNN)",
        "제안: KeyPrintNet 17채널\n(문장용·템포워핑·AS-Norm)",
        "제안: KeyPrintNet 9채널\n(비밀번호 배포·타이밍 전용)",
    ]
    steps = ["step1_raw_manhattan", "step2_vanilla_cnn", "step3_keyprint_proposed", "step4_keyprint_9ch_password_encoder"]
    f1_vals = [ablation_summary[k]["macro_f1"] for k in steps]
    eer_vals = [ablation_summary[k]["eer"] * 100 for k in steps]
    x_ab = np.arange(len(labels))
    bars = ax.bar(x_ab, f1_vals, width=0.55, color=["#94A3B8", "#38BDF8", "#2563EB", "#059669"], edgecolor="#1E293B")
    for b, f1v, eerv in zip(bars, f1_vals, eer_vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            f1v + 0.012,
            f"F1 {f1v:.3f}\n(EER {eerv:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
        )
    ax.set_xticks(x_ab)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylim(0.50, 1.00)
    ax.set_ylabel("세션 간 드리프트 검증 Macro F1 (Session 1 → 2~8)")
    ax.set_title("(c) 문제 해결 검증: 숙련도 속도 드리프트 극복 실험", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_error_analysis_and_ablation.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "num_false_rejections": len(fn_cases),
        "num_false_acceptances": len(fp_cases),
        "worst_fn": fn_cases[0] if fn_cases else None,
        "worst_fp": fp_cases[0] if fp_cases else None,
    }


def plot_fig7_typist_identification(
    subjects_10: List[str],
    cm_10way: np.ndarray,
    id_comparison: Dict[str, Dict[str, float]],
    sent_len_curve: List[Dict],
    fig_dir: Path,
) -> None:
    """
    Plot 3-panel evaluation figure for Task 2:
    1:N Multi-User Typing-Practice Automatic Identification ("Who Is Typing?").
    """
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2))

    # Panel A: 10x10 Unseen Subject Identification Confusion Matrix
    ax = axes[0]
    cm_pct = cm_10way / np.maximum(1, cm_10way.sum(axis=1, keepdims=True)) * 100.0
    sns.heatmap(
        cm_pct,
        annot=np.round(cm_pct).astype(int),
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=subjects_10,
        yticklabels=subjects_10,
        ax=ax,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 8.0, "fontweight": "bold"},
    )
    ax.set_xlabel("예측된 타이핑 인물 (Predicted Typist)")
    ax.set_ylabel("실제 타이핑 인물 (True Unseen Typist)")
    kp_top1 = id_comparison["KeyPrintNet"]["top1_accuracy"] * 100
    kp_top3 = id_comparison["KeyPrintNet"]["top3_accuracy"] * 100
    ax.set_title(
        f"(a) 1:N 동일 문구 10인 자동 분류 혼동 행렬 (%)\n(Top-1={kp_top1:.1f}%, Top-3={kp_top3:.1f}%, 무작위=10%)",
        fontsize=11.3,
        fontweight="bold",
    )

    # Panel B: Model Comparison on 1:N Unseen Typist Identification
    ax = axes[1]
    model_names = ["ScaledManhattan\n(통계 베이스라인)", "Vanilla1DCNN\n(딥러닝 베이스라인)", "KeyPrintNet\n(제안 모델)"]
    keys = ["ScaledManhattan", "Vanilla1DCNN", "KeyPrintNet"]
    top1_vals = [id_comparison[k]["top1_accuracy"] * 100 for k in keys]
    top3_vals = [id_comparison[k]["top3_accuracy"] * 100 for k in keys]
    x = np.arange(len(keys))
    w = 0.34
    b1 = ax.bar(x - w / 2, top1_vals, width=w, color="#2563EB", label="Top-1 정확도 (%)", edgecolor="#1E293B")
    b2 = ax.bar(x + w / 2, top3_vals, width=w, color="#38BDF8", label="Top-3 정확도 (%)", edgecolor="#1E293B")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax.axhline(10.0, color="#EF4444", linestyle=":", lw=1.5, label="무작위 추측 (10.0%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9.5)
    ax.set_ylim(0, 109)
    ax.set_ylabel("식별 정확도 (%) ↑")
    ax.set_title("(b) 모델별 1:N 미학습 인물 분류 정확도 비교", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel C: Practice Sentence Length Sensitivity on KeyRecs Unseen Subjects
    ax = axes[2]
    if sent_len_curve:
        lens_chars = [r["num_chars"] for r in sent_len_curve]
        s_top1 = [r["same_session_top1"] * 100 for r in sent_len_curve]
        s_top3 = [r["same_session_top3"] * 100 for r in sent_len_curve]
        c_top1 = [r["cross_session_top1"] * 100 for r in sent_len_curve]
        n_way = sent_len_curve[0]["num_classes"]
        ax.plot(lens_chars, s_top3, "--^", color="#0EA5E9", lw=2.0, markersize=6.5, label=f"타자연습 Top-3 식별률 ({n_way}인)")
        ax.plot(lens_chars, s_top1, "-o", color="#2563EB", lw=2.4, markersize=7.0, label=f"동일 세션 문장 Top-1 식별률 ({n_way}인)")
        ax.plot(lens_chars, c_top1, "-s", color="#10B981", lw=2.1, markersize=6.5, label=f"타 세션·타 문장 Top-1 식별률 ({n_way}인)")
        ax.axhline(100.0 / n_way, color="#EF4444", linestyle=":", lw=1.5, label=f"무작위 추측 ({100.0/n_way:.1f}%)")
        for lx, vy in zip(lens_chars, s_top1):
            ax.text(lx, vy + 1.8, f"{vy:.1f}%", ha="center", fontsize=8.8, fontweight="bold", color="#1E3A8A")
        ax.set_xticks(lens_chars)
        ax.set_xticklabels([f"{c}자\n(L={c-1})" for c in lens_chars])
        ax.set_ylim(0, 105)
        ax.set_xlabel("타자연습 문장 길이 (글자 수 / 디그래프 전이 수)")
        ax.set_ylabel("미학습 인물 식별 정확도 (%) ↑")
        ax.set_title("(c) 타자연습 문장 길이에 따른 인물 분류 정확도 향상", fontsize=11.3, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.4)
        ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_typist_identification.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig8_feature_upgrade(
    web_id: Dict[str, Dict[str, Dict]],
    password_1to1: Dict[str, Dict],
    password_touch: Dict[str, Dict],
    password_transfer: Dict[str, Dict],
    fig_dir: Path,
) -> None:
    """Before/after figure for key-identity channels, touch input and behavioural fusion."""
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
    colors = ["#94A3B8", "#2563EB", "#059669"]

    def _bars(ax, groups, labels, title, chance=None, ylabel="Top-1 정확도 (%) ↑"):
        x = np.arange(len(groups))
        w = 0.8 / len(labels)
        for j, lab in enumerate(labels):
            vals = [g[1][j] for g in groups]
            bars = ax.bar(x + (j - (len(labels) - 1) / 2) * w, vals, width=w, color=colors[j], edgecolor="#1E293B", label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
        if chance is not None:
            ax.axhline(chance, color="#EF4444", linestyle=":", lw=1.4, label=f"무작위 ({chance:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
        ax.set_ylim(0, 105)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.3, fontweight="bold")
        ax.legend(fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    kb, tc = web_id["keyboard"], web_id["touch"]
    n_way = kb["keyprint_17ch"]["cross_session"]["num_classes"]
    pick = lambda d, split: d[split]["top1_accuracy"] * 100  # noqa: E731
    _bars(
        axes[0],
        [
            ("같은 날 다른 문장", [pick(kb["legacy_9ch"], "same_session"), pick(kb["keyprint_17ch"], "same_session"), pick(kb["keyprint_17ch_behavior"], "same_session")]),
            ("다른 날 다른 글", [pick(kb["legacy_9ch"], "cross_session"), pick(kb["keyprint_17ch"], "cross_session"), pick(kb["keyprint_17ch_behavior"], "cross_session")]),
        ],
        ["기존 9채널", "17채널 (키 위치)", "17채널 + 행동 프로필"],
        f"(a) 키보드 문장 입력 {n_way}인 식별 (웹 캡처 재현, 3문장 등록)",
        chance=100.0 / n_way,
    )
    _bars(
        axes[1],
        [
            ("같은 날 다른 문장", [pick(tc["legacy_9ch_imputed_holds"], "same_session"), pick(tc["keyprint_17ch"], "same_session"), pick(tc["keyprint_17ch_behavior"], "same_session")]),
            ("다른 날 다른 글", [pick(tc["legacy_9ch_imputed_holds"], "cross_session"), pick(tc["keyprint_17ch"], "cross_session"), pick(tc["keyprint_17ch_behavior"], "cross_session")]),
        ],
        ["기존 9채널 (누름시간 대체)", "17채널 (터치 학습)", "17채널 + 행동 프로필"],
        f"(b) 터치 키보드 시뮬레이션 {n_way}인 식별 (누름 시간 없음)",
        chance=100.0 / n_way,
    )
    pt = password_transfer
    _bars(
        axes[2],
        [
            ("CMU\n(학습한 암호)", [password_1to1["legacy_9ch"]["roc_auc"] * 100, password_1to1["keyprint_17ch"]["roc_auc"] * 100]),
            ("KeyRecs 고정문\n(다른 날)", [pt["keyrecs_fixed_cross_session"]["legacy_9ch"]["roc_auc"] * 100, pt["keyrecs_fixed_cross_session"]["keyprint_17ch"]["roc_auc"] * 100]),
            ("반복 단어\n(미학습 문자열)", [pt["repeated_words_unseen"]["legacy_9ch"]["roc_auc"] * 100, pt["repeated_words_unseen"]["keyprint_17ch"]["roc_auc"] * 100]),
            ("터치\n(CMU 시뮬)", [password_touch["legacy_9ch_imputed_holds"]["roc_auc"] * 100, password_touch["keyprint_17ch"]["roc_auc"] * 100]),
        ],
        ["기존 9채널 (비밀번호 배포 모델)", "17채널"],
        "(c) 1:1 비밀번호 인증 ROC-AUC — 암호 내용별",
        ylabel="ROC-AUC (%) ↑",
    )
    axes[2].set_ylim(50, 102)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_feature_upgrade.png", bbox_inches="tight")
    plt.close(fig)


def export_web_demo_bundle(
    test_split,
    sentence_test: Optional[SentenceSplitData],
    meta: Dict,
    cohort: np.ndarray,
    threshold: float,
    eer: float,
    macro_f1: float,
    roc_auc: float,
    identification_metrics: Dict,
    web_models_dir: Path,
    extra_payload: Optional[Dict] = None,
) -> None:
    """
    Export scaler parameters, AS-Norm cohort, calibrated threshold, Mode 1 (1:1 Guard)
    presets, AND Mode 2 (1:N Typing-Practice Who-Is-Typing Arena) real unseen galleries!
    """
    _, hold_cols, dd_cols, ud_cols = get_feature_columns()
    web_models_dir.mkdir(parents=True, exist_ok=True)

    def _row_to_stroke(row: pd.Series) -> Dict:
        return stroke_payload(
            fixed_stroke_capture(
                [row[c] for c in hold_cols], [row[c] for c in dd_cols], [row[c] for c in ud_cols], CMU_PASSWORD_CODES
            )
        )

    # Mode 1: 1:1 Password Intrusion Guard Presets (4 Unseen CMU Subjects)
    preset_profiles = []
    for idx_u in range(min(4, len(test_split.subjects))):
        sname = test_split.subjects[idx_u]
        u_df = test_split.df[test_split.subject_ids == idx_u]
        enroll_rows = u_df[(u_df["sessionIndex"] == 1) & (u_df["rep"] <= 5)]
        genuine_row = u_df[(u_df["sessionIndex"] == 6) & (u_df["rep"] == 10)].iloc[0]

        other_uid = (idx_u + 1) % len(test_split.subjects)
        impostor_row = test_split.df[
            (test_split.subject_ids == other_uid)
            & (test_split.sessions == 4)
            & (test_split.reps == 5)
        ].iloc[0]

        preset_profiles.append(
            {
                "subject": sname,
                "label": f"CMU 미학습 피험자 {sname}",
                "password": ".tie5Roanl",
                "enrollment_strokes": [_row_to_stroke(r) for _, r in enroll_rows.iterrows()],
                "genuine_test_stroke": _row_to_stroke(genuine_row),
                "skilled_impostor_stroke": _row_to_stroke(impostor_row),
                "impostor_subject": test_split.subjects[other_uid],
            }
        )

    # Mode 2: 1:N Typing-Practice Multi-User Identification Galleries (8 Unseen People + 2 Outsiders!)
    arena_galleries = []

    # Gallery A: Real English Sentence Typing Practice (KeyRecs Unseen 8 People, 25-char sentences)
    if sentence_test is not None and len(sentence_test.subjects) >= 10:
        sent_members = []
        for uid in range(8):
            sname = sentence_test.subjects[uid]
            s1_idx = np.where((sentence_test.subject_ids == uid) & (sentence_test.sessions == 1))[0]
            s2_idx = np.where((sentence_test.subject_ids == uid) & (sentence_test.sessions == 2))[0]
            e_indices = s1_idx[:5]
            q_same = int(s1_idx[6])
            q_cross = int(s2_idx[1])

            def _s_stroke(i: int) -> Dict:
                return {
                    **stroke_payload(
                        fixed_stroke_capture(
                            sentence_test.holds[i], sentence_test.dds[i], sentence_test.uds[i], sentence_test.codes[i]
                        )
                    ),
                    "typed_sample": sentence_test.texts[i],
                }

            sent_members.append(
                {
                    "id": sname,
                    "name": f"참가자 {sname.upper()}",
                    "badge": f"Unseen #{uid + 1}",
                    "enrollment_strokes": [_s_stroke(int(i)) for i in e_indices],
                    "blind_test_stroke": _s_stroke(q_same),
                    "cross_sentence_stroke": _s_stroke(q_cross),
                }
            )

        outsider_idx = int(np.where((sentence_test.subject_ids == 9) & (sentence_test.sessions == 1))[0][2])
        arena_galleries.append(
            {
                "gallery_id": "keyrecs_english_sentence",
                "title": "영문 타자연습 실측 8인 갤러리 (KeyRecs Unseen)",
                "practice_text": "we believe that we can c",
                "description": "실제 영문 에세이 타자연습(25자 시퀀스)을 수행한 미학습 피험자 8인의 타건 프로필",
                "members": sent_members,
                "outsider_stroke": {
                    "name": f"미등록 외부인 ({sentence_test.subjects[9].upper()})",
                    **stroke_payload(
                        fixed_stroke_capture(
                            sentence_test.holds[outsider_idx],
                            sentence_test.dds[outsider_idx],
                            sentence_test.uds[outsider_idx],
                            sentence_test.codes[outsider_idx],
                        )
                    ),
                },
            }
        )

    # Gallery B: Fixed Shared Phrase 8-Person Gallery (CMU Unseen 8 People + Outsider #9)
    cmu_members = []
    for uid in range(min(8, len(test_split.subjects))):
        sname = test_split.subjects[uid]
        u_df = test_split.df[test_split.subject_ids == uid]
        e_rows = u_df[(u_df["sessionIndex"] == 1) & (u_df["rep"] <= 5)]
        b_row = u_df[(u_df["sessionIndex"] == 5) & (u_df["rep"] == 12)].iloc[0]
        cmu_members.append(
            {
                "id": sname,
                "name": f"피험자 {sname}",
                "badge": f"CMU Unseen #{uid + 1}",
                "enrollment_strokes": [_row_to_stroke(r) for _, r in e_rows.iterrows()],
                "blind_test_stroke": _row_to_stroke(b_row),
            }
        )
    out_cmu_row = test_split.df[(test_split.subject_ids == 9) & (test_split.sessions == 4) & (test_split.reps == 8)].iloc[0]
    arena_galleries.append(
        {
            "gallery_id": "cmu_shared_phrase",
            "title": "동일 문구 8인 타자 식별 갤러리 (CMU Unseen)",
            "practice_text": ".tie5Roanl",
            "description": "8명의 미학습 인물이 완전히 동일한 문자열을 쳤을 때 리듬만으로 누구인지 분류",
            "members": cmu_members,
            "outsider_stroke": {
                "name": f"미등록 외부인 ({test_split.subjects[9]})",
                **_row_to_stroke(out_cmu_row),
            },
        }
    )

    config_payload = {
        "model_name": "KeyPrintNet (Positional Multi-Scale ResSE-1DCNN + ASP/DCT + SubCenter-CosFace + AS-Norm)",
        "model_version": MODEL_VERSION,
        "embedding_dim": 64,
        "in_channels": N_CHANNELS,
        "channel_names": CHANNEL_NAMES,
        "channel_mean": meta["channel_mean"],
        "channel_std": meta["channel_std"],
        "outlier_bounds": meta["outlier_bounds"],
        "cohort_centroids": [[round(float(v), 5) for v in row] for row in cohort],
        "calibrated_threshold": round(float(threshold), 4),
        "strict_threshold": round(min(0.92, float(threshold) + 0.10), 4),
        "open_set_id_threshold": round(max(0.35, float(threshold) - 0.04), 4),
        "test_metrics": {
            "macro_f1": macro_f1,
            "roc_auc": roc_auc,
            "eer": eer,
            "identification_10way_top1": identification_metrics["cmu_10way"]["KeyPrintNet"]["top1_accuracy"],
            "identification_10way_top3": identification_metrics["cmu_10way"]["KeyPrintNet"]["top3_accuracy"],
        },
        "benchmark_presets": preset_profiles,
        "typing_arena_galleries": arena_galleries,
    }
    if extra_payload:
        config_payload.update(extra_payload)
    with open(web_models_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    t0 = time.time()
    set_all_seeds(SEED)
    chosen_font = configure_korean_matplotlib()
    device = get_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"=== KeyPrint Guard & Arena Pipeline | Device: {device} ({gpu_name}) | Font: {chosen_font} ===")

    root = Path(__file__).resolve().parent.parent
    out_root = Path(os.environ["KEYPRINT_OUTPUT_ROOT"]).resolve() if os.environ.get("KEYPRINT_OUTPUT_ROOT") else root
    csv_path = root / "data" / "raw" / "DSL-StrongPasswordData.csv"
    proc_dir = out_root / "data" / "processed"
    models_dir = out_root / "models"
    fig_dir = out_root / "assets" / "figures"
    web_models_dir = out_root / "web" / "public" / "models"
    parity_fixture_path = out_root / "web" / "scripts" / "fixtures" / "feature_parity.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "legacy").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(csv_path)
    splits, meta = prepare_full_pipeline(csv_path, proc_dir, seed=SEED)
    sentence_splits: Dict[str, SentenceSplitData] = meta.pop("_sentence_splits", {})
    kr_fixed_train = meta.pop("_keyrecs_fixed_train", None)
    meta.pop("_keyrecs_runs", None)
    kr_subjs = meta.pop("_keyrecs_splits_subjs", None)

    sent_train = sentence_splits.get("train")
    sent_val = sentence_splits.get("val")
    sent_test = sentence_splits.get("test")

    chan_mean = np.array(meta["channel_mean"], dtype=np.float32)
    chan_std = np.array(meta["channel_std"], dtype=np.float32)
    bounds = {k: tuple(v) for k, v in meta["outlier_bounds"].items()}
    touch_hold = float(meta["touch_hold_impute_sec"])

    n_test = len(splits["test"].df)
    test_unclipped4 = np.stack(
        [
            splits["test"].raw_holds[:, :-1],
            splits["test"].raw_holds[:, 1:],
            splits["test"].raw_dds,
            splits["test"].raw_uds,
        ],
        axis=1,
    ).reshape(n_test, -1)
    test_clipped4_unscaled = splits["test"].seq_unscaled[:, :4, :].reshape(n_test, -1)
    test_clipped4_scaled = splits["test"].seq_scaled[:, :4, :].reshape(n_test, -1)
    test_bio9 = splits["test"].seq_scaled[:, :N_TIMING_CHANNELS, :].reshape(n_test, -1)
    test_bio17 = splits["test"].seq_scaled[:, :N_CHANNELS, :].reshape(n_test, -1)

    sil_raw4 = float(silhouette_score(test_unclipped4, splits["test"].subject_ids, metric="cosine"))
    sil_clipped4_unscaled = float(silhouette_score(test_clipped4_unscaled, splits["test"].subject_ids, metric="cosine"))
    sil_clipped4_scaled = float(silhouette_score(test_clipped4_scaled, splits["test"].subject_ids, metric="cosine"))
    sil_bio9 = float(silhouette_score(test_bio9, splits["test"].subject_ids, metric="cosine"))
    sil_bio17 = float(silhouette_score(test_bio17, splits["test"].subject_ids, metric="cosine"))
    print(
        f"[Preprocessing] Test Cosine Silhouette: Raw4={sil_raw4:.4f} -> Clipped4={sil_clipped4_unscaled:.4f} -> "
        f"Scaled4={sil_clipped4_scaled:.4f} -> Bio9={sil_bio9:.4f} -> Bio17={sil_bio17:.4f}"
    )

    val_trials = build_verification_trials(splits["val"], chan_mean, chan_std, k_shot=5, seed=SEED)
    test_trials = build_verification_trials(splits["test"], chan_mean, chan_std, k_shot=5, seed=SEED)

    # 1. Baseline 1: Statistical Scaled Manhattan Detector
    manhattan = ScaledManhattanDetector(shrinkage=0.35).fit_prior(
        splits["train"].raw_flat, splits["train"].subject_ids
    )
    yv_m, sv_m = score_manhattan_trials(manhattan, splits["val"], val_trials)
    thresh_m = find_optimal_val_threshold(yv_m, sv_m)
    yt_m, st_m = score_manhattan_trials(manhattan, splits["test"], test_trials)
    metrics_manhattan = summarize_metrics(yt_m, st_m, thresh_m)
    metrics_manhattan["impostor_breakdown"] = impostor_breakdown_manhattan(manhattan, splits["test"], test_trials, thresh_m)
    print(f"[Result 1:1] ScaledManhattan Test: {metrics_manhattan}")

    report_path = models_dir / "evaluation_report.json"
    prev_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if (REUSE_WEIGHTS and report_path.exists())
        else None
    )

    # 2a. Baseline 2: Scratch Vanilla 1D-CNN (same dual-corpus Train, no augmentation, no AS-Norm)
    vanilla_pt = models_dir / "vanilla_1dcnn.pt"
    if REUSE_WEIGHTS and vanilla_pt.exists():
        vanilla_model = Vanilla1DCNN(in_channels=splits["train"].seq_scaled.shape[1], embed_dim=64)
        vanilla_model.load_state_dict(torch.load(vanilla_pt, map_location="cpu", weights_only=True))
        vanilla_model.to(device).eval()
        hist_vanilla = (prev_report or {}).get("histories", {}).get("Vanilla1DCNN", [])
        print("[Reuse] loaded saved Vanilla1DCNN weights")
    else:
        vanilla_model, hist_vanilla = train_vanilla_baseline(
            splits["train"],
            splits["val"],
            val_trials,
            device,
            epochs=VANILLA_EPOCHS,
            batch_size=256,
            sentence_train=sent_train,
            kr_fixed_train=kr_fixed_train,
            use_augmentation=False,
            tag="Vanilla1DCNN",
        )
        torch.save(vanilla_model.cpu().state_dict(), vanilla_pt)
        vanilla_model.to(device)

    val_emb_v = encode_sequences(vanilla_model, splits["val"].seq_scaled, device)
    val_bot_v = encode_sequences(vanilla_model, val_trials["bot_seq"], device)
    yv_v, sv_v, _ = score_embedding_trials(val_emb_v, val_bot_v, val_trials, cohort=None, single_centroid=False)
    thresh_v = find_optimal_val_threshold(yv_v, sv_v)

    test_emb_v = encode_sequences(vanilla_model, splits["test"].seq_scaled, device)
    test_bot_v = encode_sequences(vanilla_model, test_trials["bot_seq"], device)
    yt_v, st_v, _ = score_embedding_trials(test_emb_v, test_bot_v, test_trials, cohort=None, single_centroid=False)
    metrics_vanilla = summarize_metrics(yt_v, st_v, thresh_v)
    metrics_vanilla["impostor_breakdown"] = impostor_breakdown_embedding(
        test_emb_v, test_bot_v, test_trials, thresh_v, cohort=None, single_centroid=False
    )
    print(f"[Result 1:1] Vanilla1DCNN Test: {metrics_vanilla}")

    # 2b. Controlled Vanilla1DCNN_Aug (+Same Dual-Corpus & Augmentation Recipe) & Vanilla1DCNN_Aug_ASNorm
    vanilla_aug_pt = models_dir / "vanilla_1dcnn_aug.pt"
    if REUSE_WEIGHTS and vanilla_aug_pt.exists():
        vanilla_aug_model = Vanilla1DCNN(in_channels=splits["train"].seq_scaled.shape[1], embed_dim=64)
        vanilla_aug_model.load_state_dict(torch.load(vanilla_aug_pt, map_location="cpu", weights_only=True))
        vanilla_aug_model.to(device).eval()
        hist_vanilla_aug = (prev_report or {}).get("histories", {}).get("Vanilla1DCNN_Aug", [])
        print("[Reuse] loaded saved Vanilla1DCNN_Aug weights")
    else:
        vanilla_aug_model, hist_vanilla_aug = train_vanilla_baseline(
            splits["train"],
            splits["val"],
            val_trials,
            device,
            epochs=VANILLA_EPOCHS,
            batch_size=256,
            sentence_train=sent_train,
            kr_fixed_train=kr_fixed_train,
            use_augmentation=True,
            bounds=bounds,
            chan_mean=chan_mean,
            chan_std=chan_std,
            touch_prob=TOUCH_PROB,
            touch_hold=touch_hold,
            tag="Vanilla1DCNN_Aug",
        )
        torch.save(vanilla_aug_model.cpu().state_dict(), vanilla_aug_pt)
        vanilla_aug_model.to(device)

    train_emb_va = encode_sequences(vanilla_aug_model, splits["train"].seq_scaled, device)
    cohort_va = compute_cohort_centroids(train_emb_va, splits["train"].subject_ids)
    val_emb_va = encode_sequences(vanilla_aug_model, splits["val"].seq_scaled, device)
    val_bot_va = encode_sequences(vanilla_aug_model, val_trials["bot_seq"], device)
    test_emb_va = encode_sequences(vanilla_aug_model, splits["test"].seq_scaled, device)
    test_bot_va = encode_sequences(vanilla_aug_model, test_trials["bot_seq"], device)

    yv_va, sv_va, _ = score_embedding_trials(val_emb_va, val_bot_va, val_trials, cohort=None, single_centroid=False)
    thresh_va = find_optimal_val_threshold(yv_va, sv_va)
    yt_va, st_va, _ = score_embedding_trials(test_emb_va, test_bot_va, test_trials, cohort=None, single_centroid=False)
    metrics_vanilla_aug = summarize_metrics(yt_va, st_va, thresh_va)
    metrics_vanilla_aug["impostor_breakdown"] = impostor_breakdown_embedding(
        test_emb_va, test_bot_va, test_trials, thresh_va, cohort=None, single_centroid=False
    )

    yv_va_as, sv_va_as, _ = score_embedding_trials(val_emb_va, val_bot_va, val_trials, cohort=cohort_va, single_centroid=False)
    thresh_va_as = find_optimal_val_threshold(yv_va_as, sv_va_as)
    yt_va_as, st_va_as, _ = score_embedding_trials(test_emb_va, test_bot_va, test_trials, cohort=cohort_va, single_centroid=False)
    metrics_vanilla_aug_as = summarize_metrics(yt_va_as, st_va_as, thresh_va_as)
    metrics_vanilla_aug_as["impostor_breakdown"] = impostor_breakdown_embedding(
        test_emb_va, test_bot_va, test_trials, thresh_va_as, cohort=cohort_va, single_centroid=False
    )
    print(f"[Result 1:1] Vanilla1DCNN_Aug Test: {metrics_vanilla_aug}")
    print(f"[Result 1:1] Vanilla1DCNN_Aug_ASNorm Test: {metrics_vanilla_aug_as}")

    # 3a. Proposed Sentence Encoder: KeyPrintNet 17ch (from scratch unless REUSE_WEIGHTS and .pt exists)
    kp17_pt = models_dir / "keyprint_net.pt"
    if REUSE_WEIGHTS and kp17_pt.exists():
        keyprint_model = KeyPrintNet(in_channels=N_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18)
        keyprint_model.load_state_dict(torch.load(kp17_pt, map_location="cpu", weights_only=True))
        keyprint_model.to(device).eval()
        hist_keyprint = (prev_report or {}).get("histories", {}).get("KeyPrintNet", [])
        print("[Reuse] loaded saved KeyPrintNet 17ch weights")
    else:
        keyprint_model, hist_keyprint = train_keyprint_net(
            splits["train"],
            splits["val"],
            val_trials,
            bounds,
            chan_mean,
            chan_std,
            device,
            epochs=KEYPRINT_EPOCHS,
            batch_size=256,
            use_augmentation=True,
            raw_channels_only=False,
            sentence_train=sent_train,
            kr_fixed_train=kr_fixed_train,
            sentence_val=sent_val,
            tag="KeyPrintNet_17ch",
            in_channels=N_CHANNELS,
            touch_prob=TOUCH_PROB,
            touch_hold=touch_hold,
        )
        torch.save(keyprint_model.cpu().state_dict(), kp17_pt)
        keyprint_model.to(device)

    train_emb_k = encode_sequences(keyprint_model, splits["train"].seq_scaled, device)
    cohort_k = compute_cohort_centroids(train_emb_k, splits["train"].subject_ids)

    val_emb_k = encode_sequences(keyprint_model, splits["val"].seq_scaled, device)
    val_bot_k = encode_sequences(keyprint_model, val_trials["bot_seq"], device)
    yv_k, sv_k, _ = score_embedding_trials(val_emb_k, val_bot_k, val_trials, cohort=cohort_k)
    thresh_k = find_optimal_val_threshold(yv_k, sv_k)

    test_emb_k = encode_sequences(keyprint_model, splits["test"].seq_scaled, device)
    test_bot_k = encode_sequences(keyprint_model, test_trials["bot_seq"], device)
    yt_k, st_k, detailed_k = score_embedding_trials(test_emb_k, test_bot_k, test_trials, cohort=cohort_k)
    metrics_keyprint = summarize_metrics(yt_k, st_k, thresh_k)
    metrics_keyprint["impostor_breakdown"] = impostor_breakdown_embedding(
        test_emb_k, test_bot_k, test_trials, thresh_k, cohort=cohort_k, single_centroid=False
    )
    print(f"[Result 1:1] KeyPrintNet 17ch Test: {metrics_keyprint}")

    # 3b. Proposed Content-Agnostic Password Encoder: KeyPrintNet 9ch (trained from scratch on channels 0..8!)
    L9 = N_TIMING_CHANNELS
    kp9_pt = models_dir / "legacy" / "keyprint_net_9ch.pt"
    if REUSE_WEIGHTS and kp9_pt.exists():
        legacy_model = KeyPrintNet(in_channels=L9, hidden_channels=80, embed_dim=64, dropout=0.18)
        legacy_model.load_state_dict(torch.load(kp9_pt, map_location="cpu", weights_only=True))
        legacy_model.to(device).eval()
        hist_keyprint_9ch = (prev_report or {}).get("histories", {}).get("KeyPrintNet_9ch", [])
        print("[Reuse] loaded saved KeyPrintNet_9ch weights")
    else:
        legacy_model, hist_keyprint_9ch = train_keyprint_net(
            splits["train"],
            splits["val"],
            val_trials,
            bounds,
            chan_mean,
            chan_std,
            device,
            epochs=KEYPRINT_EPOCHS,
            batch_size=256,
            use_augmentation=True,
            raw_channels_only=False,
            sentence_train=sent_train,
            kr_fixed_train=kr_fixed_train,
            sentence_val=sent_val,
            tag="KeyPrintNet_9ch",
            in_channels=L9,
            touch_prob=0.0,
            touch_hold=touch_hold,
        )
        kp9_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(legacy_model.cpu().state_dict(), kp9_pt)
        legacy_model.to(device)

    cohort_l = compute_cohort_centroids(
        encode_sequences(legacy_model, splits["train"].seq_scaled[:, :L9], device), splits["train"].subject_ids
    )
    val_emb_l = encode_sequences(legacy_model, splits["val"].seq_scaled[:, :L9], device)
    val_bot_l = encode_sequences(legacy_model, val_trials["bot_seq"][:, :L9], device)
    yv_l, sv_l, _ = score_embedding_trials(val_emb_l, val_bot_l, val_trials, cohort=cohort_l)
    test_emb_l = encode_sequences(legacy_model, splits["test"].seq_scaled[:, :L9], device)
    test_bot_l = encode_sequences(legacy_model, test_trials["bot_seq"][:, :L9], device)
    yt_l, st_l, _ = score_embedding_trials(test_emb_l, test_bot_l, test_trials, cohort=cohort_l)
    thresh_l = find_optimal_val_threshold(yv_l, sv_l)
    metrics_legacy = summarize_metrics(yt_l, st_l, thresh_l)
    metrics_legacy["impostor_breakdown"] = impostor_breakdown_embedding(
        test_emb_l, test_bot_l, test_trials, thresh_l, cohort=cohort_l, single_centroid=False
    )
    print(f"[Result 1:1] KeyPrintNet 9ch Test: {metrics_legacy}")

    # 4. Task 2 Evaluation: 1:N Multi-User Typing-Practice Identification ("Who Is Typing?")
    id_cmu_m = evaluate_1toN_identification_manhattan(
        manhattan,
        splits["test"].raw_flat,
        splits["test"].subject_ids,
        splits["test"].sessions,
        splits["test"].reps,
        splits["test"].subjects,
        k_shot=5,
        cross_session=True,
    )
    id_cmu_v, _ = evaluate_1toN_identification_embeddings(
        test_emb_v,
        splits["test"].subject_ids,
        splits["test"].sessions,
        splits["test"].reps,
        splits["test"].subjects,
        k_shot=5,
        cross_session=True,
        single_centroid=False,
    )
    id_cmu_va, _ = evaluate_1toN_identification_embeddings(
        test_emb_va,
        splits["test"].subject_ids,
        splits["test"].sessions,
        splits["test"].reps,
        splits["test"].subjects,
        k_shot=5,
        cross_session=True,
        single_centroid=False,
    )
    id_cmu_va_as, _ = evaluate_1toN_identification_embeddings(
        test_emb_va,
        splits["test"].subject_ids,
        splits["test"].sessions,
        splits["test"].reps,
        splits["test"].subjects,
        k_shot=5,
        cross_session=True,
        cohort=cohort_va,
    )
    id_cmu_k, cm_10way = evaluate_1toN_identification_embeddings(
        test_emb_k,
        splits["test"].subject_ids,
        splits["test"].sessions,
        splits["test"].reps,
        splits["test"].subjects,
        k_shot=5,
        cross_session=True,
        cohort=cohort_k,
    )
    id_cmu_l, _ = evaluate_1toN_identification_embeddings(
        test_emb_l,
        splits["test"].subject_ids,
        splits["test"].sessions,
        splits["test"].reps,
        splits["test"].subjects,
        k_shot=5,
        cross_session=True,
        cohort=cohort_l,
    )
    id_comparison_cmu = {
        "ScaledManhattan": id_cmu_m,
        "Vanilla1DCNN": id_cmu_v,
        "Vanilla1DCNN_Aug": id_cmu_va,
        "Vanilla1DCNN_Aug_ASNorm": id_cmu_va_as,
        "KeyPrintNet_9ch": id_cmu_l,
        "KeyPrintNet": id_cmu_k,
    }
    print(f"[Result 1:N Same-Phrase 10-Way Identification] {id_comparison_cmu}")

    # Sentence-Length Sensitivity on Real English Typing-Practice Corpus (KeyRecs Unseen 18 Subjects!)
    sent_len_curve = []
    sent_id_summary = {}
    if sent_test is not None and sent_train is not None:
        sent_tr_emb = encode_sequences(keyprint_model, sent_train.seq_scaled, device)
        sent_cohort = compute_cohort_centroids(sent_tr_emb, sent_train.subject_ids)

        sent_man = ScaledManhattanDetector(shrinkage=0.35).fit_prior(sent_train.raw_flat, sent_train.subject_ids)
        sent_id_m = evaluate_1toN_identification_manhattan(
            sent_man,
            sent_test.raw_flat,
            sent_test.subject_ids,
            sent_test.sessions,
            sent_test.chunk_ids,
            sent_test.subjects,
            k_shot=5,
            cross_session=False,
        )
        sent_emb_v = encode_sequences(vanilla_model, sent_test.seq_scaled, device)
        sent_id_v, _ = evaluate_1toN_identification_embeddings(
            sent_emb_v,
            sent_test.subject_ids,
            sent_test.sessions,
            sent_test.chunk_ids,
            sent_test.subjects,
            k_shot=5,
            cross_session=False,
            single_centroid=False,
            t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
        )
        sent_emb_va = encode_sequences(vanilla_aug_model, sent_test.seq_scaled, device)
        sent_va_cohort = compute_cohort_centroids(
            encode_sequences(vanilla_aug_model, sent_train.seq_scaled, device), sent_train.subject_ids
        )
        sent_id_va, _ = evaluate_1toN_identification_embeddings(
            sent_emb_va,
            sent_test.subject_ids,
            sent_test.sessions,
            sent_test.chunk_ids,
            sent_test.subjects,
            k_shot=5,
            cross_session=False,
            single_centroid=False,
            t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
        )
        sent_id_va_as, _ = evaluate_1toN_identification_embeddings(
            sent_emb_va,
            sent_test.subject_ids,
            sent_test.sessions,
            sent_test.chunk_ids,
            sent_test.subjects,
            k_shot=5,
            cross_session=False,
            cohort=sent_va_cohort,
            t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
        )
        sent_emb_k = encode_sequences(keyprint_model, sent_test.seq_scaled, device)
        sent_id_k_same, _ = evaluate_1toN_identification_embeddings(
            sent_emb_k,
            sent_test.subject_ids,
            sent_test.sessions,
            sent_test.chunk_ids,
            sent_test.subjects,
            k_shot=5,
            cross_session=False,
            cohort=sent_cohort,
            t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
        )
        sent_id_k_cross, _ = evaluate_1toN_identification_embeddings(
            sent_emb_k,
            sent_test.subject_ids,
            sent_test.sessions,
            sent_test.chunk_ids,
            sent_test.subjects,
            k_shot=5,
            cross_session=True,
            cohort=sent_cohort,
            t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
        )
        sent_cohort_legacy = compute_cohort_centroids(
            encode_sequences(legacy_model, sent_train.seq_scaled[:, :L9], device), sent_train.subject_ids
        )
        sent_emb_l = encode_sequences(legacy_model, sent_test.seq_scaled[:, :L9], device)
        sent_id_l = {
            mode: evaluate_1toN_identification_embeddings(
                sent_emb_l,
                sent_test.subject_ids,
                sent_test.sessions,
                sent_test.chunk_ids,
                sent_test.subjects,
                k_shot=5,
                cross_session=cross,
                cohort=sent_cohort_legacy,
                t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
            )[0]
            for mode, cross in (("same_session", False), ("cross_session_different_essay", True))
        }
        sent_id_summary = {
            "ScaledManhattan_same_session": sent_id_m,
            "Vanilla1DCNN_same_session": sent_id_v,
            "Vanilla1DCNN_Aug_same_session": sent_id_va,
            "Vanilla1DCNN_Aug_ASNorm_same_session": sent_id_va_as,
            "KeyPrintNet_same_session": sent_id_k_same,
            "KeyPrintNet_cross_session_different_essay": sent_id_k_cross,
            "Legacy9ch_same_session": sent_id_l["same_session"],
            "Legacy9ch_cross_session_different_essay": sent_id_l["cross_session_different_essay"],
        }
        print(f"[Result 1:N English Sentence 18-Way Identification] {sent_id_summary}")

        for sub_l in [8, 12, 18, 24]:
            tr_l = encode_sequences(keyprint_model, sent_train.seq_scaled[:, :, :sub_l], device)
            coh_l = compute_cohort_centroids(tr_l, sent_train.subject_ids)
            te_l = encode_sequences(keyprint_model, sent_test.seq_scaled[:, :, :sub_l], device)
            m_same, _ = evaluate_1toN_identification_embeddings(
                te_l,
                sent_test.subject_ids,
                sent_test.sessions,
                sent_test.chunk_ids,
                sent_test.subjects,
                k_shot=5,
                cross_session=False,
                cohort=coh_l,
                t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
            )
            m_cross, _ = evaluate_1toN_identification_embeddings(
                te_l,
                sent_test.subject_ids,
                sent_test.sessions,
                sent_test.chunk_ids,
                sent_test.subjects,
                k_shot=5,
                cross_session=True,
                cohort=coh_l,
                t_ranges=sent_test.t_ranges,
            key_idx_ranges=sent_test.key_idx_ranges,
            )
            sent_len_curve.append(
                {
                    "digraphs": sub_l,
                    "num_chars": sub_l + 1,
                    "num_classes": m_same["num_classes"],
                    "same_session_top1": m_same["top1_accuracy"],
                    "same_session_top3": m_same["top3_accuracy"],
                    "cross_session_top1": m_cross["top1_accuracy"],
                }
            )
        print(f"[Sentence Length Identification Curve] {sent_len_curve}")

    # 5. Case Study Verification: Session-1-Only Enrollment -> Sessions 2..8 Temporal Speed Drift!
    drift_val_trials = build_verification_trials(
        splits["val"], chan_mean, chan_std, k_shot=5, seed=SEED, session1_only_enroll=True
    )
    drift_test_trials = build_verification_trials(
        splits["test"], chan_mean, chan_std, k_shot=5, seed=SEED, session1_only_enroll=True
    )

    yvm_d, svm_d = score_manhattan_trials(manhattan, splits["val"], drift_val_trials)
    ytm_d, stm_d = score_manhattan_trials(manhattan, splits["test"], drift_test_trials)
    drift_manhattan = summarize_metrics(ytm_d, stm_d, find_optimal_val_threshold(yvm_d, svm_d))

    vb_v_d = encode_sequences(vanilla_model, drift_val_trials["bot_seq"], device)
    tb_v_d = encode_sequences(vanilla_model, drift_test_trials["bot_seq"], device)
    yvv_d, svv_d, _ = score_embedding_trials(val_emb_v, vb_v_d, drift_val_trials, cohort=None, single_centroid=False)
    ytv_d, stv_d, _ = score_embedding_trials(test_emb_v, tb_v_d, drift_test_trials, cohort=None, single_centroid=False)
    drift_vanilla = summarize_metrics(ytv_d, stv_d, find_optimal_val_threshold(yvv_d, svv_d))

    vb_k_d = encode_sequences(keyprint_model, drift_val_trials["bot_seq"], device)
    tb_k_d = encode_sequences(keyprint_model, drift_test_trials["bot_seq"], device)
    yvk_d, svk_d, _ = score_embedding_trials(val_emb_k, vb_k_d, drift_val_trials, cohort=cohort_k)
    ytk_d, stk_d, _ = score_embedding_trials(test_emb_k, tb_k_d, drift_test_trials, cohort=cohort_k)
    drift_keyprint = summarize_metrics(ytk_d, stk_d, find_optimal_val_threshold(yvk_d, svk_d))

    # Timing-only 9ch encoder (deployed for passwords) under the same drift protocol
    vb_l_d = encode_sequences(legacy_model, drift_val_trials["bot_seq"][:, :L9], device)
    tb_l_d = encode_sequences(legacy_model, drift_test_trials["bot_seq"][:, :L9], device)
    yvl_d, svl_d, _ = score_embedding_trials(val_emb_l, vb_l_d, drift_val_trials, cohort=cohort_l)
    ytl_d, stl_d, _ = score_embedding_trials(test_emb_l, tb_l_d, drift_test_trials, cohort=cohort_l)
    drift_legacy = summarize_metrics(ytl_d, stl_d, find_optimal_val_threshold(yvl_d, svl_d))

    ablation_summary = {
        "step1_raw_manhattan": drift_manhattan,
        "step2_vanilla_cnn": drift_vanilla,
        "step3_keyprint_proposed": drift_keyprint,
        "step4_keyprint_9ch_password_encoder": drift_legacy,
    }
    print(f"[Ablation Session-1 Drift] {ablation_summary}")

    # 6. Enrollment K-shot Sensitivity & Variable-Length Subsequence Evaluation
    kshot_results = []
    for k_s in [1, 3, 5, 10]:
        kt_val = build_verification_trials(splits["val"], chan_mean, chan_std, k_shot=k_s, seed=SEED)
        kt_test = build_verification_trials(splits["test"], chan_mean, chan_std, k_shot=k_s, seed=SEED)
        yv_ks, sv_ks, _ = score_embedding_trials(val_emb_k, val_bot_k, kt_val, cohort=cohort_k)
        t_ks = find_optimal_val_threshold(yv_ks, sv_ks)
        yt_ks, st_ks, _ = score_embedding_trials(test_emb_k, test_bot_k, kt_test, cohort=cohort_k)
        sm = summarize_metrics(yt_ks, st_ks, t_ks)
        kshot_results.append({"k_shot": k_s, **sm})

    varlen_results = {}
    for seq_l in [6, 8, 10]:
        tr_l = encode_sequences(keyprint_model, splits["train"].seq_scaled[:, :, :seq_l], device)
        coh_l = compute_cohort_centroids(tr_l, splits["train"].subject_ids)
        ve_l = encode_sequences(keyprint_model, splits["val"].seq_scaled[:, :, :seq_l], device)
        vb_l = encode_sequences(keyprint_model, val_trials["bot_seq"][:, :, :seq_l], device)
        yv_l, sv_l, _ = score_embedding_trials(ve_l, vb_l, val_trials, cohort=coh_l)
        t_l = find_optimal_val_threshold(yv_l, sv_l)

        te_l = encode_sequences(keyprint_model, splits["test"].seq_scaled[:, :, :seq_l], device)
        tb_l = encode_sequences(keyprint_model, test_trials["bot_seq"][:, :, :seq_l], device)
        yt_l, st_l, _ = score_embedding_trials(te_l, tb_l, test_trials, cohort=coh_l)
        varlen_results[f"digraphs_{seq_l}"] = summarize_metrics(yt_l, st_l, t_l)


    # 6b. Key identity, touch input & behavioural profile — evaluated exactly like the web app
    _, hold_cols_b, dd_cols_b, ud_cols_b = get_feature_columns()
    raw_sessions = {
        k: ms_rounded(v) for k, v in load_keyrecs_sessions(root / "data" / "raw" / "KeyRecs-free-text.csv").items()
    }
    behavior_model = fit_behavior_model(
        {k: to_events(v) for k, v in raw_sessions.items()}, kr_subjs["train"], window=70
    )
    print(f"[Behavior] fitted {len(behavior_model.params)} scored features on {len(kr_subjs['train'])} Train people")

    def _touch_seq(raw_holds, raw_dds, key_block):
        c_dds = np.clip(raw_dds, *bounds["DD"]).astype(np.float32)
        h_t, u_t = touchify_timings(raw_holds, raw_dds, touch_hold, bounds)
        seq = extract_sequences(h_t, c_dds, u_t, key_block, np.zeros(len(c_dds), dtype=np.float32))
        return apply_channel_scaler(seq, chan_mean, chan_std)

    # Cohorts used by AS-Norm (keyboard + touch variants, password + sentence domains)
    cmu_tr_h = splits["train"].raw_holds
    cmu_tr_d = splits["train"].raw_dds
    cohort_k_touch = compute_cohort_centroids(
        encode_sequences(keyprint_model, _touch_seq(cmu_tr_h, cmu_tr_d, splits["train"].key_block), device),
        splits["train"].subject_ids,
    )
    sent_train_touch = _touch_seq(sent_train.raw_holds, sent_train.raw_dds, sent_train.key_blocks)
    sent_cohort_touch = compute_cohort_centroids(
        encode_sequences(keyprint_model, sent_train_touch, device), sent_train.subject_ids
    )
    sent_cohort_legacy_touch = compute_cohort_centroids(
        encode_sequences(legacy_model, sent_train_touch[:, :L9], device), sent_train.subject_ids
    )

    # Browser-equivalent capture banks replayed from raw KeyRecs streams (real typos + Backspace)
    banks = {
        "val": build_capture_bank(raw_sessions, kr_subjs["val"], touch=False, target_chars=WEB_CAPTURE_CHARS),
        "test": build_capture_bank(raw_sessions, kr_subjs["test"], touch=False, target_chars=WEB_CAPTURE_CHARS),
        "val_touch": build_capture_bank(raw_sessions, kr_subjs["val"], touch=True, target_chars=WEB_CAPTURE_CHARS),
        "test_touch": build_capture_bank(raw_sessions, kr_subjs["test"], touch=True, target_chars=WEB_CAPTURE_CHARS),
    }
    for bank in banks.values():
        attach_tensors(bank, bounds, touch_hold, chan_mean, chan_std)
        embed_bank(bank, keyprint_model, device, N_CHANNELS, "emb")
        embed_bank(bank, legacy_model, device, L9, "emb_legacy")

    # CMU password strokes: only pause rhythm is observable (no typos/modifiers recorded)
    def _cmu_stats(split):
        return cmu_behavior_stats(split.df[hold_cols_b].values, split.df[dd_cols_b].values)

    val_stats_cmu, test_stats_cmu = _cmu_stats(splits["val"]), _cmu_stats(splits["test"])
    val_bot_stats = cmu_behavior_stats(*flat_to_hdu(val_trials["bot_flat"]))
    test_bot_stats = cmu_behavior_stats(*flat_to_hdu(test_trials["bot_flat"]))
    cmu_val_args = dict(
        embeddings=val_emb_k, bot_embeddings=val_bot_k, trials=val_trials, cohort=cohort_k,
        stats=val_stats_cmu, bot_stats=val_bot_stats,
    )
    cmu_test_args = dict(
        embeddings=test_emb_k, bot_embeddings=test_bot_k, trials=test_trials, cohort=cohort_k,
        stats=test_stats_cmu, bot_stats=test_bot_stats,
    )

    # Fusion weights are calibrated per domain on Validation data only (same code, same features)
    beta_kb, grid_kb = tune_beta_identification(banks["val"], "emb", sent_cohort, behavior_model)
    beta_touch, grid_touch = tune_beta_identification(banks["val_touch"], "emb", sent_cohort_touch, behavior_model)
    beta_pw, grid_pw = tune_beta_password(cmu_val_args, behavior_model)
    betas = {"password": beta_pw, "sentence_keyboard": beta_kb, "sentence_touch": beta_touch}
    beta_grid = {"password": grid_pw, "sentence_keyboard": grid_kb, "sentence_touch": grid_touch}
    print(f"[Behavior] fusion weights chosen on Validation: {betas} | grids={beta_grid}")

    # Password 1:1 with behaviour forced on at the keyboard-sentence weight (ablation) and at
    # the calibrated password weight (what the web uses)
    yv_f, sv_f = score_trials(**cmu_val_args, behavior_model=behavior_model, beta=beta_kb)
    yt_f, st_f = score_trials(**cmu_test_args, behavior_model=behavior_model, beta=beta_kb)
    metrics_keyprint_fused_forced = summarize_metrics(yt_f, st_f, find_optimal_val_threshold(yv_f, sv_f))
    yv_f, sv_f = score_trials(**cmu_val_args, behavior_model=behavior_model, beta=beta_pw)
    thresh_f = find_optimal_val_threshold(yv_f, sv_f)
    yt_f, st_f = score_trials(**cmu_test_args, behavior_model=behavior_model, beta=beta_pw)
    metrics_keyprint_fused = summarize_metrics(yt_f, st_f, thresh_f)
    print(f"[Result 1:1] KeyPrintNet + behavior(beta={beta_pw}) Test: {metrics_keyprint_fused}")
    print(f"[Result 1:1] KeyPrintNet + behavior(forced beta={beta_kb}) Test: {metrics_keyprint_fused_forced}")

    # Password on a touch keyboard (holds unknown) — 9ch vs 17ch, threshold calibrated strictly on Validation
    def _touch_trials_metrics(model, in_ch, cohort_touch):
        out = {}
        for name, split, trials in (("val", splits["val"], val_trials), ("test", splits["test"], test_trials)):
            h = split.raw_holds
            d = split.raw_dds
            emb = encode_sequences(model, _touch_seq(h, d, split.key_block)[:, :in_ch], device)
            bh, bd = flat_to_hdu(trials["bot_flat"])
            bemb = encode_sequences(model, _touch_seq(bh.astype(np.float32), bd.astype(np.float32), split.key_block)[:, :in_ch], device)
            out[name] = score_trials(emb, bemb, trials, cohort_touch)
        thr_t = find_optimal_val_threshold(*out["val"])
        return summarize_metrics(*out["val"], thr_t), summarize_metrics(*out["test"], thr_t)

    cohort_l_touch = compute_cohort_centroids(
        encode_sequences(legacy_model, _touch_seq(cmu_tr_h, cmu_tr_d, splits["train"].key_block)[:, :L9], device),
        splits["train"].subject_ids,
    )
    val_touch_9ch, test_touch_9ch = _touch_trials_metrics(legacy_model, L9, cohort_l_touch)
    val_touch_17ch, test_touch_17ch = _touch_trials_metrics(keyprint_model, N_CHANNELS, cohort_k_touch)
    password_touch_val = {
        "legacy_9ch_imputed_holds": val_touch_9ch,
        "keyprint_17ch": val_touch_17ch,
    }
    password_touch = {
        "legacy_9ch_imputed_holds": test_touch_9ch,
        "keyprint_17ch": test_touch_17ch,
    }
    print(f"[Result 1:1 touch] val={password_touch_val} | test={password_touch}")

    # Strict 3-Split Separation: evaluate password transfer separately on Validation vs Test!
    enc_candidates = {
        "keyprint_17ch": (keyprint_model, N_CHANNELS, cohort_k, metrics_keyprint["threshold"]),
        "legacy_9ch": (legacy_model, L9, cohort_l, metrics_legacy["threshold"]),
    }
    password_transfer_val = eval_password_transfer(
        root / "data" / "raw" / "KeyRecs-fixed-text.csv",
        raw_sessions,
        kr_subjs["val"],
        bounds,
        chan_mean,
        chan_std,
        enc_candidates,
        device,
        seed=SEED,
    )
    password_transfer = eval_password_transfer(
        root / "data" / "raw" / "KeyRecs-fixed-text.csv",
        raw_sessions,
        kr_subjs["test"],
        bounds,
        chan_mean,
        chan_std,
        enc_candidates,
        device,
        seed=SEED,
    )
    # Choose PASSWORD_ENCODER strictly from Validation (unseen-content FAR drift + Validation touch EER)
    val_rw = password_transfer_val.get("repeated_words_unseen", {})
    val_cost_9ch = (
        val_rw.get("legacy_9ch", {}).get("far_at_deployed_threshold", 0.0)
        + password_touch_val["legacy_9ch_imputed_holds"]["eer"]
    )
    val_cost_17ch = (
        val_rw.get("keyprint_17ch", {}).get("far_at_deployed_threshold", 0.0)
        + password_touch_val["keyprint_17ch"]["eer"]
    )
    PASSWORD_ENCODER = "legacy_9ch" if val_cost_9ch <= val_cost_17ch else "keyprint_17ch"
    password_encoder_selection = {
        "selection_split": "val_only",
        "val_cost_legacy_9ch": round(float(val_cost_9ch), 4),
        "val_cost_keyprint_17ch": round(float(val_cost_17ch), 4),
        "chosen_encoder": PASSWORD_ENCODER,
    }
    print(f"[Password transfer] val={password_transfer_val} | test={password_transfer} | selection={password_encoder_selection}")

    open_set = calibrate_open_set_threshold(banks["val"], "emb", sent_cohort, behavior_model, beta_kb)
    open_set_touch = calibrate_open_set_threshold(banks["val_touch"], "emb", sent_cohort_touch, behavior_model, beta_touch)
    print(f"[Arena] open-set threshold on Validation: keyboard={open_set} touch={open_set_touch}")

    # 6c. Deployed protocols that reject more impostors (README section 5.6)
    pw_rng = np.random.default_rng(SEED)
    pw_tabs = digraph_tables(raw_sessions, kr_subjs["val"] + kr_subjs["test"])
    pw_people = {sp: [p for p in kr_subjs[sp] if (p, 1) in pw_tabs and (p, 2) in pw_tabs] for sp in ("val", "test")}
    non_test_vocab = kr_subjs["train"] + kr_subjs["val"]
    pw_strings = {
        "val": choose_strings(raw_sessions, pw_tabs, pw_people["val"], 30, 30, pw_rng, vocab_people=non_test_vocab)
    }
    pw_strings["test"] = choose_strings(
        raw_sessions,
        pw_tabs,
        pw_people["test"],
        30,
        30,
        pw_rng,
        exclude=set(pw_strings["val"]["word"] + pw_strings["val"]["random"]),
        vocab_people=non_test_vocab,
    )
    emb9 = lambda T: encode_variable(legacy_model, list(T), device, L9)  # noqa: E731
    emb17 = lambda T: encode_variable(keyprint_model, list(T), device, N_CHANNELS)  # noqa: E731
    pw_cohorts = {"kb": cohort_l, "touch": cohort_l_touch}
    pw_deployed_thr = {"kb": metrics_legacy["threshold"], "touch": password_touch["legacy_9ch_imputed_holds"]["threshold"]}
    pw_rows = {
        sp: string_trials(
            raw_sessions, pw_tabs, pw_people[sp], pw_strings[sp], emb9, emb17, bounds, chan_mean, chan_std,
            touch_hold, pw_cohorts, pw_rng
        )
        for sp in ("val", "test")
    }
    pw_synth_meta = {
        sp: synthesis_metadata(pw_tabs, pw_people[sp], pw_strings[sp], pw_rows[sp])
        for sp in ("val", "test")
    }
    pw_dual = calibrate_dual(pw_rows["val"], PASSWORD_FRR_TARGET)
    pw_speed = person_speed(pw_tabs, pw_people["val"] + pw_people["test"])
    pw_string_eval = evaluate_strings(pw_rows["test"], pw_dual, pw_deployed_thr, pw_speed)
    print(f"[Password strings] dual calibration {pw_dual} | test {pw_string_eval} | meta={pw_synth_meta}")

    # (a2) Length-aware calibration fitted strictly on Validation strings
    used = set(pw_strings["val"]["word"] + pw_strings["val"]["random"] + pw_strings["test"]["word"] + pw_strings["test"]["random"])
    pw_len_strings = {"val": {}, "test": {}}
    pw_len_rows = {"val": {}, "test": {}}
    for sp, lengths in (("val", CALIB_LENGTHS), ("test", (8, 12, 16, 20))):
        for L in lengths:
            ws = strings_of_length(pw_tabs, pw_people[sp], L, 15, pw_rng, exclude=used)
            used |= set(ws)
            pw_len_strings[sp][L] = ws
            pw_len_rows[sp][L] = string_trials(
                raw_sessions, pw_tabs, pw_people[sp], {"random": ws}, emb9, emb17, bounds, chan_mean,
                chan_std, touch_hold, pw_cohorts, pw_rng
            )
    pw_len_table = calibrate_by_length(pw_len_rows["val"], PASSWORD_FRR_TARGET)
    pw_len_eval = evaluate_lengths(pw_len_rows["test"], pw_dual, pw_len_table)
    print(f"[Password length] table {pw_len_table} | test {pw_len_eval}")

    # (a3) Capital letters: Shift vs CapsLock habit evaluated at exact deployed `base_tau + tau_offset`
    cap_rates = capital_method_rates(raw_sessions)
    caps_share = float(np.mean([cap_rates[(p, 1)] for p in kr_subjs["train"] if (p, 1) in cap_rates]))
    prm12 = params_for_length(pw_len_table, PASSWORD_MIN_LENGTH, "kb")
    cap_rows = {
        sp: capital_trials(
            pw_tabs, cap_rates, pw_people[sp], PASSWORD_MIN_LENGTH, n, emb9, emb17, bounds, chan_mean, chan_std,
            touch_hold, prm12, pw_rng
        )
        for sp, n in (("val", 12), ("test", 15))
    }
    pw_capital = capital_fusion(cap_rows["val"], cap_rows["test"], caps_share, base_tau=prm12["tau"])
    both_days = [p for p in kr_subjs["train"] + kr_subjs["val"] + kr_subjs["test"] if (p, 1) in cap_rates and (p, 2) in cap_rates]
    pw_capital["habit_same_across_days"] = round(float(np.mean([(cap_rates[(p, 1)] > 0.5) == (cap_rates[(p, 2)] > 0.5) for p in both_days])), 4)
    pw_capital["people_mostly_capslock"] = int(sum(1 for p in both_days if cap_rates[(p, 1)] > 0.5))
    pw_capital["people_with_capitals"] = len(both_days)
    print(f"[Password capitals] {pw_capital}")

    # (b) Template adaptation: pass UNCLIPPED raw holds & DD intervals so raw UD is computed before clipping!
    kf = pd.read_csv(root / "data" / "raw" / "KeyRecs-fixed-text.csv")
    kf.columns = [c.strip() for c in kf.columns]
    kf = kf.dropna(subset=KEYRECS_HOLD_COLS + KEYRECS_DD_COLS).reset_index(drop=True)
    kf["rep"] = kf.groupby(["participant", "session"]).cumcount()
    adapt_margins = (0.0, 0.5, 1.0, 1.5)
    pw_adapt = {}
    for sp in ("val", "test"):
        sub = kf[kf["participant"].isin(kr_subjs[sp])].reset_index(drop=True)
        raw_hh = sub[KEYRECS_HOLD_COLS].values.astype(np.float32)
        raw_dd = sub[KEYRECS_DD_COLS].values.astype(np.float32)
        pw_adapt[sp] = {
            mode: adaptation_eval(
                raw_hh, raw_dd, sub["participant"].values, sub["session"].values, sub["rep"].values, "vpwjkeurkb",
                emb9, emb17, bounds, chan_mean, chan_std, touch_hold, pw_dual, pw_deployed_thr, pw_cohorts,
                adapt_margins, np.random.default_rng(SEED), mode
            )
            for mode in ("kb", "touch")
        }

    def _adapt_cost(r):
        frr = np.mean([v for k, v in r.items() if k.startswith("frr_")])
        return 0.5 * (frr + r["far"])

    adapt_margin = min(adapt_margins, key=lambda mg: sum(_adapt_cost(pw_adapt["val"][md][f"dual_adapt_margin_{mg:g}"]) for md in ("kb", "touch")))
    print(f"[Password adaptation] margin chosen on Validation = {adapt_margin} | test {pw_adapt['test']}")

    # (c) Typing practice: 5 enrolment sentences, 2 query sentences, threshold + top1-top2 margin
    arena_protocol = {}
    for dev, vb, tb, coh, bta, old_thr in (
        ("keyboard", banks["val"], banks["test"], sent_cohort, beta_kb, open_set["threshold"]),
        ("touch", banks["val_touch"], banks["test_touch"], sent_cohort_touch, beta_touch, open_set_touch["threshold"]),
    ):
        mats = {
            (sp, k, q): open_set_matrix(b, "emb", coh, behavior_model, bta, k, q)
            for sp, b in (("val", vb), ("test", tb))
            for k, q in ((3, 1), (ARENA_ENROLL_SENTENCES, ARENA_QUERY_SENTENCES))
        }
        old_val = open_set_eval(mats[("val", 3, 1)], 5, old_thr, 0.0)
        rule = tune_open_set_rule(mats[("val", ARENA_ENROLL_SENTENCES, ARENA_QUERY_SENTENCES)], 5, old_val["correct_accept"])
        arena_protocol[dev] = {"rule": rule, "validation_target_correct_accept": old_val["correct_accept"]}
        for g in (3, 5, 10):
            arena_protocol[dev][f"gallery_{g}"] = {
                "single_sentence_enroll3": open_set_eval(mats[("test", 3, 1)], g, old_thr, 0.0),
                "two_sentences_enroll5_margin": open_set_eval(
                    mats[("test", ARENA_ENROLL_SENTENCES, ARENA_QUERY_SENTENCES)], g, rule["threshold"], rule["margin"]
                ),
            }
    print(f"[Arena protocol] {arena_protocol}")

    # (d) Extra practice on an existing profile: 1:1 gate calibrated on Validation
    arena_add = {
        "keyboard": sentence_add_check(banks["val"], banks["test"], "emb", sent_cohort, behavior_model, beta_kb, ARENA_ENROLL_SENTENCES),
        "touch": sentence_add_check(banks["val_touch"], banks["test_touch"], "emb", sent_cohort_touch, behavior_model, beta_touch,
                                    ARENA_ENROLL_SENTENCES),
    }
    print(f"[Arena add check] {arena_add}")

    web_identification = {
        "keyboard": {
            "legacy_9ch": eval_web_identification(banks["test"], "emb_legacy", sent_cohort_legacy),
            "keyprint_17ch": eval_web_identification(banks["test"], "emb", sent_cohort),
            "keyprint_17ch_behavior": eval_web_identification(banks["test"], "emb", sent_cohort, 16, behavior_model, beta_kb),
            "behavior_only": eval_web_identification(banks["test"], "emb", sent_cohort, 16, behavior_model, 1.0, behavior_only=True),
        },
        "touch": {
            "legacy_9ch_imputed_holds": eval_web_identification(banks["test_touch"], "emb_legacy", sent_cohort_legacy_touch),
            "keyprint_17ch": eval_web_identification(banks["test_touch"], "emb", sent_cohort_touch),
            "keyprint_17ch_behavior": eval_web_identification(banks["test_touch"], "emb", sent_cohort_touch, 16, behavior_model, beta_touch),
            "keyprint_17ch_behavior_keyboard_beta": eval_web_identification(banks["test_touch"], "emb", sent_cohort_touch, 16, behavior_model, beta_kb),
        },
    }
    print(f"[Result 1:N web-faithful captures] {web_identification}")

    troubleshooting_ablations = compute_troubleshooting_ablations(
        splits["test"],
        test_trials,
        bounds,
        chan_mean,
        chan_std,
        touch_hold,
        legacy_model,
        keyprint_model,
        cohort_l,
        sent_cohort,
        behavior_model,
        beta_kb,
        banks["test"],
        device,
    )
    print(f"[Troubleshooting Ablations] {troubleshooting_ablations}")

    # 7. Export ONNX FIRST so we compute sha17, sha9, and model_hash before stamping web profiles & config!
    onnx_info = safe_export_onnx(
        keyprint_model,
        models_dir / "keyprint_net.onnx",
        web_models_dir / "keyprint_net.onnx",
        in_channels=N_CHANNELS,
    )
    keyprint_model.to(device)
    password_onnx_info = safe_export_onnx(
        legacy_model,
        models_dir / "legacy" / "keyprint_net_9ch.onnx",
        web_models_dir / "keyprint_net_password.onnx",
        in_channels=L9,
    )
    legacy_model.to(device)

    sha17 = _file_sha256(web_models_dir / "keyprint_net.onnx")
    sha9 = _file_sha256(web_models_dir / "keyprint_net_password.onnx")
    model_hash = hashlib.sha256(f"{sha17}:{sha9}".encode("utf-8")).hexdigest()[:16]
    onnx_info["sha256"] = sha17
    password_onnx_info["sha256"] = sha9

    # Real sample profiles for the web app (stamped with schema v4 + modelHash)
    def _pick(cands: Dict[str, Dict[str, float]]) -> List[str]:
        chosen: List[str] = []
        rules = [("cpm", max), ("cpm", "median"), ("cv", min), ("rollover", max)]
        for key, how in rules:
            pool = {k: v for k, v in cands.items() if k not in chosen}
            if how == "median":
                med = float(np.median([v["cpm"] for v in pool.values()]))
                chosen.append(min(pool, key=lambda k: abs(pool[k]["cpm"] - med)))
            else:
                chosen.append(how(pool, key=lambda k: pool[k][key]))
        return chosen  # order: fast, balanced, steady, rollover

    kr_cands = {}
    for pid, sess in banks["test"].items():
        if len(sess[1]) >= ARENA_ENROLL_SENTENCES + 1 and len(sess[2]) >= 2:
            st = [capture_stats(c) for c in sess[1][:ARENA_ENROLL_SENTENCES]]
            kr_cands[pid] = {
                "cpm": float(np.mean([x["cpm"] for x in st])),
                "rollover": float(np.mean([x["rolloverPct"] for x in st])),
                "cv": dd_cv(sess[1][:ARENA_ENROLL_SENTENCES]),
            }
    test_df = splits["test"].df
    cmu_cands, cmu_rows = {}, {}
    for uid, subj in enumerate(splits["test"].subjects):
        idx = np.where((splits["test"].subject_ids == uid) & (splits["test"].sessions == 1) & (splits["test"].reps <= 5))[0]
        g_idx = np.where((splits["test"].subject_ids == uid) & (splits["test"].sessions == 6) & (splits["test"].reps == 10))[0]
        if len(idx) < 5 or len(g_idx) == 0:
            continue
        caps = [
            fixed_stroke_capture(test_df.iloc[i][hold_cols_b], test_df.iloc[i][dd_cols_b], test_df.iloc[i][ud_cols_b], CMU_PASSWORD_CODES)
            for i in idx
        ]
        st = [capture_stats(c) for c in caps]
        cmu_cands[subj] = {
            "cpm": float(np.mean([x["cpm"] for x in st])),
            "rollover": float(np.mean([x["rolloverPct"] for x in st])),
            "cv": dd_cv(caps),
        }
        cmu_rows[subj] = (idx, int(g_idx[0]), caps)

    sample_meta = [
        ("서연 (샘플·고속연타)", "#2563eb"),
        ("민준 (샘플·균형리듬)", "#059669"),
        ("지우 (샘플·정박타건)", "#d97706"),
        ("현우 (샘플·롤오버형)", "#7c3aed"),
    ]
    mvp_sample_profiles = []
    for (name, color), kr_id, cmu_id in zip(sample_meta, _pick(kr_cands), _pick(cmu_cands)):
        e_caps = banks["test"][kr_id][1][:ARENA_ENROLL_SENTENCES]
        beh = None
        for c in e_caps:
            beh = merge(beh, c["behavior"])
        arena_tmpl = template_payload(np.stack([c["emb"] for c in e_caps]), e_caps, beh)
        q_cap = banks["test"][kr_id][2][1]
        arena_tmpl["testQueryStroke"] = {**stroke_payload(q_cap), "behavior": q_cap["behavior"]}

        idx, g_i, caps = cmu_rows[cmu_id]
        pw_beh = None
        for i in idx:
            pw_beh = merge(pw_beh, test_stats_cmu[i])
        pw_tmpl = template_payload(test_emb_l[idx], caps, pw_beh)
        pw_tmpl["target"] = ".tie5Roanl"
        t17 = template_payload(test_emb_k[idx], caps, pw_beh)
        pw_tmpl["enc17"] = {"centroid": t17["centroid"], "exemplars": t17["exemplars"]}
        g_row = test_df.iloc[g_i]
        pw_tmpl["testGenuineStroke"] = {
            **stroke_payload(fixed_stroke_capture(g_row[hold_cols_b], g_row[dd_cols_b], g_row[ud_cols_b], CMU_PASSWORD_CODES)),
            "behavior": test_stats_cmu[g_i],
        }
        mvp_sample_profiles.append(
            {
                "id": f"sample_{kr_id}",
                "name": name,
                "color": color,
                "isSample": True,
                "schemaVersion": SCHEMA_VERSION,
                "preprocessVersion": PREPROCESS_VERSION,
                "modelHash": model_hash,
                "modelVersion": MODEL_VERSION,
                "createdAt": SAMPLE_PROFILE_TIMESTAMP,
                "updatedAt": SAMPLE_PROFILE_TIMESTAMP,
                "source": {"keyrecs_participant": kr_id, "cmu_subject": cmu_id},
                "password": pw_tmpl,
                "arena": {"en": arena_tmpl},
            }
        )
    print(f"[Web] rebuilt sample profiles (modelHash={model_hash}): {[(p['name'], p['source']) for p in mvp_sample_profiles]}")

    # JS <-> Python parity fixture (keyboard + touch replays of real typo-laden segments)
    fx_raw = raw_sessions[(kr_subjs["val"][0], 1)]
    export_parity_fixture(
        parity_fixture_path,
        [fx_raw[0:150], fx_raw[600:760], fx_raw[1500:1640]],
        bounds,
        touch_hold,
        chan_mean,
        chan_std,
        behavior_model,
        beta_kb,
        dual_params=pw_dual,
        display_slope=DISPLAY_SLOPE,
        extra={
            "length_cases": [
                {"length": L, "mode": md, "expected": params_for_length(pw_len_table, L, md)}
                for L in (6, 9, 12, 13, 17, 20, 30)
                for md in ("kb", "touch")
            ],
            "capital_cases": [
                {
                    "q_caps": qc,
                    "n_caps": nc,
                    "n_total": nt,
                    "caps_share": pw_capital["caps_share"],
                    "llr": capital_llr(qc, nc, nt, pw_capital["caps_share"]),
                }
                for qc in (True, False)
                for nc, nt in ((0, 4), (4, 4), (2, 4), (1, 9))
            ],
        },
    )

    feature_upgrade = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "model_hash": model_hash,
        "channels": CHANNEL_NAMES,
        "touch_hold_impute_sec": touch_hold,
        "touch_train_prob": TOUCH_PROB,
        "behavior_beta": betas,
        "behavior_beta_grid": beta_grid,
        "behavior_params": behavior_model.to_json(),
        "password_1to1": {
            "legacy_9ch": metrics_legacy,
            "keyprint_17ch": metrics_keyprint,
            "keyprint_17ch_behavior": metrics_keyprint_fused,
            "keyprint_17ch_behavior_forced_sentence_beta": metrics_keyprint_fused_forced,
        },
        "password_1to1_touch_val": password_touch_val,
        "password_1to1_touch": password_touch,
        "password_transfer_val": password_transfer_val,
        "password_transfer_other_content": password_transfer,
        "password_encoder_selection": password_encoder_selection,
        "password_encoder_deployed": PASSWORD_ENCODER,
        "cmu_10way": {"legacy_9ch": id_cmu_l, "keyprint_17ch": id_cmu_k},
        "keyrecs_clean_chunks_18way": {
            "legacy_9ch": sent_id_l,
            "keyprint_17ch": {
                "same_session": sent_id_k_same,
                "cross_session_different_essay": sent_id_k_cross,
            },
        },
        "web_faithful_identification": web_identification,
        "arena_open_set": {"keyboard": open_set, "touch": open_set_touch},
        "web_capture_chars": WEB_CAPTURE_CHARS,
        "password_user_strings": {
            "strings": pw_strings,
            "synthesis_metadata": pw_synth_meta,
            "frr_target": PASSWORD_FRR_TARGET,
            "dual_calibration": pw_dual,
            "test": pw_string_eval,
        },
        "password_adaptation": {"margins": list(adapt_margins), "chosen_margin": adapt_margin, **pw_adapt},
        "password_capital_method": pw_capital,
        "password_length_calibration": {
            "min_length": PASSWORD_MIN_LENGTH,
            "table": pw_len_table,
            "strings": {sp: {str(L): v for L, v in d.items()} for sp, d in pw_len_strings.items()},
            "test": pw_len_eval,
        },
        "arena_protocol": {"enroll_sentences": ARENA_ENROLL_SENTENCES, "query_sentences": ARENA_QUERY_SENTENCES, **arena_protocol},
        "arena_add_check": arena_add,
    }

    identification_bundle = {
        "cmu_10way": id_comparison_cmu,
        "keyrecs_18way_sentence": sent_id_summary,
        "sentence_length_curve": sent_len_curve,
    }

    def _r5(mat):
        return [[round(float(v), 5) for v in row] for row in mat]

    export_web_demo_bundle(
        splits["test"],
        sent_test,
        meta,
        cohort_l,
        metrics_legacy["threshold"],
        metrics_legacy["eer"],
        metrics_legacy["macro_f1"],
        metrics_legacy["roc_auc"],
        identification_bundle,
        web_models_dir,
        extra_payload={
            "schema_version": SCHEMA_VERSION,
            "preprocess_version": PREPROCESS_VERSION,
            "model_hash": model_hash,
            "onnx_sha256": {
                "sentence_17ch": sha17,
                "password_9ch": sha9,
            },
            "touch_hold_impute_sec": touch_hold,
            "keyboard_layout": export_layout_for_web(),
            "sentence_cohort_centroids": _r5(sent_cohort),
            "cohort_centroids_touch": _r5(cohort_l_touch),
            "password_model": {
                "file": "keyprint_net_password.onnx",
                "in_channels": L9,
                "version": "kp9-timing-v4",
                "note": "content-agnostic timing encoder trained from scratch for user-chosen passwords",
            },
            "sentence_cohort_centroids_touch": _r5(sent_cohort_touch),
            "behavior_profile": {**behavior_model.to_json(), "beta": betas},
            "arena_open_set_threshold": arena_protocol["keyboard"]["rule"]["threshold"],
            "arena_open_set_threshold_touch": arena_protocol["touch"]["rule"]["threshold"],
            "arena_margin": arena_protocol["keyboard"]["rule"]["margin"],
            "arena_margin_touch": arena_protocol["touch"]["rule"]["margin"],
            "arena_enroll_sentences": ARENA_ENROLL_SENTENCES,
            "arena_add_threshold": arena_add["keyboard"]["threshold"],
            "arena_add_threshold_touch": arena_add["touch"]["threshold"],
            "arena_query_sentences": ARENA_QUERY_SENTENCES,
            "password_dual": {
                "keyboard": pw_dual["kb"],
                "touch": pw_dual["touch"],
                "slope": DISPLAY_SLOPE,
                "adapt_margin": adapt_margin,
                "adapt_keep": 12,
                "frr_target": PASSWORD_FRR_TARGET,
                "touch_reference_only": True,
                "by_length": pw_len_table,
                "capital_method": {
                    "weight": pw_capital["weight"],
                    "tau_offset": pw_capital["tau_offset"],
                    "caps_share": pw_capital["caps_share"],
                    "alpha": 2.0,
                },
            },
            "password_min_length": PASSWORD_MIN_LENGTH,
            "calibrated_threshold_touch": password_touch["legacy_9ch_imputed_holds"]["threshold"],
            "mvp_sample_profiles": mvp_sample_profiles,
            "feature_metrics": {
                "web_faithful_identification": web_identification,
                "password_1to1": feature_upgrade["password_1to1"],
                "password_1to1_touch": password_touch,
                "password_transfer_val": password_transfer_val,
                "password_transfer_other_content": password_transfer,
                "password_encoder_deployed": PASSWORD_ENCODER,
            },
        },
    )

    # 8. Generate All 8 Analytical Figures
    roc_data = {
        "ScaledManhattan (통계 베이스라인)": (yt_m, st_m, metrics_manhattan),
        "Vanilla1DCNN (딥러닝 베이스라인)": (yt_v, st_v, metrics_vanilla),
        "KeyPrintNet (제안 모델)": (yt_k, st_k, metrics_keyprint),
    }
    plot_fig1_preprocessing(raw_df, splits, meta, sil_raw4, sil_bio9, fig_dir)
    plot_fig2_learning_curves(hist_vanilla, hist_keyprint, fig_dir, hist_keyprint_9ch=hist_keyprint_9ch)
    plot_fig3_roc_det_kshot(roc_data, kshot_results, fig_dir)
    plot_fig4_confusion_matrices(roc_data, fig_dir)
    plot_fig5_tsne_embeddings(splits["test"], test_emb_v, test_emb_k, test_bot_v, test_bot_k, fig_dir)
    error_summary = plot_fig6_error_analysis(
        splits["test"], detailed_k, metrics_keyprint["threshold"], ablation_summary, fig_dir
    )
    plot_fig7_typist_identification(
        splits["test"].subjects,
        cm_10way,
        id_comparison_cmu,
        sent_len_curve,
        fig_dir,
    )
    plot_fig8_feature_upgrade(web_identification, feature_upgrade["password_1to1"], password_touch, password_transfer, fig_dir)

    now_utc = datetime.now(timezone.utc)
    provenance = {
        "run_id": f"kp-v4-{now_utc.strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": now_utc.isoformat(),
        "git_head": _get_git_head(root),
        "command": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "onnxruntime_version": ort.__version__,
        "seed": SEED,
        "epochs": {
            "vanilla_1dcnn": VANILLA_EPOCHS,
            "vanilla_1dcnn_aug": VANILLA_EPOCHS,
            "keyprint_net_9ch": KEYPRINT_EPOCHS,
            "keyprint_net_17ch": KEYPRINT_EPOCHS,
        },
        "reuse_weights": REUSE_WEIGHTS,
        "schema_version": SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "model_hash": model_hash,
        "disclosure": (
            "Selection rules strictly use Validation; test split was previously evaluated in earlier iterations "
            "(re-evaluated after leakage/preprocessing fixes); 9ch and 17ch checkpoints are newly trained from scratch."
        ),
        "artifact_sha256": {
            "models/vanilla_1dcnn.pt": _file_sha256(vanilla_pt),
            "models/vanilla_1dcnn_aug.pt": _file_sha256(vanilla_aug_pt),
            "models/legacy/keyprint_net_9ch.pt": _file_sha256(kp9_pt),
            "models/keyprint_net.pt": _file_sha256(kp17_pt),
            "models/keyprint_net.onnx": _file_sha256(models_dir / "keyprint_net.onnx"),
            "models/legacy/keyprint_net_9ch.onnx": _file_sha256(models_dir / "legacy" / "keyprint_net_9ch.onnx"),
            "web/public/models/keyprint_net.onnx": sha17,
            "web/public/models/keyprint_net_password.onnx": sha9,
            "web/public/models/model_config.json": _file_sha256(web_models_dir / "model_config.json"),
            "web/scripts/fixtures/feature_parity.json": _file_sha256(parity_fixture_path),
        },
    }

    elapsed = round(time.time() - t0, 2)
    report = {
        "elapsed_seconds": elapsed,
        "device": f"{device} ({gpu_name})",
        "provenance": provenance,
        "preprocessing": {
            **meta["preprocessing_before_after"],
            "silhouette_raw4_cosine": round(sil_raw4, 4),
            "silhouette_clipped4_unscaled_cosine": round(sil_clipped4_unscaled, 4),
            "silhouette_clipped4_scaled_cosine": round(sil_clipped4_scaled, 4),
            "silhouette_bio9_cosine": round(sil_bio9, 4),
            "silhouette_bio17_cosine": round(sil_bio17, 4),
            "split_counts": meta["split_counts"],
            "keyrecs_sentence_corpus": meta.get("keyrecs_sentence_corpus"),
        },
        "models": {
            "ScaledManhattan": metrics_manhattan,
            "Vanilla1DCNN": metrics_vanilla,
            "Vanilla1DCNN_Aug": metrics_vanilla_aug,
            "Vanilla1DCNN_Aug_ASNorm": metrics_vanilla_aug_as,
            "KeyPrintNet_9ch": metrics_legacy,
            "KeyPrintNet": metrics_keyprint,
        },
        "typist_identification_1toN": identification_bundle,
        "ablation_session1_drift": ablation_summary,
        "troubleshooting_ablations": troubleshooting_ablations,
        "kshot_sensitivity": kshot_results,
        "variable_length_generalization": varlen_results,
        "onnx_export": onnx_info,
        "onnx_export_password_encoder": password_onnx_info,
        "feature_upgrade": feature_upgrade,
        "error_analysis": error_summary,
        "histories": {
            "Vanilla1DCNN": hist_vanilla,
            "Vanilla1DCNN_Aug": hist_vanilla_aug,
            "KeyPrintNet_9ch": hist_keyprint_9ch,
            "KeyPrintNet": hist_keyprint,
        },
    }
    with open(models_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"=== Pipeline Complete in {elapsed}s! 8 Figures & ONNX saved (model_hash={model_hash}). ===")


if __name__ == "__main__":
    main()
