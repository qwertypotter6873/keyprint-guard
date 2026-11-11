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

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, silhouette_score

from models import KeyPrintNet, ScaledManhattanDetector, Vanilla1DCNN
from preprocessing import (
    CHANNEL_NAMES,
    DIGRAPH_LABELS,
    SEED,
    SentenceSplitData,
    get_feature_columns,
    prepare_full_pipeline,
)
from train_and_evaluate import (
    build_verification_trials,
    compute_cohort_centroids,
    configure_korean_matplotlib,
    encode_sequences,
    evaluate_1toN_identification_embeddings,
    evaluate_1toN_identification_manhattan,
    find_optimal_val_threshold,
    get_device,
    score_embedding_trials,
    score_manhattan_trials,
    set_all_seeds,
    summarize_metrics,
    train_keyprint_net,
    train_vanilla_baseline,
)


def safe_export_onnx(model: KeyPrintNet, onnx_path: Path, web_onnx_path: Path) -> Dict:
    cpu_model = KeyPrintNet(in_channels=9, hidden_channels=80, embed_dim=64, dropout=0.0)
    cpu_model.load_state_dict(model.cpu().state_dict())
    cpu_model.eval()

    dummy = torch.randn(1, 9, 10, dtype=torch.float32)
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
        x_test = np.random.RandomState(seq_len).randn(2, 9, seq_len).astype(np.float32)
        with torch.no_grad():
            pt_out = cpu_model(torch.from_numpy(x_test)).numpy()
        ort_out = sess.run(["biometric_embedding"], {"keystroke_seq": x_test})[0]
        max_diffs[f"L={seq_len}"] = float(np.max(np.abs(pt_out - ort_out)))

    size_kb = round(onnx_path.stat().st_size / 1024.0, 2)
    print(f"[ONNX] Exported {onnx_path.name} ({size_kb} KB) | Parity={max_diffs}")
    return {"onnx_size_kb": size_kb, "max_abs_diff_by_length": max_diffs}


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
    if kr_m:
        labels = [
            f"Train\n(103명 · {meta['split_counts']['train_samples'] + kr_m['train_sentence_chunks'] + kr_m['train_fixed_strokes']:,}건)",
            f"Validation\n(16명 · {meta['split_counts']['val_samples'] + kr_m['val_sentence_chunks']:,}건)",
            f"Unseen Test\n(28명 · {meta['split_counts']['test_samples'] + kr_m['test_sentence_chunks']:,}건)",
        ]
        counts = [
            meta["split_counts"]["train_samples"] + kr_m["train_sentence_chunks"] + kr_m["train_fixed_strokes"],
            meta["split_counts"]["val_samples"] + kr_m["val_sentence_chunks"],
            meta["split_counts"]["test_samples"] + kr_m["test_sentence_chunks"],
        ]
    else:
        labels = ["Train\n(35명 · 70%)", "Validation\n(6명 · 12%)", "Unseen Test\n(10명 · 18%)"]
        counts = [
            meta["split_counts"]["train_samples"],
            meta["split_counts"]["val_samples"],
            meta["split_counts"]["test_samples"],
        ]

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
    raw_vals = raw_df[col].values
    clip_vals = splits["train"].df[col].values
    ax.hist(
        np.clip(raw_vals, 0, 2.5),
        bins=45,
        alpha=0.55,
        color="#EF4444",
        label=f"정제 전 (최대 {raw_vals.max():.1f}초, 왜도 {pd.Series(raw_vals).skew():.1f})",
        density=True,
    )
    ax.hist(
        clip_vals,
        bins=45,
        alpha=0.68,
        color="#2563EB",
        label=f"정제 후 (상한 {clip_vals.max():.2f}초, 왜도 {pd.Series(clip_vals).skew():.2f})",
        density=True,
    )
    ax.set_xlabel("키 간 전환 지연 시간 DD.i.e (초)")
    ax.set_ylabel("밀도 (Density)")
    ax.set_title("(b) 생리학적 이상치 정제 전·후 분포 비교", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[2]
    categories = ["전체 컬럼 평균 왜도\n(낮을수록 정규성 우수)", "피험자 군집 응집도×10\n(Silhouette, 높을수록 우수)"]
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
    ax.set_ylim(0, max(before_metrics) * 1.22)
    ax.set_title("(c) 9채널 템포 정규화 전·후 정량 지표 개선", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_preprocessing_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_learning_curves(
    hist_vanilla: List[Dict], hist_keyprint: List[Dict], fig_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ev = [r["epoch"] for r in hist_vanilla]
    ek = [r["epoch"] for r in hist_keyprint]

    axes[0].plot(ev, [r["train_loss"] for r in hist_vanilla], "--", color="#64748B", lw=2.0, label="Vanilla1DCNN (CE Loss)")
    axes[0].plot(ek, [r["train_loss"] for r in hist_keyprint], "-", color="#2563EB", lw=2.3, label="KeyPrintNet (CosFace+SupCon)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("(a) 에포크별 학습 손실(Loss) 수렴 곡선", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].plot(ev, [r["val_eer"] * 100 for r in hist_vanilla], "--o", markersize=3, color="#EF4444", lw=1.9, label="Vanilla1DCNN (Baseline)")
    axes[1].plot(ek, [r["val_eer"] * 100 for r in hist_keyprint], "-o", markersize=3, color="#10B981", lw=2.2, label="KeyPrintNet (Proposed)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation EER (%) ↓ (낮을수록 우수)")
    axes[1].set_title("(b) 검증셋 동일오류율(Val EER) 변화", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(ev, [r["val_macro_f1"] for r in hist_vanilla], "--s", markersize=3, color="#F59E0B", lw=1.9, label="Vanilla1DCNN (Baseline)")
    axes[2].plot(ek, [r["val_macro_f1"] for r in hist_keyprint], "-s", markersize=3, color="#2563EB", lw=2.2, label="KeyPrintNet (Proposed)")
    axes[2].axhline(0.85, color="#DC2626", linestyle=":", lw=1.5, label="정량 목표 (Macro F1 ≥ 0.85)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation Macro F1 ↑ (높을수록 우수)")
    axes[2].set_title("(c) 검증셋 Macro F1-Score 향상 추이", fontsize=12, fontweight="bold")
    axes[2].legend(loc="lower right")
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
        "최종 제안: KeyPrintNet\n(9채널 비율·템포워핑·AS-Norm)",
    ]
    f1_vals = [
        ablation_summary["step1_raw_manhattan"]["macro_f1"],
        ablation_summary["step2_vanilla_cnn"]["macro_f1"],
        ablation_summary["step3_keyprint_proposed"]["macro_f1"],
    ]
    eer_vals = [
        ablation_summary["step1_raw_manhattan"]["eer"] * 100,
        ablation_summary["step2_vanilla_cnn"]["eer"] * 100,
        ablation_summary["step3_keyprint_proposed"]["eer"] * 100,
    ]
    x_ab = np.arange(len(labels))
    bars = ax.bar(x_ab, f1_vals, width=0.5, color=["#94A3B8", "#38BDF8", "#2563EB"], edgecolor="#1E293B")
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
    ax.set_xticklabels(labels, fontsize=9.2)
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
) -> None:
    """
    Export scaler parameters, AS-Norm cohort, calibrated threshold, Mode 1 (1:1 Guard)
    presets, AND Mode 2 (1:N Typing-Practice Who-Is-Typing Arena) real unseen galleries!
    """
    _, hold_cols, dd_cols, ud_cols = get_feature_columns()
    web_models_dir.mkdir(parents=True, exist_ok=True)

    def _row_to_stroke(row: pd.Series) -> Dict:
        return {
            "holds": [round(float(row[c]), 5) for c in hold_cols],
            "dds": [round(float(row[c]), 5) for c in dd_cols],
            "uds": [round(float(row[c]), 5) for c in ud_cols],
        }

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
                    "holds": [round(float(x), 5) for x in sentence_test.holds[i]],
                    "dds": [round(float(x), 5) for x in sentence_test.dds[i]],
                    "uds": [round(float(x), 5) for x in sentence_test.uds[i]],
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
                    "holds": [round(float(x), 5) for x in sentence_test.holds[outsider_idx]],
                    "dds": [round(float(x), 5) for x in sentence_test.dds[outsider_idx]],
                    "uds": [round(float(x), 5) for x in sentence_test.uds[outsider_idx]],
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
        "embedding_dim": 64,
        "in_channels": 9,
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
    csv_path = root / "data" / "raw" / "DSL-StrongPasswordData.csv"
    proc_dir = root / "data" / "processed"
    models_dir = root / "models"
    fig_dir = root / "assets" / "figures"
    web_models_dir = root / "web" / "public" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(csv_path)
    splits, meta = prepare_full_pipeline(csv_path, proc_dir, seed=SEED)
    sentence_splits: Dict[str, SentenceSplitData] = meta.pop("_sentence_splits", {})
    kr_fixed_train = meta.pop("_keyrecs_fixed_train", None)
    meta.pop("_keyrecs_runs", None)
    meta.pop("_keyrecs_splits_subjs", None)

    sent_train = sentence_splits.get("train")
    sent_val = sentence_splits.get("val")
    sent_test = sentence_splits.get("test")

    chan_mean = np.array(meta["channel_mean"], dtype=np.float32)
    chan_std = np.array(meta["channel_std"], dtype=np.float32)
    bounds = {k: tuple(v) for k, v in meta["outlier_bounds"].items()}

    test_raw4 = splits["test"].seq_unscaled[:, :4, :].reshape(len(splits["test"].df), -1)
    test_bio9 = splits["test"].seq_scaled.reshape(len(splits["test"].df), -1)
    sil_raw4 = float(silhouette_score(test_raw4, splits["test"].subject_ids, metric="cosine"))
    sil_bio9 = float(silhouette_score(test_bio9, splits["test"].subject_ids, metric="cosine"))
    print(f"[Preprocessing] Test Cosine Silhouette: Raw 4-Ch={sil_raw4:.4f} -> Normalized 9-Ch={sil_bio9:.4f}")

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
    print(f"[Result 1:1] ScaledManhattan Test: {metrics_manhattan}")

    # 2. Baseline 2: Scratch Vanilla 1D-CNN (Cross-Entropy)
    vanilla_model, hist_vanilla = train_vanilla_baseline(
        splits["train"],
        splits["val"],
        val_trials,
        device,
        epochs=35,
        batch_size=256,
        sentence_train=sent_train,
        kr_fixed_train=kr_fixed_train,
    )
    torch.save(vanilla_model.cpu().state_dict(), models_dir / "vanilla_1dcnn.pt")
    vanilla_model.to(device)

    val_emb_v = encode_sequences(vanilla_model, splits["val"].seq_scaled, device)
    val_bot_v = encode_sequences(vanilla_model, val_trials["bot_seq"], device)
    yv_v, sv_v, _ = score_embedding_trials(val_emb_v, val_bot_v, val_trials)
    thresh_v = find_optimal_val_threshold(yv_v, sv_v)

    test_emb_v = encode_sequences(vanilla_model, splits["test"].seq_scaled, device)
    test_bot_v = encode_sequences(vanilla_model, test_trials["bot_seq"], device)
    yt_v, st_v, _ = score_embedding_trials(test_emb_v, test_bot_v, test_trials)
    metrics_vanilla = summarize_metrics(yt_v, st_v, thresh_v)
    print(f"[Result 1:1] Vanilla1DCNN Test: {metrics_vanilla}")

    # 3. Proposed Model: KeyPrintNet (Multi-Corpus + Positional ResSE-1DCNN + ASP/DCT + SubCenter-CosFace + AS-Norm)
    keyprint_model, hist_keyprint = train_keyprint_net(
        splits["train"],
        splits["val"],
        val_trials,
        bounds,
        chan_mean,
        chan_std,
        device,
        epochs=35,
        batch_size=256,
        use_augmentation=True,
        raw_channels_only=False,
        sentence_train=sent_train,
        kr_fixed_train=kr_fixed_train,
        sentence_val=sent_val,
        tag="KeyPrintNet",
    )
    torch.save(keyprint_model.cpu().state_dict(), models_dir / "keyprint_net.pt")
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
    print(f"[Result 1:1] KeyPrintNet Test: {metrics_keyprint}")

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
    id_comparison_cmu = {
        "ScaledManhattan": id_cmu_m,
        "Vanilla1DCNN": id_cmu_v,
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
        )
        sent_id_summary = {
            "ScaledManhattan_same_session": sent_id_m,
            "Vanilla1DCNN_same_session": sent_id_v,
            "KeyPrintNet_same_session": sent_id_k_same,
            "KeyPrintNet_cross_session_different_essay": sent_id_k_cross,
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
    yvv_d, svv_d, _ = score_embedding_trials(val_emb_v, vb_v_d, drift_val_trials, cohort=None)
    ytv_d, stv_d, _ = score_embedding_trials(test_emb_v, tb_v_d, drift_test_trials, cohort=None)
    drift_vanilla = summarize_metrics(ytv_d, stv_d, find_optimal_val_threshold(yvv_d, svv_d))

    vb_k_d = encode_sequences(keyprint_model, drift_val_trials["bot_seq"], device)
    tb_k_d = encode_sequences(keyprint_model, drift_test_trials["bot_seq"], device)
    yvk_d, svk_d, _ = score_embedding_trials(val_emb_k, vb_k_d, drift_val_trials, cohort=cohort_k)
    ytk_d, stk_d, _ = score_embedding_trials(test_emb_k, tb_k_d, drift_test_trials, cohort=cohort_k)
    drift_keyprint = summarize_metrics(ytk_d, stk_d, find_optimal_val_threshold(yvk_d, svk_d))

    ablation_summary = {
        "step1_raw_manhattan": drift_manhattan,
        "step2_vanilla_cnn": drift_vanilla,
        "step3_keyprint_proposed": drift_keyprint,
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

    # 7. Export ONNX + Verify Numerical Parity + Export Web Assets
    onnx_info = safe_export_onnx(
        keyprint_model,
        models_dir / "keyprint_net.onnx",
        web_models_dir / "keyprint_net.onnx",
    )
    keyprint_model.to(device)

    identification_bundle = {
        "cmu_10way": id_comparison_cmu,
        "keyrecs_18way_sentence": sent_id_summary,
        "sentence_length_curve": sent_len_curve,
    }

    export_web_demo_bundle(
        splits["test"],
        sent_test,
        meta,
        cohort_k,
        metrics_keyprint["threshold"],
        metrics_keyprint["eer"],
        metrics_keyprint["macro_f1"],
        metrics_keyprint["roc_auc"],
        identification_bundle,
        web_models_dir,
    )

    # 8. Generate All 7 Analytical Figures
    roc_data = {
        "ScaledManhattan (통계 베이스라인)": (yt_m, st_m, metrics_manhattan),
        "Vanilla1DCNN (딥러닝 베이스라인)": (yt_v, st_v, metrics_vanilla),
        "KeyPrintNet (제안 모델)": (yt_k, st_k, metrics_keyprint),
    }
    plot_fig1_preprocessing(raw_df, splits, meta, sil_raw4, sil_bio9, fig_dir)
    plot_fig2_learning_curves(hist_vanilla, hist_keyprint, fig_dir)
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

    elapsed = round(time.time() - t0, 2)
    report = {
        "elapsed_seconds": elapsed,
        "device": f"{device} ({gpu_name})",
        "preprocessing": {
            **meta["preprocessing_before_after"],
            "silhouette_raw4_cosine": round(sil_raw4, 4),
            "silhouette_bio9_cosine": round(sil_bio9, 4),
            "keyrecs_sentence_corpus": meta.get("keyrecs_sentence_corpus"),
        },
        "models": {
            "ScaledManhattan": metrics_manhattan,
            "Vanilla1DCNN": metrics_vanilla,
            "KeyPrintNet": metrics_keyprint,
        },
        "typist_identification_1toN": identification_bundle,
        "ablation_session1_drift": ablation_summary,
        "kshot_sensitivity": kshot_results,
        "variable_length_generalization": varlen_results,
        "onnx_export": onnx_info,
        "error_analysis": error_summary,
        "histories": {
            "Vanilla1DCNN": hist_vanilla,
            "KeyPrintNet": hist_keyprint,
        },
    }
    with open(models_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"=== Pipeline Complete in {elapsed}s! 7 Figures & ONNX saved. ===")


if __name__ == "__main__":
    main()
