"""
Constructs notebooks/keyprint_training_pipeline.ipynb as a non-retraining canonical
verification notebook.

Executing the generated notebook NEVER retrains neural models (avoiding divergence from
canonical checkpoints). Instead it:
  1) Loads raw CSVs & runs `prepare_full_pipeline` in an isolated TemporaryDirectory to
     verify lossless `KeyRecs-free-text.csv` parsing (`562,398` transitions + `198`
     session-end strokes + `26` quote rows), subject-disjoint split counts & `%` ratios,
     non-overlapping 25-key sentence windows, and 5-stage preprocessing silhouettes.
  2) Loads canonical checkpoints (`vanilla_1dcnn.pt`, `vanilla_1dcnn_aug.pt`,
     `legacy/keyprint_net_9ch.pt`, `keyprint_net.pt`) and canonical JSON reports
     (`evaluation_report.json`, `password_calibration.json`, `typing_style_analysis.json`).
  3) Runs live forward-pass inference on held-out Test samples and asserts live metrics
     match `evaluation_report.json` within `1e-5`; renders training curves from saved
     histories; displays controlled baselines, CMU human (`1,600`) vs bot (`200`)
     impostor breakdown, 1:N (`3문장 단일질의 기준선` vs `5문장 등록 · 2문장 질의 배포 프로토콜`),
     and `§5.6~5.7` disjoint-`occ_id` synthesis / fixed-text / calibration analyses.
  4) Verifies live PyTorch-vs-ONNX numerical parity across `L in [5, 10, 18, 24, 32]`
     and `model_config.json` schema `v4` SHA-256 provenance.
"""

from __future__ import annotations

import os
from pathlib import Path
import nbformat as nbf


def create_notebook(nb_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# 🔐 KeyPrint Guard & Arena: 타이핑 지문(Keystroke Dynamics) 기반 해커 침입 차단(1:1) 및 타자연습 인물 자동 식별(1:N) 공식 검증 노트북\n"
        "- **학번 / 이름**: 201935607 김대현\n"
        "- **노트북 역할 (Non-Retraining Canonical Verification)**:\n"
        "  - 본 노트북은 재실행 시 가중치가 달라지는 중복 재학습을 수행하지 않고, 공식 파이프라인(`src/visualize_and_run.py`, `src/eval_password_calibration.py`, `src/eval_typing_style.py`)이 생성한 **정규 가중치(`.pt`, `.onnx`)·학습 에포크 로그(`histories`)·평가 리포트 JSON 3종**을 로드하여 **실시간 순방향 추론 재현성(`±1e-5`)과 전 구간 지표를 검증**합니다.\n"
        "- **데이터셋**:\n"
        "  1. **CMU Keystroke Dynamics Benchmark** (`20,400` 세션 = `51명 × 8세션 × 50회`, 고정 암호 `.tie5Roanl`, 학술 공개)\n"
        "  2. **KeyRecs Multi-Session Sentence & Fixed-Text Dataset** (Zenodo `10.5281/zenodo.7886742`, CC-BY 4.0, 원시 `99명 × 2세션` 자유 필사 `562,596` 데이터 라인 = 유효 전이 `562,398`행 + 세션 종단 타건 `198`행 → 무오타 비중첩 `25자` 문장 시퀀스 통과 `96명` + 고정 문자열 `vpwjkeurkb`)\n"
        "---"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 1. 실험 환경 설정 및 원시 이중 코퍼스 로드"))
    cells.append(nbf.v4.new_code_cell(
        "import hashlib, json, os, sys, tempfile, time\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import torch\n"
        "import onnxruntime as ort\n"
        "from sklearn.metrics import silhouette_score\n"
        "from IPython.display import Image, display as ipy_display\n\n"
        "DEFAULT_ROOT = Path('..').resolve()\n"
        "REPO_ROOT = Path(os.environ.get('KEYPRINT_REPO_ROOT', DEFAULT_ROOT)).resolve()\n"
        "if not (REPO_ROOT / 'src').exists():\n"
        "    REPO_ROOT = Path.cwd().resolve()\n"
        "OUT_ROOT = Path(os.environ.get('KEYPRINT_OUTPUT_ROOT', REPO_ROOT)).resolve()\n"
        "sys.path.insert(0, str(REPO_ROOT / 'src'))\n\n"
        "from preprocessing import (\n"
        "    SEED, CHANNEL_NAMES, N_CHANNELS, N_TIMING_CHANNELS, prepare_full_pipeline\n"
        ")\n"
        "from models import ScaledManhattanDetector, Vanilla1DCNN, KeyPrintNet\n"
        "from train_and_evaluate import (\n"
        "    set_all_seeds, configure_korean_matplotlib, get_device,\n"
        "    build_verification_trials, encode_sequences, compute_cohort_centroids,\n"
        "    score_manhattan_trials, score_embedding_trials,\n"
        "    evaluate_1toN_identification_embeddings, evaluate_1toN_identification_manhattan,\n"
        "    find_optimal_val_threshold, summarize_metrics, impostor_breakdown_embedding, impostor_breakdown_manhattan\n"
        ")\n\n"
        "set_all_seeds(SEED)\n"
        "font_used = configure_korean_matplotlib()\n"
        "device = get_device()\n"
        "print(f'PyTorch {torch.__version__} | Device: {device} | Seed: {SEED} | Font: {font_used}')\n"
        "print(f'REPO_ROOT: {REPO_ROOT}\\nOUT_ROOT : {OUT_ROOT}')\n\n"
        "raw_df = pd.read_csv(REPO_ROOT / 'data/raw/DSL-StrongPasswordData.csv')\n"
        "print(f'[CMU 데이터] 크기: {raw_df.shape[0]:,} 행 × {raw_df.shape[1]} 열 | 결측치: {raw_df.isnull().sum().sum()}건 | 피험자: {raw_df[\"subject\"].nunique()}명')\n"
        "raw_df.head(3)"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 2. 무손실 파싱 · 피험자 완전 분리 분할(`%` 비율) · 5단계 전처리 실루엣 검증\n"
        "- `KeyRecs-free-text.csv` 줄 단위 무손실 파서로 비이스케이프 큰따옴표(`\"`) `26`행과 세션 종단 타건 `198`행을 포함한 `562,596`개 원시 데이터 라인(`562,398` 전이 + `198` 종단 홀드, 손상 `0`행)을 전량 복구합니다.\n"
        "- 원시 `UD = raw_DD - raw_H_curr`를 먼저 계산한 뒤 Train 백분위수 경계로 `H, DD, UD`를 독립 클리핑하며, 문장 청크는 **25키 비중첩 창(`stride=25`)**으로 추출합니다."
    ))
    cells.append(nbf.v4.new_code_cell(
        "with tempfile.TemporaryDirectory() as _tmp_proc:\n"
        "    splits, meta = prepare_full_pipeline(\n"
        "        REPO_ROOT / 'data/raw/DSL-StrongPasswordData.csv',\n"
        "        Path(_tmp_proc),\n"
        "        seed=SEED,\n"
        "    )\n"
        "sentence_splits = meta.pop('_sentence_splits', {})\n"
        "meta.pop('_keyrecs_fixed_train', None)\n"
        "meta.pop('_keyrecs_runs', None)\n"
        "meta.pop('_keyrecs_splits_subjs', None)\n\n"
        "report = json.loads((OUT_ROOT / 'models/evaluation_report.json').read_text(encoding='utf-8'))\n"
        "pw_cal = json.loads((OUT_ROOT / 'models/password_calibration.json').read_text(encoding='utf-8'))\n"
        "style_rep = json.loads((OUT_ROOT / 'models/typing_style_analysis.json').read_text(encoding='utf-8'))\n"
        "web_cfg = json.loads((OUT_ROOT / 'web/public/models/model_config.json').read_text(encoding='utf-8'))\n\n"
        "kr_meta = meta['keyrecs_sentence_corpus']\n"
        "assert kr_meta['total_lines'] == 562596 and kr_meta['valid_transitions'] == 562398\n"
        "assert kr_meta['session_end_strokes'] == 198 and kr_meta['recovered_quote_rows'] == 26 and kr_meta['corrupt_rows'] == 0\n"
        "assert kr_meta['sentence_stride_keys'] == 25 and kr_meta['sentence_window_keys'] == 25\n"
        "print('=== [무손실 KeyRecs 파서 & 비중첩 창 검증 통과] ===')\n"
        "print(json.dumps(kr_meta, indent=2, ensure_ascii=False))\n\n"
        "print('\\n=== [CMU 피험자 완전 분리 분할 (비율 % 포함)] ===')\n"
        "print(json.dumps(meta['split_counts'], indent=2, ensure_ascii=False))\n\n"
        "# 5단계 Test 코사인 실루엣 직접 재산출 및 리포트 일치 검증\n"
        "n_test = len(splits['test'].df)\n"
        "sils = {\n"
        "    'silhouette_raw4_cosine': round(float(silhouette_score(np.stack([splits['test'].raw_holds[:, :-1], splits['test'].raw_holds[:, 1:], splits['test'].raw_dds, splits['test'].raw_uds], axis=1).reshape(n_test, -1), splits['test'].subject_ids, metric='cosine')), 4),\n"
        "    'silhouette_clipped4_unscaled_cosine': round(float(silhouette_score(splits['test'].seq_unscaled[:, :4, :].reshape(n_test, -1), splits['test'].subject_ids, metric='cosine')), 4),\n"
        "    'silhouette_clipped4_scaled_cosine': round(float(silhouette_score(splits['test'].seq_scaled[:, :4, :].reshape(n_test, -1), splits['test'].subject_ids, metric='cosine')), 4),\n"
        "    'silhouette_bio9_cosine': round(float(silhouette_score(splits['test'].seq_scaled[:, :N_TIMING_CHANNELS, :].reshape(n_test, -1), splits['test'].subject_ids, metric='cosine')), 4),\n"
        "    'silhouette_bio17_cosine': round(float(silhouette_score(splits['test'].seq_scaled[:, :N_CHANNELS, :].reshape(n_test, -1), splits['test'].subject_ids, metric='cosine')), 4),\n"
        "}\n"
        "for k, v in sils.items():\n"
        "    assert abs(v - report['preprocessing'][k]) <= 1e-4, f'{k}: {v} != {report[\"preprocessing\"][k]}'\n"
        "ba = meta['preprocessing_before_after']\n"
        "display(pd.DataFrame([\n"
        "    {'항목': 'Train 모집단 크기 (동일 모집단 전·후)', '수치': f\"{ba['population']} (n={ba['population_n']:,})\"},\n"
        "    {'항목': '평균 왜도 (전 -> 후)', '수치': f\"{ba['before_mean_skewness']} -> {ba['after_mean_skewness']}\"},\n"
        "    {'항목': '평균 첨도 (전 -> 후)', '수치': f\"{ba['before_mean_kurtosis']} -> {ba['after_mean_kurtosis']}\"},\n"
        "    *[{'항목': k, '수치': v} for k, v in sils.items()],\n"
        "]))\n"
        "ipy_display(Image(filename=str(OUT_ROOT / 'assets/figures/fig1_preprocessing_comparison.png'), width=960))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 3. 공식 가중치 로드 · 실시간 순방향 추론 재현성(`±1e-5`) 검증 · 저장된 학습 곡선 확인\n"
        "- 모델을 재학습하지 않고 `models/*.pt`를 로드하여 미학습 테스트셋(`splits['test']`)에 대해 순방향 추론을 수행한 뒤, `models/evaluation_report.json`에 기록된 공식 지표와 `1e-5` 이내로 완전 일치하는지 검증합니다."
    ))
    cells.append(nbf.v4.new_code_cell(
        "chan_mean = np.array(meta['channel_mean'], dtype=np.float32)\n"
        "chan_std = np.array(meta['channel_std'], dtype=np.float32)\n"
        "val_trials = build_verification_trials(splits['val'], chan_mean, chan_std, k_shot=5, seed=SEED)\n"
        "test_trials = build_verification_trials(splits['test'], chan_mean, chan_std, k_shot=5, seed=SEED)\n\n"
        "# 1) ScaledManhattan\n"
        "manhattan = ScaledManhattanDetector(shrinkage=0.35).fit_prior(splits['train'].raw_flat, splits['train'].subject_ids)\n"
        "yv_m, sv_m = score_manhattan_trials(manhattan, splits['val'], val_trials)\n"
        "thresh_m = find_optimal_val_threshold(yv_m, sv_m)\n"
        "yt_m, st_m = score_manhattan_trials(manhattan, splits['test'], test_trials)\n"
        "live_models = {'ScaledManhattan': summarize_metrics(yt_m, st_m, thresh_m)}\n\n"
        "# 2) Vanilla1DCNN & Vanilla1DCNN_Aug (+ASNorm)\n"
        "vanilla_net = Vanilla1DCNN(in_channels=N_CHANNELS, embed_dim=64)\n"
        "vanilla_net.load_state_dict(torch.load(OUT_ROOT / 'models/vanilla_1dcnn.pt', map_location='cpu', weights_only=True))\n"
        "vanilla_net.to(device).eval()\n"
        "yv_v, sv_v, _ = score_embedding_trials(encode_sequences(vanilla_net, splits['val'].seq_scaled, device), encode_sequences(vanilla_net, val_trials['bot_seq'], device), val_trials, cohort=None, single_centroid=False)\n"
        "yt_v, st_v, _ = score_embedding_trials(encode_sequences(vanilla_net, splits['test'].seq_scaled, device), encode_sequences(vanilla_net, test_trials['bot_seq'], device), test_trials, cohort=None, single_centroid=False)\n"
        "live_models['Vanilla1DCNN'] = summarize_metrics(yt_v, st_v, find_optimal_val_threshold(yv_v, sv_v))\n\n"
        "vanilla_aug = Vanilla1DCNN(in_channels=N_CHANNELS, embed_dim=64)\n"
        "vanilla_aug.load_state_dict(torch.load(OUT_ROOT / 'models/vanilla_1dcnn_aug.pt', map_location='cpu', weights_only=True))\n"
        "vanilla_aug.to(device).eval()\n"
        "coh_va = compute_cohort_centroids(encode_sequences(vanilla_aug, splits['train'].seq_scaled, device), splits['train'].subject_ids)\n"
        "ve_va = encode_sequences(vanilla_aug, splits['val'].seq_scaled, device)\n"
        "vb_va = encode_sequences(vanilla_aug, val_trials['bot_seq'], device)\n"
        "te_va = encode_sequences(vanilla_aug, splits['test'].seq_scaled, device)\n"
        "tb_va = encode_sequences(vanilla_aug, test_trials['bot_seq'], device)\n"
        "yv_va, sv_va, _ = score_embedding_trials(ve_va, vb_va, val_trials, cohort=None, single_centroid=False)\n"
        "yt_va, st_va, _ = score_embedding_trials(te_va, tb_va, test_trials, cohort=None, single_centroid=False)\n"
        "live_models['Vanilla1DCNN_Aug'] = summarize_metrics(yt_va, st_va, find_optimal_val_threshold(yv_va, sv_va))\n"
        "yv_vas, sv_vas, _ = score_embedding_trials(ve_va, vb_va, val_trials, cohort=coh_va, single_centroid=False)\n"
        "yt_vas, st_vas, _ = score_embedding_trials(te_va, tb_va, test_trials, cohort=coh_va, single_centroid=False)\n"
        "live_models['Vanilla1DCNN_Aug_ASNorm'] = summarize_metrics(yt_vas, st_vas, find_optimal_val_threshold(yv_vas, sv_vas))\n\n"
        "# 3) KeyPrintNet 9ch & KeyPrintNet 17ch\n"
        "kp9 = KeyPrintNet(in_channels=N_TIMING_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18)\n"
        "kp9.load_state_dict(torch.load(OUT_ROOT / 'models/legacy/keyprint_net_9ch.pt', map_location='cpu', weights_only=True))\n"
        "kp9.to(device).eval()\n"
        "coh_9 = compute_cohort_centroids(encode_sequences(kp9, splits['train'].seq_scaled[:, :N_TIMING_CHANNELS], device), splits['train'].subject_ids)\n"
        "yv_9, sv_9, _ = score_embedding_trials(encode_sequences(kp9, splits['val'].seq_scaled[:, :N_TIMING_CHANNELS], device), encode_sequences(kp9, val_trials['bot_seq'][:, :N_TIMING_CHANNELS], device), val_trials, cohort=coh_9)\n"
        "yt_9, st_9, _ = score_embedding_trials(encode_sequences(kp9, splits['test'].seq_scaled[:, :N_TIMING_CHANNELS], device), encode_sequences(kp9, test_trials['bot_seq'][:, :N_TIMING_CHANNELS], device), test_trials, cohort=coh_9)\n"
        "live_models['KeyPrintNet_9ch'] = summarize_metrics(yt_9, st_9, find_optimal_val_threshold(yv_9, sv_9))\n\n"
        "kp17 = KeyPrintNet(in_channels=N_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18)\n"
        "kp17.load_state_dict(torch.load(OUT_ROOT / 'models/keyprint_net.pt', map_location='cpu', weights_only=True))\n"
        "kp17.to(device).eval()\n"
        "coh_17 = compute_cohort_centroids(encode_sequences(kp17, splits['train'].seq_scaled, device), splits['train'].subject_ids)\n"
        "yv_17, sv_17, _ = score_embedding_trials(encode_sequences(kp17, splits['val'].seq_scaled, device), encode_sequences(kp17, val_trials['bot_seq'], device), val_trials, cohort=coh_17)\n"
        "yt_17, st_17, _ = score_embedding_trials(encode_sequences(kp17, splits['test'].seq_scaled, device), encode_sequences(kp17, test_trials['bot_seq'], device), test_trials, cohort=coh_17)\n"
        "live_models['KeyPrintNet'] = summarize_metrics(yt_17, st_17, find_optimal_val_threshold(yv_17, sv_17))\n\n"
        "for m_name, live_m in live_models.items():\n"
        "    rep_m = report['models'][m_name]\n"
        "    for metric_key in ('eer', 'roc_auc', 'macro_f1'):\n"
        "        diff = abs(live_m[metric_key] - rep_m[metric_key])\n"
        "        assert diff <= 1e-5, f'{m_name}.{metric_key} mismatch: live={live_m[metric_key]} vs report={rep_m[metric_key]}'\n"
        "print('모든 6개 통제 비교 모델 실시간 순방향 추론 지표가 evaluation_report.json과 1e-5 이내 완전 일치합니다!')\n\n"
        "# 저장된 에포크 학습 로그 요약 및 학습 곡선 시각화\n"
        "hist = report['histories']\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))\n"
        "for label, key in [('Vanilla1DCNN', 'Vanilla1DCNN'), ('Vanilla1DCNN_Aug', 'Vanilla1DCNN_Aug'), ('KeyPrintNet 9ch', 'KeyPrintNet_9ch'), ('KeyPrintNet 17ch', 'KeyPrintNet')]:\n"
        "    rows_h = hist.get(key, [])\n"
        "    if rows_h:\n"
        "        axes[0].plot([r['epoch'] for r in rows_h], [r['train_loss'] for r in rows_h], label=f'{label} (n={len(rows_h)} ep)')\n"
        "        axes[1].plot([r['epoch'] for r in rows_h], [r['val_eer'] * 100 for r in rows_h], label=label)\n"
        "axes[0].set_title('Saved Train Loss Curve'); axes[0].set_xlabel('Epoch'); axes[0].legend()\n"
        "axes[1].set_title('Saved Validation EER (%) ↓'); axes[1].set_xlabel('Epoch'); axes[1].legend()\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 4. 통제된 베이스라인 비교 · 실측 타인(`1,600`) vs 합성 봇(`200`) 분리 · 1:N 인물 식별 비교\n"
        "- **과업 1 (1:1 검증)**: 동일 증강·동일 AS-Norm 채점을 부여한 `Vanilla1DCNN_Aug` / `Vanilla1DCNN_Aug_ASNorm` 통제 비교와, Validation 임계값에서의 실측 인간 침입자(`1,600`건) 대 매크로 봇(`200`건) 분리 성적을 함께 제시합니다.\n"
        "- **과업 2 (1:N 식별)**: 초기 **3문장 단일질의 기준선(`k=3, q=1`)**과 실제 웹 서비스의 **5문장 등록 · 2문장 질의 배포 프로토콜(`k=5, q=2, open-set margin`)**을 명확히 구분해 비교합니다."
    ))
    cells.append(nbf.v4.new_code_cell(
        "model_order = [\n"
        "    ('ScaledManhattan', 'ScaledManhattan (통계 베이스라인)'),\n"
        "    ('Vanilla1DCNN', 'Vanilla1DCNN (기본 무증강 · 무보정 코사인)'),\n"
        "    ('Vanilla1DCNN_Aug', 'Vanilla1DCNN (+동일 템포·터치 증강)'),\n"
        "    ('Vanilla1DCNN_Aug_ASNorm', 'Vanilla1DCNN (+동일 증강 + AS-Norm 코호트 보정)'),\n"
        "    ('KeyPrintNet_9ch', 'KeyPrintNet 9ch (순수 타이밍 비밀번호 인코더)'),\n"
        "    ('KeyPrintNet', 'KeyPrintNet 17ch (제안 문장·키위치 인코더)'),\n"
        "]\n"
        "tbl_rows, brk_rows = [], []\n"
        "for key, label in model_order:\n"
        "    m = report['models'][key]\n"
        "    ib = m['impostor_breakdown']\n"
        "    tbl_rows.append({\n"
        "        '모델': label,\n"
        "        'Val 임계값': m['threshold'],\n"
        "        'Test Macro F1 ↑': m['macro_f1'],\n"
        "        'ROC-AUC ↑': m['roc_auc'],\n"
        "        'EER ↓': f\"{m['eer']*100:.2f}%\",\n"
        "        'FAR / FRR': f\"{m['far']*100:.2f}% / {m['frr']*100:.2f}%\",\n"
        "    })\n"
        "    brk_rows.append({\n"
        "        '모델': label,\n"
        "        '실측 타인(1,600) AUC / EER / FAR': f\"{ib['human_only']['roc_auc']:.4f} / {ib['human_only']['eer']*100:.2f}% / {ib['human_only']['far']*100:.2f}%\",\n"
        "        '합성 봇(200) AUC / EER / FAR': f\"{ib['bot_only']['roc_auc']:.4f} / {ib['bot_only']['eer']*100:.2f}% / {ib['bot_only']['far']*100:.2f}%\",\n"
        "        '통합(1,800) AUC / EER / FAR': f\"{ib['combined']['roc_auc']:.4f} / {ib['combined']['eer']*100:.2f}% / {ib['combined']['far']*100:.2f}%\",\n"
        "    })\n"
        "print('=== [§5.1] 1:1 통제 베이스라인 및 제안 모델 비교 (Unseen CMU 10명) ===')\n"
        "display(pd.DataFrame(tbl_rows))\n"
        "print('=== [§5.1] 동일 Validation 임계값 기준 실측 인간 침입자(1,600) vs 합성 매크로 봇(200) 분리 ===')\n"
        "display(pd.DataFrame(brk_rows))\n\n"
        "fu = report['feature_upgrade']\n"
        "id_rows = []\n"
        "for m_key in ('ScaledManhattan', 'Vanilla1DCNN', 'Vanilla1DCNN_Aug', 'Vanilla1DCNN_Aug_ASNorm', 'KeyPrintNet_9ch', 'KeyPrintNet'):\n"
        "    r = report['typist_identification_1toN']['cmu_10way'][m_key]\n"
        "    id_rows.append({'프로토콜': 'CMU 동일 암호 10인 분류 (k=5, 교차 세션)', '모델': m_key, '성능 요약 1': f\"Top-1 {r['top1_accuracy']*100:.2f}%\", '성능 요약 2': f\"Top-3 {r['top3_accuracy']*100:.2f}%\"})\n"
        "for dev_name, blk in fu['web_faithful_identification'].items():\n"
        "    for m_key, res in blk.items():\n"
        "        id_rows.append({\n"
        "            '프로토콜': f'KeyRecs 18인 폐쇄형 [3문장 단일질의 기준선 k=3, q=1] ({dev_name})',\n"
        "            '모델': m_key,\n"
        "            '성능 요약 1': f\"Top-1 당일 {res['same_session']['top1_accuracy']*100:.1f}% / 익일 {res['cross_session']['top1_accuracy']*100:.1f}%\",\n"
        "            '성능 요약 2': f\"Top-3 당일 {res['same_session']['top3_accuracy']*100:.1f}% / 익일 {res['cross_session']['top3_accuracy']*100:.1f}%\",\n"
        "        })\n"
        "ap = fu['arena_protocol']\n"
        "for dev_name in ('keyboard', 'touch'):\n"
        "    g10_base = ap[dev_name]['gallery_10']['single_sentence_enroll3']\n"
        "    g10_dep = ap[dev_name]['gallery_10']['two_sentences_enroll5_margin']\n"
        "    id_rows.append({\n"
        "        '프로토콜': f'Arena 10인 개방형 [3문장 단일질의 기준선 k=3, q=1] ({dev_name})',\n"
        "        '모델': 'KeyPrintNet 17ch + Behavior',\n"
        "        '성능 요약 1': f\"정식별 {g10_base['correct_accept']*100:.1f}% / 오식별 {g10_base['misidentified']*100:.1f}%\",\n"
        "        '성능 요약 2': f\"외부인 오수락 {g10_base['outsider_accepted']*100:.1f}%\",\n"
        "    })\n"
        "    id_rows.append({\n"
        "        '프로토콜': f'Arena 10인 개방형 [5문장 등록 · 2문장 질의 배포 프로토콜 k=5, q=2] ({dev_name})',\n"
        "        '모델': 'KeyPrintNet 17ch + Behavior + Margin',\n"
        "        '성능 요약 1': f\"정식별 {g10_dep['correct_accept']*100:.1f}% / 오식별 {g10_dep['misidentified']*100:.1f}%\",\n"
        "        '성능 요약 2': f\"외부인 오수락 {g10_dep['outsider_accepted']*100:.1f}% (기준선 대비 개선)\",\n"
        "    })\n"
        "print('=== [§5.2] 1:N 인물 식별: 3문장 단일질의 기준선 vs 5문장·2문장 배포 프로토콜 ===')\n"
        "display(pd.DataFrame(id_rows))\n"
        "for fig_name in ['fig2_learning_curves.png', 'fig3_roc_det_kshot.png', 'fig4_confusion_matrices.png', 'fig5_embedding_tsne.png', 'fig6_error_analysis_and_ablation.png', 'fig7_typist_identification.png']:\n"
        "    ipy_display(Image(filename=str(OUT_ROOT / 'assets/figures' / fig_name), width=960))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 5. `§5.6~5.7` 비밀번호 전이 · Occurrence 분리 합성 · 배포 임계값 일치 · 문제 해결 절제 로그\n"
        "- **인코더 선택 격리**: `password_transfer_val`(`Validation` 피험자 전용) 결과만으로 비밀번호 인코더를 선택하고, `password_transfer_other_content`(`Test` 피험자 전용)로 최종 검증합니다.\n"
        "- **디그래프 합성 표본 격리**: 같은 날(`Session 1`) 등록과 본인 질의가 동일한 원본 타건(`occ_id`)을 절대 재사용하지 않도록 분리(`disjoint_occ_enforced=True`)했습니다.\n"
        "- **대문자 습관 배포 임계값 일치**: 브라우저(`biometrics.js`)와 동일하게 `base_tau + tau_offset`에서 평가합니다."
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('=== [Validation 전용 인코더 선택 기록] ===')\n"
        "print(json.dumps(fu['password_encoder_selection'], indent=2, ensure_ascii=False))\n\n"
        "us = fu['password_user_strings']\n"
        "print('\\n=== [§5.6] 디그래프 합성 메타데이터 (동일 세션 occ_id 완전 분리) ===')\n"
        "print(json.dumps(us['synthesis_metadata'], indent=2, ensure_ascii=False))\n\n"
        "cm = fu['password_capital_method']\n"
        "print('\\n=== [§5.7] 대문자 입력 습관 결합 (배포 임계값 base_tau + tau_offset 일치 평가) ===')\n"
        "print(json.dumps({\n"
        "    'weight': cm['weight'],\n"
        "    'tau_offset': cm['tau_offset'],\n"
        "    'deployed_threshold_no_capitals': cm['deployed_threshold_no_capitals'],\n"
        "    'deployed_threshold_with_capitals': cm['deployed_threshold_with_capitals'],\n"
        "    'test_without': cm['test_without'],\n"
        "    'test_with': cm['test_with'],\n"
        "}, indent=2, ensure_ascii=False))\n\n"
        "print('\\n=== [password_calibration.json 요약 (Val/Test 분리 검증)] ===')\n"
        "assert pw_cal['split_isolation']['val_test_pooled'] is False\n"
        "cal_rows = [\n"
        "    {'시나리오': k, '본인/침입자': f\"{v['num_genuine']}/{v['num_impostor']}\",\n"
        "     'AS-Norm AUC (FAR/FRR)': f\"{v['as_norm_auc']:.3f} ({v['as_norm']['far']*100:.1f}%/{v['as_norm']['frr']*100:.1f}%)\",\n"
        "     'Dual AUC (FAR/FRR)': f\"{v['dual_auc']:.3f} ({v['dual']['far']*100:.1f}%/{v['dual']['frr']*100:.1f}%)\"}\n"
        "    for k, v in pw_cal['scenarios'].items()\n"
        "]\n"
        "display(pd.DataFrame(cal_rows))\n\n"
        "print('=== [typing_style_analysis.json 피험자 집계 & 분산 분해] ===')\n"
        "print(json.dumps({'participant_counts': style_rep['participant_counts'], 'variance_decomposition': style_rep['variance_decomposition'], 'same_speed_ideal_eer': style_rep['same_speed_ideal_eer']}, indent=2, ensure_ascii=False))\n\n"
        "print('\\n=== [§9.2 문제 해결 사례 재현 절제 실험 (Tempo Normalization & Backspace Stitch)] ===')\n"
        "print(json.dumps(report['troubleshooting_ablations'], indent=2, ensure_ascii=False))\n"
        "ipy_display(Image(filename=str(OUT_ROOT / 'assets/figures/fig8_feature_upgrade.png'), width=960))"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 6. 웹 배포용 ONNX 가변 길이 정밀도(`max_err < 1e-5`) 및 스키마 `v4` 해시 무결성 검증"))
    cells.append(nbf.v4.new_code_cell(
        "assert web_cfg['schema_version'] == 4 and web_cfg['preprocess_version'] == 'kp-preproc-v4'\n"
        "sha17 = hashlib.sha256((OUT_ROOT / 'web/public/models/keyprint_net.onnx').read_bytes()).hexdigest()\n"
        "sha9 = hashlib.sha256((OUT_ROOT / 'web/public/models/keyprint_net_password.onnx').read_bytes()).hexdigest()\n"
        "expected_hash = hashlib.sha256(f'{sha17}:{sha9}'.encode('utf-8')).hexdigest()[:16]\n"
        "assert web_cfg['onnx_sha256'] == {'sentence_17ch': sha17, 'password_9ch': sha9}\n"
        "assert web_cfg['model_hash'] == expected_hash\n"
        "for sp in web_cfg['mvp_sample_profiles']:\n"
        "    assert sp['schemaVersion'] == 4 and sp['preprocessVersion'] == 'kp-preproc-v4' and sp['modelHash'] == expected_hash\n"
        "print(f'Schema v4 & ONNX SHA-256 검증 통과 | model_hash={expected_hash}')\n\n"
        "for title, onnx_file, pt_file, n_ch in [\n"
        "    ('문장용 KeyPrintNet 17채널', OUT_ROOT / 'web/public/models/keyprint_net.onnx', OUT_ROOT / 'models/keyprint_net.pt', N_CHANNELS),\n"
        "    ('비밀번호용 KeyPrintNet 9채널', OUT_ROOT / 'web/public/models/keyprint_net_password.onnx', OUT_ROOT / 'models/legacy/keyprint_net_9ch.pt', N_TIMING_CHANNELS),\n"
        "]:\n"
        "    sess = ort.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])\n"
        "    cpu_net = KeyPrintNet(in_channels=n_ch, hidden_channels=80, embed_dim=64, dropout=0.0)\n"
        "    cpu_net.load_state_dict(torch.load(pt_file, map_location='cpu', weights_only=True))\n"
        "    cpu_net.eval()\n"
        "    print(f'[{title}] ONNX 파일 크기: {onnx_file.stat().st_size / 1024:.2f} KB')\n"
        "    for seq_len in [5, 10, 18, 24, 32]:\n"
        "        sample_in = np.random.RandomState(seq_len).randn(1, n_ch, seq_len).astype(np.float32)\n"
        "        with torch.no_grad():\n"
        "            pt_emb = cpu_net(torch.from_numpy(sample_in)).numpy()\n"
        "        t0 = time.perf_counter()\n"
        "        ort_emb = sess.run(['biometric_embedding'], {'keystroke_seq': sample_in})[0]\n"
        "        dt_ms = (time.perf_counter() - t0) * 1000\n"
        "        max_err = float(np.max(np.abs(pt_emb - ort_emb)))\n"
        "        assert max_err < 1e-5, f'{title} L={seq_len} max_err={max_err}'\n"
        "        print(f'  L={seq_len:2d} ({seq_len+1:2d}자) -> 출력 {ort_emb.shape} | PyTorch-ONNX 최대 오차: {max_err:.2e} | 추론 {dt_ms:.3f} ms')"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(os.environ["KEYPRINT_OUTPUT_ROOT"]).resolve() if os.environ.get("KEYPRINT_OUTPUT_ROOT") else repo_root
    out = out_root / "notebooks" / "keyprint_training_pipeline.ipynb"
    create_notebook(out)
    print(f"Created {out}")
