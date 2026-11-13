"""
Constructs notebooks/keyprint_training_pipeline.ipynb with full executable cells
covering EDA, 4-stage Dual-Corpus Preprocessing (CMU + KeyRecs), Direct Training of
Baseline & Proposed Models on GPU, 1:1 Password Intrusion Verification, 1:N Typing-Practice
Multi-User Identification ("Who Is Typing?"), Error Analysis, and ONNX Parity Verification.
"""

from pathlib import Path
import nbformat as nbf


def create_notebook(nb_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# 🔐 KeyPrint Guard & Arena: 타이핑 지문(Keystroke Dynamics) 기반 해커 침입 차단(1:1) 및 타자연습 인물 자동 식별(1:N) 파이프라인\n"
        "- **학번 / 이름**: 201935607 김대현\n"
        "- **데이터셋**:\n"
        "  1. **CMU Keystroke Dynamics Benchmark** (`20,400` 세션 = `51명 × 8세션 × 50회`, 고정 암호 `.tie5Roanl`)\n"
        "  2. **KeyRecs Multi-Session Sentence & Fixed-Text Dataset** (Zenodo `10.5281/zenodo.7886742`, CC-BY 4.0, `96명` 영문 타자연습 문장 `562,583` 원시 키 이벤트 → 무오타 `25자` 문장 시퀀스 `12,848`건 + 고정 문자열)\n"
        "- **핵심 목표**:\n"
        "  1. **[모드 1 · 1:1 해커 차단]** 비밀번호가 유출된 상황에서도 타건 미세 타이밍으로부터 **64차원 생체 지문 임베딩**을 추출하여 본인 여부를 검증하고 해커·매크로 봇 침입을 실시간 차단\n"
        "  2. **[모드 2 · 1:N 인물 분류]** 여러 사용자가 타자연습 문장을 입력해 두면, 이후 **동일한 문장(또는 새로운 연습 문장)**을 타이핑했을 때 **누가 쳤는지 자동으로 분류(Who-Is-Typing)**\n"
        "---"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 1. 실험 환경 설정 및 이중 벤치마크 데이터 로드 (출처·규모·클래스 분포 확인)"))
    cells.append(nbf.v4.new_code_cell(
        "import sys, json, time\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import torch\n"
        "import onnxruntime as ort\n\n"
        "ROOT = Path('..').resolve()\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n\n"
        "from preprocessing import (\n"
        "    SEED, CHANNEL_NAMES, DIGRAPH_LABELS, N_CHANNELS, N_TIMING_CHANNELS, TOUCH_PROB, get_feature_columns,\n"
        "    prepare_full_pipeline\n"
        ")\n"
        "from models import ScaledManhattanDetector, Vanilla1DCNN, KeyPrintNet\n"
        "from train_and_evaluate import (\n"
        "    set_all_seeds, configure_korean_matplotlib, get_device,\n"
        "    build_verification_trials, train_vanilla_baseline, train_keyprint_net,\n"
        "    encode_sequences, compute_cohort_centroids, score_manhattan_trials, score_embedding_trials,\n"
        "    evaluate_1toN_identification_embeddings, evaluate_1toN_identification_manhattan,\n"
        "    find_optimal_val_threshold, summarize_metrics\n"
        ")\n\n"
        "set_all_seeds(SEED)\n"
        "font_used = configure_korean_matplotlib()\n"
        "device = get_device()\n"
        "gpu_desc = torch.cuda.get_device_name(0) if device.type == 'cuda' else str(device)\n"
        "print(f'PyTorch {torch.__version__} | Device: {device} ({gpu_desc}) | Seed: {SEED} | Font: {font_used}')\n\n"
        "raw_df = pd.read_csv(ROOT / 'data/raw/DSL-StrongPasswordData.csv')\n"
        "print(f'[CMU 데이터] 크기: {raw_df.shape[0]:,} 행 × {raw_df.shape[1]} 열 | 결측치: {raw_df.isnull().sum().sum()}건')\n"
        "print(f'[CMU 피험자] 총 {raw_df[\"subject\"].nunique()}명 | 피험자당 샘플 수: {raw_df[\"subject\"].value_counts().unique().tolist()} (완전 균형 분포)')\n"
        "raw_df.head(5)"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 2. 데이터 전처리 (4가지 기법 구현 및 전·후 정량/시각 비교)\n"
        "1. **생리학적 이상치 윈저라이징 & 연속 타자 구간 복원**: 입력 중 딴짓 등으로 발생한 최대 `25.99초` 지연 이상치를 Train 백분위수 기반으로 클리핑하고, 영문 타자연습 원시 이벤트에서 누적 타임스탬프 복원 및 수정키 분절로 100% 무오타 문장 시퀀스 추출\n"
        "2. **17채널 디그래프(Digraph) 시퀀스 피처 엔지니어링**: 원시 타이밍(`H_curr, H_next, DD, UD`)에 **템포 불변 리듬 비율(`Rel_H, Rel_DD`)**, **중첩비(`Duty_Ratio`)**, **연타 롤오버 크기(`Rollover_Mag`)**, **국소 가속도(`Local_Accel`)** 9채널 + **키 위치 7채널**(`KeyboardEvent.code` 기반 행·열·거리·같은 손·같은 손가락) + **누름 시간 측정 여부(`Hold_Valid`)** 1채널\n"
        "3. **피험자 완전 분리 분할(`Subject-Disjoint Split, seed=42`) & 누수 없는 채널별 Z-Score 표준화**: 학습 `103명` (`CMU 35명 + KeyRecs 68명`) / 검증 `16명` / **미학습 테스트 `28명` (`CMU 10명 + KeyRecs 18명`)**으로 인물을 완전히 격리\n"
        "4. **탄성 템포 워핑(Smooth Tempo Warping) 데이터 증강**: 세션 간 타자 속도(WPM) 변화(`0.80x~1.25x`) 및 미세 지터 모사"
    ))
    cells.append(nbf.v4.new_code_cell(
        "splits, meta = prepare_full_pipeline(\n"
        "    ROOT / 'data/raw/DSL-StrongPasswordData.csv',\n"
        "    ROOT / 'data/processed',\n"
        "    seed=SEED\n"
        ")\n"
        "sentence_splits = meta.pop('_sentence_splits', {})\n"
        "kr_fixed_train = meta.pop('_keyrecs_fixed_train', None)\n"
        "meta.pop('_keyrecs_runs', None)\n"
        "meta.pop('_keyrecs_splits_subjs', None)\n\n"
        "ba = meta['preprocessing_before_after']\n"
        "comp_df = pd.DataFrame([\n"
        "    {'지표': '전체 타이밍 컬럼 평균 왜도 (Mean Skewness)', '전처리 전': ba['before_mean_skewness'], '전처리 후': ba['after_mean_skewness']},\n"
        "    {'지표': '최대 컬럼 왜도 (Max Skewness, DD.i.e)', '전처리 전': ba['before_max_skewness'], '전처리 후': ba['after_max_skewness']},\n"
        "    {'지표': '전체 타이밍 컬럼 평균 첨도 (Mean Kurtosis)', '전처리 전': ba['before_mean_kurtosis'], '전처리 후': ba['after_mean_kurtosis']},\n"
        "])\n"
        "print('=== [전처리 전·후 분포 정규성 비교] ===')\n"
        "display(comp_df)\n"
        "print('=== [CMU 암호 코퍼스 분할 현황] ===')\n"
        "print(json.dumps(meta['split_counts'], indent=2, ensure_ascii=False))\n"
        "print('=== [KeyRecs 영문 타자연습 문장 코퍼스 분할 현황] ===')\n"
        "print(json.dumps(meta['keyrecs_sentence_corpus'], indent=2, ensure_ascii=False))\n"
        "print(f'실제 인간 타이핑 내 음수 비행시간(연타 롤오버, UD < 0) 발생 비율: {ba[\"rollover_ratio_pct\"]}%')"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display as ipy_display\n"
        "ipy_display(Image(filename=str(ROOT / 'assets/figures/fig1_preprocessing_comparison.png'), width=960))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 3. 모델 직접 학습 (통계 베이스라인 + 딥러닝 베이스라인 + 제안 모델 KeyPrintNet)\n"
        "- **Model 1 (`ScaledManhattanDetector`)**: CMU 벤치마크 논문(Killourhy & Maxion, 2009)의 표준 통계 이상 탐지 모델\n"
        "- **Model 2 (`Vanilla1DCNN`, 직접 학습)**: 위치 비인지 3층 1D-CNN + Global Average Pooling + Softmax Cross-Entropy (`103명` 다중 길이 학습)\n"
        "- **Model 3 (`KeyPrintNet`, 제안 모델 직접 학습)**: 연속 상대 위치 고조파(`Positional Harmonics`) + 다중 스케일 잔차 SE-Conv1D + **스펙트럼 시간 피라미드(1D-DCT) & Attentive Statistics Pooling** 결합, **Sub-Center CosFace + Supervised Contrastive Loss** 및 **AS-Norm 코호트 교정** 적용"
    ))
    cells.append(nbf.v4.new_code_cell(
        "chan_mean = np.array(meta['channel_mean'], dtype=np.float32)\n"
        "chan_std = np.array(meta['channel_std'], dtype=np.float32)\n"
        "bounds = {k: tuple(v) for k, v in meta['outlier_bounds'].items()}\n"
        "sent_train = sentence_splits.get('train')\n"
        "sent_val = sentence_splits.get('val')\n"
        "sent_test = sentence_splits.get('test')\n\n"
        "val_trials = build_verification_trials(splits['val'], chan_mean, chan_std, k_shot=5, seed=SEED)\n"
        "test_trials = build_verification_trials(splits['test'], chan_mean, chan_std, k_shot=5, seed=SEED)\n\n"
        "# 1) Statistical Baseline\n"
        "manhattan = ScaledManhattanDetector(shrinkage=0.35).fit_prior(splits['train'].raw_flat, splits['train'].subject_ids)\n"
        "yv_m, sv_m = score_manhattan_trials(manhattan, splits['val'], val_trials)\n"
        "thresh_m = find_optimal_val_threshold(yv_m, sv_m)\n"
        "yt_m, st_m = score_manhattan_trials(manhattan, splits['test'], test_trials)\n"
        "m_manhattan = summarize_metrics(yt_m, st_m, thresh_m)\n\n"
        "# 2) Neural Baseline (Scratch Vanilla 1D-CNN)\n"
        "vanilla_net, hist_v = train_vanilla_baseline(\n"
        "    splits['train'], splits['val'], val_trials, device, epochs=35, batch_size=256,\n"
        "    sentence_train=sent_train, kr_fixed_train=kr_fixed_train\n"
        ")\n"
        "test_emb_v = encode_sequences(vanilla_net, splits['test'].seq_scaled, device)\n"
        "yv_v, sv_v, _ = score_embedding_trials(encode_sequences(vanilla_net, splits['val'].seq_scaled, device), encode_sequences(vanilla_net, val_trials['bot_seq'], device), val_trials)\n"
        "thresh_v = find_optimal_val_threshold(yv_v, sv_v)\n"
        "yt_v, st_v, _ = score_embedding_trials(test_emb_v, encode_sequences(vanilla_net, test_trials['bot_seq'], device), test_trials)\n"
        "m_vanilla = summarize_metrics(yt_v, st_v, thresh_v)\n\n"
        "# 3) Proposed Model (KeyPrintNet + Multi-Corpus Metric Learning + AS-Norm)\n"
        "keyprint_net, hist_k = train_keyprint_net(\n"
        "    splits['train'], splits['val'], val_trials, bounds, chan_mean, chan_std,\n"
        "    device, epochs=35, batch_size=256, use_augmentation=True,\n"
        "    sentence_train=sent_train, kr_fixed_train=kr_fixed_train, sentence_val=sent_val,\n"
        "    tag='KeyPrintNet', in_channels=N_CHANNELS, touch_prob=TOUCH_PROB,\n"
        "    touch_hold=float(meta['touch_hold_impute_sec'])\n"
        ")\n"
        "cohort_k = compute_cohort_centroids(encode_sequences(keyprint_net, splits['train'].seq_scaled, device), splits['train'].subject_ids)\n"
        "test_emb_k = encode_sequences(keyprint_net, splits['test'].seq_scaled, device)\n"
        "yv_k, sv_k, _ = score_embedding_trials(encode_sequences(keyprint_net, splits['val'].seq_scaled, device), encode_sequences(keyprint_net, val_trials['bot_seq'], device), val_trials, cohort=cohort_k)\n"
        "thresh_k = find_optimal_val_threshold(yv_k, sv_k)\n"
        "yt_k, st_k, _ = score_embedding_trials(test_emb_k, encode_sequences(keyprint_net, test_trials['bot_seq'], device), test_trials, cohort=cohort_k)\n"
        "m_keyprint = summarize_metrics(yt_k, st_k, thresh_k)"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 4. 미학습 테스트셋 객관적 성능 비교\n"
        "### 4-1. [과업 1] 1:1 유출 암호 해커 침입 차단 평가 (Unseen 10명 · 3,600 검증 쌍)"
    ))
    cells.append(nbf.v4.new_code_cell(
        "results_1to1 = pd.DataFrame([\n"
        "    {\n"
        "        '모델 (Model)': 'ScaledManhattan (통계 베이스라인)',\n"
        "        'Test Macro F1 ↑': m_manhattan['macro_f1'],\n"
        "        'ROC-AUC ↑': m_manhattan['roc_auc'],\n"
        "        'EER (동일오류율) ↓': f\"{m_manhattan['eer']*100:.2f}%\",\n"
        "        'Accuracy ↑': f\"{m_manhattan['accuracy']*100:.2f}%\",\n"
        "        '해커 오허용률 (FAR) ↓': f\"{m_manhattan['far']*100:.2f}%\",\n"
        "        '본인 오차단율 (FRR) ↓': f\"{m_manhattan['frr']*100:.2f}%\",\n"
        "        '가변 문장 길이 지원': '불가 (고정 차원)',\n"
        "    },\n"
        "    {\n"
        "        '모델 (Model)': 'Vanilla1DCNN (딥러닝 베이스라인)',\n"
        "        'Test Macro F1 ↑': m_vanilla['macro_f1'],\n"
        "        'ROC-AUC ↑': m_vanilla['roc_auc'],\n"
        "        'EER (동일오류율) ↓': f\"{m_vanilla['eer']*100:.2f}%\",\n"
        "        'Accuracy ↑': f\"{m_vanilla['accuracy']*100:.2f}%\",\n"
        "        '해커 오허용률 (FAR) ↓': f\"{m_vanilla['far']*100:.2f}%\",\n"
        "        '본인 오차단율 (FRR) ↓': f\"{m_vanilla['frr']*100:.2f}%\",\n"
        "        '가변 문장 길이 지원': '가능',\n"
        "    },\n"
        "    {\n"
        "        '모델 (Model)': 'KeyPrintNet (제안 모델 · 최종 채택)',\n"
        "        'Test Macro F1 ↑': m_keyprint['macro_f1'],\n"
        "        'ROC-AUC ↑': m_keyprint['roc_auc'],\n"
        "        'EER (동일오류율) ↓': f\"{m_keyprint['eer']*100:.2f}%\",\n"
        "        'Accuracy ↑': f\"{m_keyprint['accuracy']*100:.2f}%\",\n"
        "        '해커 오허용률 (FAR) ↓': f\"{m_keyprint['far']*100:.2f}%\",\n"
        "        '본인 오차단율 (FRR) ↓': f\"{m_keyprint['frr']*100:.2f}%\",\n"
        "        '가변 문장 길이 지원': '가능 (ONNX 실시간)',\n"
        "    },\n"
        "])\n"
        "display(results_1to1)"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### 4-2. [과업 2] 1:N 타자연습 인물 자동 분류 평가 (동일 문구 10인 분류 & 영문 문장 18인 분류)"
    ))
    cells.append(nbf.v4.new_code_cell(
        "id_m = evaluate_1toN_identification_manhattan(\n"
        "    manhattan, splits['test'].raw_flat, splits['test'].subject_ids,\n"
        "    splits['test'].sessions, splits['test'].reps, splits['test'].subjects, k_shot=5, cross_session=True\n"
        ")\n"
        "id_v, _ = evaluate_1toN_identification_embeddings(\n"
        "    test_emb_v, splits['test'].subject_ids, splits['test'].sessions,\n"
        "    splits['test'].reps, splits['test'].subjects, k_shot=5, cross_session=True\n"
        ")\n"
        "id_k, _ = evaluate_1toN_identification_embeddings(\n"
        "    test_emb_k, splits['test'].subject_ids, splits['test'].sessions,\n"
        "    splits['test'].reps, splits['test'].subjects, k_shot=5, cross_session=True, cohort=cohort_k\n"
        ")\n\n"
        "sent_coh = compute_cohort_centroids(encode_sequences(keyprint_net, sent_train.seq_scaled, device), sent_train.subject_ids)\n"
        "sent_id_k, _ = evaluate_1toN_identification_embeddings(\n"
        "    encode_sequences(keyprint_net, sent_test.seq_scaled, device),\n"
        "    sent_test.subject_ids, sent_test.sessions, sent_test.chunk_ids, sent_test.subjects,\n"
        "    k_shot=5, cross_session=False, cohort=sent_coh\n"
        ")\n\n"
        "id_df = pd.DataFrame([\n"
        "    {'과업 / 모델': '동일 문구 10인 분류 - ScaledManhattan', '후보 인물 수': '10명 (무작위 10%)', 'Top-1 정확도 ↑': f\"{id_m['top1_accuracy']*100:.2f}%\", 'Top-3 정확도 ↑': f\"{id_m['top3_accuracy']*100:.2f}%\", 'Macro F1 ↑': id_m['macro_f1']},\n"
        "    {'과업 / 모델': '동일 문구 10인 분류 - Vanilla1DCNN', '후보 인물 수': '10명 (무작위 10%)', 'Top-1 정확도 ↑': f\"{id_v['top1_accuracy']*100:.2f}%\", 'Top-3 정확도 ↑': f\"{id_v['top3_accuracy']*100:.2f}%\", 'Macro F1 ↑': id_v['macro_f1']},\n"
        "    {'과업 / 모델': '동일 문구 10인 분류 - KeyPrintNet (제안)', '후보 인물 수': '10명 (무작위 10%)', 'Top-1 정확도 ↑': f\"{id_k['top1_accuracy']*100:.2f}%\", 'Top-3 정확도 ↑': f\"{id_k['top3_accuracy']*100:.2f}%\", 'Macro F1 ↑': id_k['macro_f1']},\n"
        "    {'과업 / 모델': '영문 타자연습 문장(25자) 18인 분류 - KeyPrintNet', '후보 인물 수': '18명 (무작위 5.6%)', 'Top-1 정확도 ↑': f\"{sent_id_k['top1_accuracy']*100:.2f}%\", 'Top-3 정확도 ↑': f\"{sent_id_k['top3_accuracy']*100:.2f}%\", 'Macro F1 ↑': sent_id_k['macro_f1']},\n"
        "])\n"
        "display(id_df)"
    ))

    cells.append(nbf.v4.new_code_cell(
        "for fig_name in [\n"
        "    'fig2_learning_curves.png',\n"
        "    'fig3_roc_det_kshot.png',\n"
        "    'fig4_confusion_matrices.png',\n"
        "    'fig5_embedding_tsne.png',\n"
        "    'fig6_error_analysis_and_ablation.png',\n"
        "    'fig7_typist_identification.png',\n"
        "]:\n"
        "    ipy_display(Image(filename=str(ROOT / 'assets/figures' / fig_name), width=960))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 5. 확장: 키 위치 채널 · 터치 입력 · 행동 프로필 (웹 캡처 그대로 재현한 평가)\n"
        "- **키 위치 7채널**: 같은 타이밍이라도 어떤 키 쌍인지(거리·같은 손·같은 손가락)에 따라 의미가 달라서, 모델이 키 조건부 리듬을 배우도록 입력에 추가\n"
        "- **터치 입력**: 모바일 소프트 키보드는 `keyup` 시점을 주지 않아 누름 시간을 알 수 없음 → Train 누름 시간 중앙값으로 채우고 `Hold_Valid=0`으로 표시, 학습 시 18% 확률로 같은 상황을 모사\n"
        "- **행동 프로필**: 백스페이스 습관·Shift/CapsLock 전략·멈춤 리듬 16개 특성을 two-covariance LLR로 점수화하고, 영역(비밀번호/키보드 문장/터치 문장)별 가중치 β를 검증셋에서 결정\n"
        "- **비밀번호 인코더 선택**: 사용자는 자기 비밀번호(학습에 없던 문자열)를 쓰므로, 학습하지 않은 문자열에서도 점수가 유지되는지 비교한 뒤 배포 인코더를 결정\n"
        "\n아래 표는 `src/visualize_and_run.py`가 저장한 `models/evaluation_report.json`의 `feature_upgrade` 결과를 그대로 읽어 보여줍니다."
    ))
    cells.append(nbf.v4.new_code_cell(
        "report = json.load(open(ROOT / 'models/evaluation_report.json', encoding='utf-8'))\n"
        "fu = report['feature_upgrade']\n"
        "rows = []\n"
        "for dev_name, block in fu['web_faithful_identification'].items():\n"
        "    for model_name, res in block.items():\n"
        "        rows.append({'입력': dev_name, '모델': model_name,\n"
        "                     '같은 날 다른 문장 Top-1': f\"{res['same_session']['top1_accuracy']*100:.1f}%\",\n"
        "                     '다른 날 다른 글 Top-1': f\"{res['cross_session']['top1_accuracy']*100:.1f}%\"})\n"
        "print('=== [1:N] 웹 캡처 재현 18인 식별 (3문장 등록) ===')\n"
        "display(pd.DataFrame(rows))\n"
        "print('행동 프로필 가중치 β (검증셋 결정):', fu['behavior_beta'])\n\n"
        "pw_rows = [\n"
        "    {'조건': 'CMU (학습한 암호)', '9채널 AUC': fu['password_1to1']['legacy_9ch']['roc_auc'], '17채널 AUC': fu['password_1to1']['keyprint_17ch']['roc_auc']},\n"
        "    {'조건': 'CMU 터치 시뮬레이션', '9채널 AUC': fu['password_1to1_touch']['legacy_9ch_imputed_holds']['roc_auc'], '17채널 AUC': fu['password_1to1_touch']['keyprint_17ch']['roc_auc']},\n"
        "]\n"
        "for key, label in [('keyrecs_fixed_cross_session', 'KeyRecs 고정문 (다른 날)'), ('repeated_words_unseen', '반복 단어 (미학습 문자열)')]:\n"
        "    blk = fu['password_transfer_other_content'][key]\n"
        "    pw_rows.append({'조건': label, '9채널 AUC': blk['legacy_9ch']['roc_auc'], '17채널 AUC': blk['keyprint_17ch']['roc_auc'],\n"
        "                    '배포 임계값 FAR (9ch / 17ch)': f\"{blk['legacy_9ch']['far_at_deployed_threshold']*100:.1f}% / {blk['keyprint_17ch']['far_at_deployed_threshold']*100:.1f}%\"})\n"
        "print('=== [1:1] 비밀번호 인코더 비교 → 배포:', fu['password_encoder_deployed'], '===')\n"
        "display(pd.DataFrame(pw_rows))\n"
        "ipy_display(Image(filename=str(ROOT / 'assets/figures/fig8_feature_upgrade.png'), width=960))"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 6. 웹 배포용 ONNX 두 개(문장 17채널 · 비밀번호 9채널) 가변 길이 추론 및 정밀도 일치 검증"))
    cells.append(nbf.v4.new_code_cell(
        "for title, onnx_file, pt_file, n_ch in [\n"
        "    ('문장용 KeyPrintNet 17채널', ROOT / 'models/keyprint_net.onnx', ROOT / 'models/keyprint_net.pt', N_CHANNELS),\n"
        "    ('비밀번호용 KeyPrintNet 9채널', ROOT / 'models/legacy/keyprint_net_9ch.onnx', ROOT / 'models/legacy/keyprint_net_9ch.pt', N_TIMING_CHANNELS),\n"
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
        "        max_err = np.max(np.abs(pt_emb - ort_emb))\n"
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
    out = Path(__file__).resolve().parent.parent / "notebooks" / "keyprint_training_pipeline.ipynb"
    create_notebook(out)
    print(f"Created {out}")
