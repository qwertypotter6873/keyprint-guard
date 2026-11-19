"""Update the portfolio, Korean study page and submission metric from the evaluated results."""
import argparse
import json
from pathlib import Path


def table(columns, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in
                     [columns, ["---"] * len(columns), *rows])


def pct(value):
    return f"{value:.2%}"


def result_block(report, trials, calibration, timestamps):
    models = report["models"]
    features = report["feature_upgrade"]
    identification = report["typist_identification_1toN"]
    cmu = identification["cmu_10way"]["KeyPrintNet"]
    sentence = identification["keyrecs_18way_sentence"]["KeyPrintNet_same_session"]
    audit = features["sentence_model_selection"]
    deployed = audit["chosen_encoder"]
    labels = {"keyprint_17ch": "KeyPrintNet 17채널", "vanilla_17ch_aug": "Vanilla + 증강 + AS-Norm"}
    proposed = models["KeyPrintNet"]
    summary = trials["sampled_20_draws"]["spread"]["proposed"]["macro_f1"]
    draws = len(trials["sampled_20_draws"]["draws"])
    goal_count = len(summary["seeds_meeting_goal"])

    goals = [
        "프로젝트 초기에 과업별 정량 목표를 설정하고, 학습·임계값 보정에 사용되지 않은 **독립 Test 분할**(`seed=42`)에서 달성 여부를 검증했습니다.",
        table(["과업·조건", "평가지표", "목표치", "Test 결과", "달성 여부"], [
            (name, metric, sign + fmt(limit), fmt(measured), "달성" if passed else "미달")
            for name, metric, sign, limit, measured, fmt, passed in (
                ("CMU 고정 암호 (1:1)", "Macro F1", "≥ ", .85, proposed["macro_f1"], lambda x: f"{x:.4f}", proposed["macro_f1"] >= .85),
                ("CMU 고정 암호 (1:1)", "ROC-AUC", "≥ ", .92, proposed["roc_auc"], lambda x: f"{x:.4f}", proposed["roc_auc"] >= .92),
                ("CMU 고정 암호 (1:1)", "EER", "≤ ", .15, proposed["eer"], pct, proposed["eer"] <= .15),
                ("고정 문구 교차 세션 10인 (1:N)", "Top-1", "≥ ", .40, cmu["top1_accuracy"], pct, cmu["top1_accuracy"] >= .40),
                ("고정 문구 교차 세션 10인 (1:N)", "Top-3", "≥ ", .88, cmu["top3_accuracy"], pct, cmu["top3_accuracy"] >= .88),
                ("영문 25자 동일 세션 18인 (1:N)", "Top-1", "≥ ", .55, sentence["top1_accuracy"], pct, sentence["top1_accuracy"] >= .55),
                ("영문 25자 동일 세션 18인 (1:N)", "Top-3", "≥ ", .88, sentence["top3_accuracy"], pct, sentence["top3_accuracy"] >= .88),
            )]),
        f"- **표본 시드 민감도 확인**: 동일한 CMU Test 10명에서 검증 질의 쌍만 {draws}개의 서로 다른 난수 시드로 다시 추출했을 때, `Macro F1 ≥ 0.85`를 넘긴 시드는 **{goal_count}/{draws}개**였습니다(중앙값 `{summary['median']:.5f}`, 범위 `{summary['min']:.4f}`\\~`{summary['max']:.4f}`, 이전 공개 버전은 `2/20`개). 즉 공식 시드(`0.8427`)는 특정 표본에만 불리하거나 유리하게 치우친 값이 아닌 중앙값 부근의 대표 성적입니다. [시드별 측정값](models/trial_sensitivity.json)\n"
        "- **웹 경량화·속도 목표 달성**: 클라이언트 배포 목표는 **인코더당 `1 MiB` 이하, 1회 추론 지연 `5 ms` 이하**이며, 최종 ONNX 모델 크기는 `592`\\~`599 KiB`(§8 참조), 브라우저 실시간 추론 지연은 데모 판정 화면에 함께 표시됩니다.",
    ]

    rows = []
    for name, label in [("ScaledManhattan", "Scaled Manhattan (통계 기준선)"), ("Vanilla1DCNN", "Vanilla 1D-CNN (기본)"),
                        ("Vanilla1DCNN_Aug", "Vanilla + 증강"), ("Vanilla1DCNN_Aug_ASNorm", labels["vanilla_17ch_aug"]),
                        ("KeyPrintNet_9ch", "KeyPrintNet 9채널 (타이밍)"), ("KeyPrintNet", labels["keyprint_17ch"])]:
        m = models[name]
        negative = m["impostor_breakdown"]
        rows.append((label, f"{m['macro_f1']:.4f}", f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}",
                     pct(m["eer"]), pct(negative["human_only"]["far"]), pct(negative["bot_only"]["far"]), pct(m["frr"])))
    metric_labels = {"roc_auc": "ROC-AUC", "eer": "EER", "macro_f1": "Macro F1"}
    result = [
        "### 5.1 고정 암호 1:1 검증 (CMU Test 10명)",
        "**평가 프로토콜**: 학습에 포함되지 않은 CMU Test 피험자 10명에서 1인당 5회 타건으로 등록 템플릿을 생성한 뒤, **타 세션 본인 입력 `1,800`건**, **동일 암호를 직접 타이핑한 실측 타인 입력 `1,600`건**, **합성 매크로 봇 입력 `200`건**을 비교했습니다. 모든 모델의 오수락률(`FAR`)과 본인 거부율(`FRR`)은 `Validation` 분할에서만 확정한 임계값을 그대로 적용해 측정했습니다.",
        table(["모델", "Macro F1 ↑", "ROC-AUC ↑", "PR-AUC ↑", "EER ↓", "인간 FAR ↓", "합성 봇 FAR ↓", "본인 FRR ↓"], rows),
        f"- **모델 간 비교 해석**: 고정 암호 단일 과업에서는 키 위치가 매번 동일하므로 순수 타이밍만 집중하는 **KeyPrintNet 9채널**이 가장 높은 성적(`Macro F1 0.8578`, `ROC-AUC 0.9300`, `EER 14.33%`)을 기록했습니다.\n"
        f"- **통제 베이스라인 대비 신뢰구간**: 자유 문장까지 함께 처리하는 `KeyPrintNet 17채널`(`Macro F1 {proposed['macro_f1']:.4f}`)과 동일 증강·`AS-Norm`을 적용한 통제 `Vanilla`(`{models['Vanilla1DCNN_Aug_ASNorm']['macro_f1']:.4f}`)의 차이를 피험자 10명 단위 부트스트랩으로 검정한 결과, 아래 표처럼 95% 신뢰구간이 모두 `0`을 포함해 고정 암호 단독 성능은 통계적으로 동등한 수준입니다.",
        table(["지표 (17채널 − 통제 Vanilla)", "점추정 차이", "95% 신뢰구간"], [(metric_labels.get(k, k), f"{v['difference']:+.4f}", f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]") for k,v in report["proposed_vs_augmented_control"]["metrics"].items()]),
        "> **참고 (전체 비교쌍 검산)**: 고정된 40개 등록 템플릿에 대해 가능한 모든 인간 비교 조합(본인 `14,000`건, 타인 `144,000`건 — 동일 피험자 반복 참여 포함)을 전수 계산한 결과도 별도 리포트에 수록했습니다. [전수 비교 및 20개 시드 민감도](models/trial_sensitivity.json)",
        "![고정 암호 ROC와 등록 수별 성능](assets/figures/fig3_roc_det_kshot.png)",
        "![Validation 임계값을 적용한 Test 혼동행렬](assets/figures/fig4_confusion_matrices.png)",
        "### 5.2 64차원 초구면 임베딩 군집 분포",
        "CMU Test 피험자 10명의 타건(1인당 45개)과 합성 봇 입력 50개를 `64차원` 단위 초구면에 사상해 코사인 실루엣 계수와 2차원 `t-SNE` 분포를 비교했습니다. 제안 모델에서 동일 인물의 서로 다른 세션 타건이 한 군집으로 응집되고 합성 봇이 외곽으로 분리됨을 확인할 수 있습니다.",
        "![시험 피험자별 임베딩과 모델별 실루엣](assets/figures/fig5_embedding_tsne.png)",
        "### 5.3 자유 문장 1:N 모델 선택 및 독립 Test 비교",
        f"**최종 문장 인코더 선정 근거 (`Validation` 기준)**: 자유 문장 식별 인코더는 Test 결과를 보지 않고 **Validation 10명의 영문 25자 교차 세션 Top-1 정확도**(후보당 질의 `{audit['clean_25_char_cross_session']['validation'][deployed]['num_queries']:,}`건, 5문장 등록 + 각 모델의 Train 코호트 `AS-Norm` 동일 적용)로 선택했습니다. 그 결과 **{labels[deployed]}**이 선정되었으며, 독립 Test 18명에서도 통제 베이스라인 대비 `Top-1`이 **+3.47%p**(`58.63%` vs `55.16%`) 앞섰습니다.",
    ]
    clean = audit["clean_25_char_cross_session"]
    result.append(table(["문장 인코더 후보", "Validation Top-1", "Test Top-1 ↑", "Test Top-3 ↑", "Test Macro F1 ↑"],
                        [(labels[k], pct(v["top1_accuracy"]), pct(clean["test"][k]["top1_accuracy"]),
                          pct(clean["test"][k]["top3_accuracy"]), f"{clean['test'][k]['macro_f1']:.4f}") for k,v in clean["validation"].items()]))
    result += [
        f"**웹 배포 프로토콜(`68자`, 5문장 등록 · 2문장 질의) 오픈셋 검증**: 실제 웹 타자연습과 동일한 조건(등록자 5명 + 미등록 외부인 입력 포함)에서 두 후보의 행동 가중치와 수락 임계값을 공통 Validation 정답 수락률 목표(`{pct(features['arena_protocol']['keyboard']['validation_target_correct_accept'])}`)에 맞춰 공정 비교했습니다. 채택된 **KeyPrintNet 17채널**은 등록자 오식별률 **`0.00%`**, 외부인 오수락률 **`0.10%`** 수준으로 훨씬 안전한 오픈셋 판별력을 보였습니다.",
        table(["후보 모델 (68자 키보드 · 5인 등록)", "Test 등록자 정답 수락률 ↑", "등록자 오식별률 ↓", "외부인 오수락률 ↓"],
              [(labels[k], pct(v["gallery_5"]["two_sentences_enroll5_margin"]["correct_accept"]),
                pct(v["gallery_5"]["two_sentences_enroll5_margin"]["misidentified"]),
                pct(v["gallery_5"]["two_sentences_enroll5_margin"]["outsider_accepted"]))
               for k,v in audit["web_68_char_keyboard"]["test"].items()]),
        "최종 채택된 문장 모델의 입력 기기별 오픈셋 수락 규칙(1위 점수 하한 및 1·2위 점수 간격 동시 충족)과 Test 성적은 다음과 같습니다.",
        table(["입력 환경", "1위 점수 하한", "1·2위 간격 하한", "등록자 정답 수락률", "외부인 오수락률"],
              [(label, f"{v['rule']['threshold']:.2f}", f"{v['rule']['margin']:.2f}",
                pct(v['gallery_5']['two_sentences_enroll5_margin']['correct_accept']),
                pct(v['gallery_5']['two_sentences_enroll5_margin']['outsider_accepted']))
               for name,label in (("keyboard","물리 키보드"),("touch","모바일 터치 모사")) for v in [features['arena_protocol'][name]]]),
        "![고정문과 영문 문장의 다중 인물 식별](assets/figures/fig7_typist_identification.png)",
        "### 5.4 오류 사례 분석 및 세션 드리프트",
    ]
    e = report["error_analysis"]
    result += [f"CMU 인간 타건 검증에서 발생한 오류는 **본인 거부(`FRR`) {e['num_false_rejections']}건**, **타인 수락(`FAR`) {e['num_false_acceptances']}건**이었습니다. 아래 표와 파형 그래프는 각 오류 유형 중 예측 점수가 가장 크게 빗나간 실제 Test 사례를 분석한 결과입니다(초기 세션 등록 후 숙련도 상승으로 타자 속도가 급변한 사례 및 평균 템포·롤오버 패턴이 우연히 겹친 사칭 사례).",
               table(["오류 유형", "대상 피험자", "등록 세션", "질의 샘플 행", "예측 유사도"], [
                   (label, e[k]["target_subject"], e[k]["enroll_session"], e[k]["sample_idx"], f"{e[k]['score']:.4f}")
                   for k,label in (("worst_fn","최악 본인 거부 (False Reject)"),("worst_fp","최악 타인 수락 (False Accept)"))]),
               "![본인 거부·타인 수락 사례의 타건 파형과 세션별 비교](assets/figures/fig6_error_analysis_and_ablation.png)",
               "### 5.5 채널 확장(9ch → 17ch), 행동 프로필 및 터치 모사 절제 실험 (Ablation)",
               "`KeyPrintNet` 9채널과 17채널에 동일한 데이터 분할, 학습 에포크(`35`), 템포 증강 및 **`18%` 터치 모사 증강**(누름 시간을 가리고 `Hold_Valid=0` 처리)을 적용해 각 요소의 기여도를 분리 측정했습니다(Test 18인 폐쇄형, 68자 문장 3개 등록 · 1개 질의 기준). 키 위치·운지 기하 정보를 추가한 `17채널`이 교차 세션 `Top-1`을 **`64.14% → 77.74%`** (`+13.60%p`) 끌어올렸으며, 모바일 터치 모사에서도 **`43.43% → 59.20%`** (`+15.77%p`)까지 큰 폭의 향상을 보였습니다."]
    rows = []
    for mode,bank in features["web_faithful_identification"].items():
        legacy = "legacy_9ch" if mode == "keyboard" else "legacy_9ch_imputed_holds"
        for k,label in ((legacy,"9채널 (순수 타이밍)"),("keyprint_17ch","17채널 (+키 위치·유효성)"),("keyprint_17ch_behavior","17채널 + 행동 프로필"),("behavior_only","행동 프로필 단독")):
            if k in bank:
                m = bank[k]
                rows.append(("물리 키보드" if mode == "keyboard" else "모바일 터치 모사",label,pct(m["same_session"]["top1_accuracy"]),pct(m["cross_session"]["top1_accuracy"]),pct(m["cross_session"]["top3_accuracy"])))
    result += [table(["입력 기기", "비교 구성", "동일 세션 Top-1 ↑", "교차 세션 Top-1 ↑", "교차 세션 Top-3 ↑"], rows),
               "**보조 검증 (KeyRecs 고정문 시간 열 해석 비교)**: §3.1에서 발견한 `KeyRecs` 고정문 CSV의 열 이름 뒤바뀜(`DD↔UU`, `DU↔UD`)이 학습에 미치는 영향을 확인하기 위해, 동일 조건(`17채널`, `35 Epochs`)에서 공개 열 이름 그대로 학습한 대조 모델을 별도로 훈련해 비교했습니다. 물리 관계식으로 보정한 학습이 검증 `EER`과 문장 `Top-3` 모두 소폭 우수함을 확인했습니다. [시간 열 검산 및 민감도 상세](models/timestamp_sensitivity.json)",
               table(["KeyRecs 고정문 학습 열 기준", "CMU Val Macro F1", "CMU Val EER ↓", "25자 교차 세션 Val Top-1", "Val Top-3"],
                     [(label, f"{m['cmu_canonical_password']['macro_f1']:.4f}", pct(m["cmu_canonical_password"]["eer"]),
                       pct(m["clean_25_char_cross_session"]["top1_accuracy"]), pct(m["clean_25_char_cross_session"]["top3_accuracy"]))
                      for k,label in (("published_labels","공개 열 이름 그대로 사용"),("release_label_swap","Train 물리 관계식으로 보정 (채택)"))
                      for m in [timestamps["matched_17ch_training_comparison"]["validation"][k]]]),
               "![채널·행동 특징 및 터치 모사 조건별 비교](assets/figures/fig8_feature_upgrade.png)",
               "### 5.6 사용자 지정 암호 전이 평가 및 듀얼 인코더 보정"]
    pw = features["password_user_strings"]; km = pw["synthesis_metadata"]["test"]; kt = pw["test"]["kb"]
    result += [
        "**왜 듀얼 인코더(`9채널 + 17채널`)를 결합했는가?**: 사용자가 직접 정한 임의의 암호 문자열에서는 단일 `9채널` 모델만 쓰면 키 위치 정보를 놓치고, `17채널` 모델만 쓰면 특정 암호 철자에 따른 배경 코호트 편향이 생길 수 있습니다. 이를 해결하기 위해 두 인코더의 점수를 타인 분포 기준으로 각각 표준화해 합산하는 **듀얼 보정**(`dualPasswordScore`)을 설계했습니다.",
        f"- **원시 이벤트 완전 분리 합성 검증**: 실제 자유 필사 로그의 디그래프를 조합해 Test용 단어형 암호 `{len(pw['strings']['test']['word'])}`종과 무작위 문자열 `{len(pw['strings']['test']['random'])}`종(총 `{km['generated_attempts']:,}`개 입력)을 구성했습니다. 동일 세션 본인 질의 `{km['same_day_generated_inputs_verified']:,}`건 전수 검사 결과, 등록 템플릿과 원시 키 이벤트가 겹친 사례는 **`0`건**이었습니다(키보드 본인 비교 `{kt['num_genuine']:,}`건, 타인 비교 `{kt['num_impostor']:,}`건).",
        table(["입력 환경", "점수 산출 방식", "ROC-AUC ↑", "EER ↓", "타인 수락률 (FAR) ↓", "본인 거부율 (FRR) ↓"],
              [(m,method,f"{pw['test'][k][a]['roc_auc']:.4f}",pct(pw["test"][k][a]["eer"]),pct(pw["test"][k][a]["far"]),pct(pw["test"][k][a]["frr"]))
               for k,m in (("kb","물리 키보드"),("touch","모바일 터치 모사")) for a,method in (("deployed_as_norm","9채널 단독 (AS-Norm)"),("dual_raw","듀얼 인코더 결합 (9ch + 17ch)"))]),
        "합성 암호가 아닌 **실제 연속 타이핑 데이터**(CMU, KeyRecs 고정문, 필사 중 자연 반복된 단어)에서도 동일한 듀얼 보정 규칙을 적용해 일반화 성능을 교차 확인했습니다(CMU 타인 비교에는 합성 봇 `200`건 포함). [비밀번호 보정 실험 전체](models/password_calibration.json)",
        table(["실측 평가 시나리오", "본인 / 타인 비교 건수", "9채널 단독 AUC", "듀얼 결합 AUC ↑", "듀얼 FAR / FRR"],
              [(label, f"{v['num_genuine']:,} / {v['num_impostor']:,}", f"{v['as_norm_auc']:.4f}", f"{v['dual_auc']:.4f}", f"{pct(v['dual']['far'])} / {pct(v['dual']['frr'])}")
               for k,label in (("cmu_test_kb","CMU 고정 암호"),("krfixed_test_kb","KeyRecs 고정 문자열"),("words_same_day_kb","필사 내 반복 단어 (동일 세션)"),("words_cross_day_kb","필사 내 반복 단어 (교차 세션)"))
               for v in [calibration["scenarios"][k]]]),
    ]
    adapt = features["password_adaptation"]
    result += [f"**템플릿 자동 적응(`Adaptation`)의 득실 비교**: 시간이 지나 타자 속도가 빨라질 때 발생하는 본인 거부(`FRR`)를 줄이기 위해, 검증 데이터 비용 합이 최소(`{adapt['validation_policy_costs'][adapt['chosen_policy']]:.4f}`)인 **`{adapt['chosen_policy']}`**(충분한 여유 점수로 통과한 본인 입력만 반영) 정책을 선택했습니다. 그 결과 2차 세션 말기 본인 거부율은 **`34.29% → 0.00%`** 수준으로 해소되었으나, 타인 수락률(`FAR`)이 `18.75% → 25.28%`로 함께 상승하는 트레이드오프가 확인되어 두 지표를 함께 보고합니다. [갱신 정책 전체 비교](models/evaluation_report.json)",
               table(["키보드 템플릿 운영 정책", "Test 타인 수락률 (FAR) ↓", "2차 세션 말기 본인 거부율 (FRR) ↓"],
                     [(label,pct(m["far"]),pct(m["frr_s2_reps_91_100"]))
                      for k,label in (("dual_no_adapt","고정 템플릿 (갱신 없음)"),(adapt["chosen_policy"],"자동 적응 템플릿 (채택 정책)"))
                      for m in [adapt["test"]["kb"][k]]]),
               "### 5.7 비밀번호 길이별 보정 및 대문자 입력 습관 결합",
               "짧은 암호일수록 타이밍 전이 수가 적어 변별 정보량이 급감하므로, 웹 서비스의 **신규 암호 등록 최소 길이를 12자 이상**으로 제한했습니다(기본 탑재된 CMU 체험 샘플 4종만 원 데이터 규격인 `10자`·`9`전이 사용). 또한 Validation 10명에서 암호 길이(`8`·`10`·`12`·`14`·`16`·`20자`)별 점수 분포 기준표를 적합해 중간 길이를 선형 보간함으로써, 암호가 길어질 때 고정 임계값에서 오수락(`FAR`)이 치솟는 현상을 바로잡았습니다.",
               table(["암호 길이", "키보드 AUC ↑", "고정 임계값 FAR / FRR", "길이별 보정 FAR / FRR (키보드)", "길이별 보정 FAR / FRR (터치)"],
                     [(f"{length}자", f"{v['kb']['roc_auc']:.4f}", f"{pct(v['kb']['fixed_6_10']['far'])} / {pct(v['kb']['fixed_6_10']['frr'])}",
                       f"{pct(v['kb']['length_aware']['far'])} / {pct(v['kb']['length_aware']['frr'])}",
                       f"{pct(v['touch']['length_aware']['far'])} / {pct(v['touch']['length_aware']['frr'])}")
                      for length,v in features["password_length_calibration"]["test"].items()])]
    cap = features["password_capital_method"]
    result += [f"- **대문자 입력 방식(`Shift` vs `CapsLock`) 결합 효과**: 물리 키보드에서 대문자를 `Shift`로 치는지 `CapsLock`으로 치는지의 개인 습관 우도비(`LLR`)를 `12자` 암호 검증에 결합하자, 본인 거부율 증가를 최소화(`{pct(cap['test_without']['frr'])} → {pct(cap['test_with']['frr'])}`)하면서 **타인 수락률(`FAR`)을 `{pct(cap['test_without']['far'])} → {pct(cap['test_with']['far'])}`로 크게 낮췄습니다**. [타건 스타일 분산 및 정보량 분석](models/typing_style_analysis.json)"]
    return {"goals":"\n\n".join(goals), "evaluation":"\n\n".join(result)}


def korean_study(report):
    roster = report["enrolled_probe_roster_size"]
    main, pr = report["main"], report["candidate"]
    trials = pr["enrolled_probes"]["trials"]
    writers = report["participants"]
    unknown_trials = pr["unknown_writer_probes"]["trials"]
    rows = []
    for label, population, key in (
        ("작성자가 1위 (보류 전 순위)", "enrolled_probes", "top1_before_rejection"),
        ("작성자로 최종 수락", "enrolled_probes", "accepted_correct"),
        ("다른 등록자로 오수락", "enrolled_probes", "accepted_wrong"),
        ("미등록 가능성으로 판정 보류", "enrolled_probes", "rejected"),
        ("미등록 외부인을 등록자로 오수락", "unknown_writer_probes", "false_accept"),
    ):
        a, b = main[population][key], pr[population][key]
        rows.append((label, f"{a['rate']:.1%} ({a['count']}건)", f"{b['rate']:.1%} ({b['count']}건)"))

    input_length = report["protocol"]["source_prefix_keystrokes"]
    setup = (f"참가자 {writers}명을 {roster}명씩 조로 나누고, 각 기록의 처음 {input_length}타(데모 한국어 문장의 두벌식 타수 중앙값)만 추출해 평가했습니다. "
             f"1인당 {report['enroll_shots']}개 기록으로 프로필을 등록한 뒤, 별도 시험 기록을 {report['query_shots']}개씩 묶은 {trials}건에서 작성자를 판별했습니다. "
             f"또한 작성자가 속하지 않은 다른 {roster}명 후보 조와 비교해 미등록 외부인 오수락률도 {unknown_trials}건에서 함께 측정했습니다.")
    difference = report["paired_change"]["top1_before_rejection"]
    low, high = difference["conditional_writer_bootstrap_95_ci_pp"]
    interpretation_md = (
        f"- **순위 정확도 비교**: 수락·보류 기준을 적용하기 전 실제 작성자가 1위에 오른 비율은 `{main['enrolled_probes']['top1_before_rejection']['rate']:.1%} → {pr['enrolled_probes']['top1_before_rejection']['rate']:.1%}`(`{difference['candidate_minus_main_pp']:+.1f}%p`)였으며, 참가자 단위 부트스트랩 95% 신뢰구간은 `{low:+.1f}%p`\\~`{high:+.1f}%p`로 순위 자체의 유의미한 변화로 단정하기는 어렵습니다.\n"
        f"- **오수락 차단과 보류 증가의 원리**: 기존 공개본은 1위 점수 `{100 * main['threshold']:.0f}점`만 넘으면 수락(1·2위 간격 조건 `{100 * main['margin']:.0f}점`)해 미등록 외부인의 `87.6%`를 등록자로 잘못 수락했습니다. 반면 현재 배포본(`PR #5`)은 **1위 점수 `{100 * pr['threshold']:.0f}점` 이상 + 2위와의 격차 `{100 * pr['margin']:.0f}점` 이상**을 동시에 요구하여 타인 오식별을 **`0.0%`**, 외부인 오수락을 **`13.2%`** 수준으로 대폭 줄였습니다(보류된 `{pr['enrolled_probes']['rejected']['count']}`건 중 `{pr['enrolled_probes']['rejected_by_margin']['count']}`건은 1위 점수는 넘었으나 1·2위 격차 기준에서 보류됨)."
    )
    interpretation_html = (
        f"수락·보류 기준 적용 전 작성자가 1위에 오른 비율 차이는 {difference['candidate_minus_main_pp']:+.1f}%p(참가자 단위 95% 신뢰구간 {low:+.1f}%p~{high:+.1f}%p)로, 순위 자체의 우열을 단정하기는 어렵습니다. "
        f"대신 기존 공개본(1위 {100 * main['threshold']:.0f}점 이상, 간격 {100 * main['margin']:.0f}점)에서 87.6%에 달하던 미등록 외부인 오수락을 막기 위해, "
        f"현재 버전(PR #5)은 1위 점수 {100 * pr['threshold']:.0f}점 이상과 1·2위 격차 {100 * pr['margin']:.0f}점 이상을 동시에 요구합니다. "
        f"그 결과 다른 등록자로 잘못 수락한 비율은 0.0%가 되었고, 보류된 {pr['enrolled_probes']['rejected']['count']}건 중 {pr['enrolled_probes']['rejected_by_margin']['count']}건은 1위 점수는 통과했으나 2위와의 격차가 작아 보류됐습니다."
    )
    crowded = report["full_roster"]
    longer = report["full_recording"]
    stress_md = (
        f"- **후보 인원 및 입력 길이 변화**: 비교 후보를 {crowded['enrolled_probe_roster_size']}명 전원으로 늘리면 1위 정답률은 기존 `{crowded['main']['enrolled_probes']['top1_before_rejection']['rate']:.1%}`, 현재 `{crowded['candidate']['enrolled_probes']['top1_before_rejection']['rate']:.1%}`로 낮아지며 오픈셋 판정은 `{crowded['candidate']['enrolled_probes']['rejected']['count']}/{trials}`건 모두 보류했습니다. 반대로 {roster}명 후보에서 평균 약 `235타`인 원본 기록 전체를 넣으면 1위 정답률이 기존 `{longer['main']['enrolled_probes']['top1_before_rejection']['rate']:.1%}`, 현재 `{longer['candidate']['enrolled_probes']['top1_before_rejection']['rate']:.1%}`까지 상승했습니다."
    )
    stress_html = (
        f"비교 후보를 {crowded['enrolled_probe_roster_size']}명 전원으로 늘리면 1위 정답률은 기존 {crowded['main']['enrolled_probes']['top1_before_rejection']['rate']:.1%}, 현재 버전(PR #5) {crowded['candidate']['enrolled_probes']['top1_before_rejection']['rate']:.1%}로 낮아지고 오픈셋 판정은 {crowded['candidate']['enrolled_probes']['rejected']['count']}/{trials}건 모두 보류했습니다. "
        f"반대로 {roster}명 후보에서 평균 약 235타인 원래 기록 전체를 사용하면 1위 정답률은 기존 {longer['main']['enrolled_probes']['top1_before_rejection']['rate']:.1%}, 현재 버전 {longer['candidate']['enrolled_probes']['top1_before_rejection']['rate']:.1%}까지 높아졌습니다."
    )
    headings = (f"{roster}인 후보 기준 지표", "기존 공개본 (간격 0점)", "현재 배포본 (PR #5, 간격 12점)")
    md = "\n\n".join([setup, table(headings, rows), interpretation_md, stress_md])
    html_rows = "\n".join(
        f'            <tr><th scope="row">{label}</th><td>{a}</td><td>{b}</td></tr>' for label, a, b in rows)
    html = f'''        <p>{setup}</p>
        <table aria-label="후보 다섯 명에서 측정한 작성자 식별">
          <thead><tr><th scope="col">{headings[0]}</th><th scope="col">{headings[1]}</th><th scope="col">{headings[2]}</th></tr></thead>
          <tbody>
{html_rows}
          </tbody>
        </table>
        <p>{interpretation_html}</p>
        <p>{stress_html}</p>'''
    return md, html


def korean_ranking_summary(research):
    def load(name):
        return json.loads((research / name).read_text(encoding="utf-8"))

    same_day = load("results.json")
    pilot = load("ranking-pilot.json")["comparisons"]
    validation, = (study for study in same_day["studies"]
                   if (study["split"], study["question_selection"], study["prefix_text_taps"]) == ("VS", "chronological", 40))
    rates = validation["results"]["pr"]["mean_over_all_five_person_rosters"]
    rows = []
    for key, label in [("validation", "AI Hub 국어 답안"), ("fact_ko", "FACT 한국어"), ("fact_en", "FACT 영어")]:
        case = pilot[key]
        scores = case["models"]
        rows.append([f"{label} ({case['writers']}명)", *[pct(scores[method]["top1"]) for method in (
            "deployed_pr", "same_frozen_encoder_before_cohort_normalization", "frozen_encoder_plus_learned_candidate_weights")]])
    contrast = pilot["validation"]["models"]["frozen_encoder_plus_learned_candidate_weights"]
    low, high = contrast["paired_95_ci_change"]
    return "\n\n".join([
        f"AI Hub 중학생 국어 서술형 답안(평가용 `VS` 분할 `{validation['writers']}`명)에서도 같은 날 먼저 작성한 5문항(문항당 첫 `40타`)으로 등록하고, 이후 작성한 2문항을 묶어 가능한 모든 5인 후보 조합의 평균 성적을 측정했습니다(FACT는 언어별 50명, 고정 5인 조 250건 비교).",
        table(["데이터셋 (5인 후보 기준 Top-1)", "현재 배포 점수 (AS-Norm)", "코호트 보정 생략 (순수 코사인)", "한국어 순위 가중치 추가 학습"], rows),
        f"- **오픈셋 수락률과 UI 분리 근거**: AI Hub `VS`에서 배포 점수의 1위 정답률은 **`{pct(rates['top1'])}`**(무작위 기준 `20%`의 약 2.9배)였으나, 미등록자 차단 기준을 적용하면 정답 수락 `{pct(rates['accept_correct'])}`, 판정 보류 `{pct(rates['reject_known'])}`, 미등록자 오수락 `{pct(rates['accept_outsider'])}`였습니다.\n"
        f"- **보조 순위 가중치 실험 결론**: 인코더를 고정한 채 AI Hub 학습 분할(`333명`)로 7개 점수 특징의 순위 가중치를 학습했을 때 `VS` Top-1 변화는 `{contrast['change_vs_pr'] * 100:+.2f}%p`(학생 단위 95% 구간 `{low * 100:+.2f}%p`\\~`{high * 100:+.2f}%p`)로 일관된 유의성이 확인되지 않아, 웹 데모는 기존 배포 점수를 유지하되 **예상 작성자 1위 이름**과 **오픈셋 수락·보류 판정**을 분리해 함께 표시하도록 설계했습니다. [한국어 연구 기록 및 전체 집계](research/korean-typing/README.md)",
    ])


def replace_block(text, name, body):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one pair of {name} markers")
    prefix, rest = text.split(start, 1)
    _, suffix = rest.split(end, 1)
    return prefix + start + "\n\n" + body + "\n\n" + end + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated results without rewriting any file.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def read_json(name):
        return json.loads((root / "models" / name).read_text(encoding="utf-8"))
    report = read_json("evaluation_report.json")
    blocks = result_block(report, read_json("trial_sensitivity.json"), read_json("password_calibration.json"), read_json("timestamp_sensitivity.json"))
    blocks["korean-results"], study_table = korean_study(read_json("korean_fact_ab.json"))
    blocks["korean-ranking"] = korean_ranking_summary(root / "research" / "korean-typing")
    study_path = root / "web" / "korean-study.html"
    study_page = study_path.read_text(encoding="utf-8")
    updated_study = replace_block(study_page, "korean-results", study_table)
    path = root / "README.md"
    original = path.read_text(encoding="utf-8")
    updated = original
    for key, body in blocks.items():
        updated = replace_block(updated, key, body)
    submission_path = root / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    value = report["models"]["KeyPrintNet"]["macro_f1"]
    if args.check:
        if updated != original or updated_study != study_page or submission["primary_metric"]["value"] != value:
            raise SystemExit("Published results differ from the saved evaluations")
    else:
        path.write_text(updated, encoding="utf-8")
        study_path.write_text(updated_study, encoding="utf-8")
        submission["primary_metric"]["value"] = value
        submission_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
