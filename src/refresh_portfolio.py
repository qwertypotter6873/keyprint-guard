"""Update the marked README results and submission metric from the evaluated checkpoints."""
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
        "최초에 설정한 목표를 고정했습니다. 공식 표본은 시드 42를 사용합니다. 앞선 프로젝트 버전에서도 같은 시험 참가자를 평가했습니다.",
        table(["과업·조건", "지표", "목표", "Test 결과", "달성 여부"], [
            (name, metric, sign + fmt(limit), fmt(measured), "달성" if passed else "미달")
            for name, metric, sign, limit, measured, fmt, passed in (
                ("CMU 고정 암호", "Macro F1", "≥ ", .85, proposed["macro_f1"], lambda x: f"{x:.4f}", proposed["macro_f1"] >= .85),
                ("CMU 고정 암호", "ROC-AUC", "≥ ", .92, proposed["roc_auc"], lambda x: f"{x:.4f}", proposed["roc_auc"] >= .92),
                ("CMU 고정 암호", "EER", "≤ ", .15, proposed["eer"], pct, proposed["eer"] <= .15),
                ("고정 문구 교차 세션 10인", "Top-1", "≥ ", .40, cmu["top1_accuracy"], pct, cmu["top1_accuracy"] >= .40),
                ("고정 문구 교차 세션 10인", "Top-3", "≥ ", .88, cmu["top3_accuracy"], pct, cmu["top3_accuracy"] >= .88),
                ("영문 25자 동일 세션 18인", "Top-1", "≥ ", .55, sentence["top1_accuracy"], pct, sentence["top1_accuracy"] >= .55),
                ("영문 25자 동일 세션 18인", "Top-3", "≥ ", .88, sentence["top3_accuracy"], pct, sentence["top3_accuracy"] >= .88),
            )]),
        f"같은 CMU Test 10명에서 질의만 다시 뽑았을 때 Macro F1이 0.85 이상인 시드는 **{goal_count}/{draws}개**였습니다. 중앙값은 {summary['median']:.5f}, 범위는 {summary['min']:.4f}–{summary['max']:.4f}입니다. 이전 공개 버전은 2/20개였습니다. [시드별 측정값](models/trial_sensitivity.json)",
        "웹 목표는 인코더당 1 MiB 이하, 한 번의 추론 지연 5 ms 이하입니다. 파일 크기는 §8에, 기기별 지연은 데모의 판정 결과에 표시합니다.",
    ]

    rows = []
    for name, label in [("ScaledManhattan", "Scaled Manhattan"), ("Vanilla1DCNN", "Vanilla"),
                        ("Vanilla1DCNN_Aug", "Vanilla + 증강"), ("Vanilla1DCNN_Aug_ASNorm", labels["vanilla_17ch_aug"]),
                        ("KeyPrintNet_9ch", "KeyPrintNet 9채널"), ("KeyPrintNet", labels["keyprint_17ch"])]:
        m = models[name]
        negative = m["impostor_breakdown"]
        rows.append((label, f"{m['macro_f1']:.4f}", f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}",
                     pct(m["eer"]), pct(negative["human_only"]["far"]), pct(negative["bot_only"]["far"]), pct(m["frr"])))
    result = [
        "### 5.1 고정 암호 1:1 검증",
        "CMU Test 10명에서 등록자별 5회 타건으로 템플릿을 만들었습니다. 검증에는 타 세션 본인 비교 1,800건, 같은 암호를 친 실측 타인 비교 1,600건, 합성 입력 비교 200건을 사용했습니다. 각 모델의 FAR와 FRR은 Validation에서 결정한 임계값에서 측정했습니다.",
        table(["모델", "Macro F1 ↑", "ROC-AUC ↑", "PR-AUC ↑", "EER ↓", "인간 FAR ↓", "합성 FAR ↓", "본인 FRR ↓"], rows),
        f"공식 시드에서 17채널의 Macro F1은 {proposed['macro_f1']:.4f}, 증강·AS-Norm 통제 모델은 {models['Vanilla1DCNN_Aug_ASNorm']['macro_f1']:.4f}입니다. 대표 지표 목표 통과 횟수는 §1.3에 제시했습니다. 등록자 10명을 짝지어 복원추출한 모델 차이의 95% 구간은 다음과 같습니다.",
        table(["17채널 − 증강 Vanilla", "차이", "95% 구간"], [(k, f"{v['difference']:+.4f}", f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]") for k,v in report["proposed_vs_augmented_control"]["metrics"].items()]),
        "정해진 40개 등록 템플릿에서 가능한 인간 비교 전체는 본인 14,000건, 타인 144,000건입니다. 같은 사람이 여러 번 참여합니다. [전체 비교와 20개 시드 결과](models/trial_sensitivity.json)",
        "![고정 암호 ROC와 등록 수별 성능](assets/figures/fig3_roc_det_kshot.png)",
        "![Validation 임계값을 적용한 Test 혼동행렬](assets/figures/fig4_confusion_matrices.png)",
        "### 5.2 임베딩의 분포",
        "CMU 시험 피험자 10명의 타건을 1인당 45개씩 뽑아 64차원 공간의 코사인 실루엣을 계산했습니다. 2차원 t-SNE에는 합성 입력 50개도 함께 표시했습니다.",
        "![시험 피험자별 임베딩과 모델별 실루엣](assets/figures/fig5_embedding_tsne.png)",
        "### 5.3 문장 모델 선택과 독립 Test 비교",
        f"선택 규칙은 **Validation 10명의 25자 영문 교차 세션 Top-1**입니다. 각 후보는 같은 5개 문장으로 등록하고 자신의 Train 코호트로 AS-Norm을 적용합니다. 동점이면 사전에 정한 우선순위를 사용합니다. 선택한 문장 인코더는 **{labels[deployed]}**입니다. 아래 두 후보의 Validation 질의는 각각 {audit['clean_25_char_cross_session']['validation'][deployed]['num_queries']:,}건입니다.",
    ]
    clean = audit["clean_25_char_cross_session"]
    result.append(table(["후보", "Validation Top-1", "Test Top-1", "Test Top-3", "Test Macro F1"],
                        [(labels[k], pct(v["top1_accuracy"]), pct(clean["test"][k]["top1_accuracy"]),
                          pct(clean["test"][k]["top3_accuracy"]), f"{clean['test'][k]['macro_f1']:.4f}") for k,v in clean["validation"].items()]))
    result += [
        f"68자 키보드 배포 조건에서는 두 후보 모두 등록 5문장·질의 2문장, 등록자 5명으로 평가했습니다. 각 후보의 행동 가중치와 임계값은 공통 Validation 정답 수락률 목표 {pct(features['arena_protocol']['keyboard']['validation_target_correct_accept'])}에서 확정했습니다. Test 외부인은 등록 목록에 없는 시험 참가자입니다.",
        table(["후보", "Test 등록자 정답 수락률 ↑", "등록자 오식별률 ↓", "외부인 오수락률 ↓"],
              [(labels[k], pct(v["gallery_5"]["two_sentences_enroll5_margin"]["correct_accept"]),
                pct(v["gallery_5"]["two_sentences_enroll5_margin"]["misidentified"]),
                pct(v["gallery_5"]["two_sentences_enroll5_margin"]["outsider_accepted"]))
               for k,v in audit["web_68_char_keyboard"]["test"].items()]),
        "선택한 모델의 기기별 판정 기준과 등록자 5명 조건의 Test 결과는 다음과 같습니다. 점수와 1·2위 간격을 모두 충족하면 수락합니다.",
        table(["입력", "1위 점수 하한", "1·2위 간격 하한", "등록자 정답 수락률", "외부인 오수락률"],
              [(label, f"{v['rule']['threshold']:.2f}", f"{v['rule']['margin']:.2f}",
                pct(v['gallery_5']['two_sentences_enroll5_margin']['correct_accept']),
                pct(v['gallery_5']['two_sentences_enroll5_margin']['outsider_accepted']))
               for name,label in (("keyboard","키보드"),("touch","터치 모사")) for v in [features['arena_protocol'][name]]]),
        "![고정문과 영문 문장의 다중 인물 식별](assets/figures/fig7_typist_identification.png)",
        "### 5.4 오류 사례와 세션 변화",
    ]
    e = report["error_analysis"]
    result += [f"사람의 타건 비교에서 본인 거부 {e['num_false_rejections']}건, 타인 수락 {e['num_false_acceptances']}건이 발생했습니다. 아래는 각 오류 유형에서 점수가 가장 크게 빗나간 실제 시험 입력입니다.",
               table(["오류", "등록 피험자", "등록 세션", "질의 행", "유사도"], [
                   (label, e[k]["target_subject"], e[k]["enroll_session"], e[k]["sample_idx"], f"{e[k]['score']:.4f}")
                   for k,label in (("worst_fn","본인 거부"),("worst_fp","타인 수락"))]),
               "![본인 거부·타인 수락 사례의 타건 파형과 세션별 비교](assets/figures/fig6_error_analysis_and_ablation.png)",
               "### 5.5 키 위치, 행동 프로필과 터치 입력",
               "9채널과 17채널 KeyPrintNet에 같은 분할, 학습 횟수, 템포 증강과 **18% 터치 모사 증강**을 적용했습니다. 17채널은 키 위치와 홀드 관측 여부를 함께 사용합니다. 아래는 68자 문장 3개 등록·1개 질의의 18인 폐쇄형 비교입니다. 터치는 데스크톱 타건에서 누름 시간을 가린 모사입니다."]
    rows = []
    for mode,bank in features["web_faithful_identification"].items():
        legacy = "legacy_9ch" if mode == "keyboard" else "legacy_9ch_imputed_holds"
        for k,label in ((legacy,"9채널"),("keyprint_17ch","17채널"),("keyprint_17ch_behavior","17채널 + 행동"),("behavior_only","행동 단독")):
            if k in bank:
                m = bank[k]
                rows.append(("키보드" if mode == "keyboard" else "터치 모사",label,pct(m["same_session"]["top1_accuracy"]),pct(m["cross_session"]["top1_accuracy"]),pct(m["cross_session"]["top3_accuracy"])))
    result += [table(["기기", "모델", "동일 세션 Top-1", "교차 세션 Top-1", "교차 세션 Top-3"], rows),
               "고정문 시간 열의 학습 효과는 17채널 모델 두 개를 같은 시드·증강·35에포크로 학습해 확인했습니다. 두 모델의 CMU 암호와 25자 문장 Validation 입력은 동일합니다. 고정문 열별 해석은 Train의 물리 관계식으로 결정했습니다.",
               table(["고정문 학습 열", "CMU Val Macro F1", "CMU Val EER", "25자 교차 세션 Val Top-1", "Top-3"],
                     [(label, f"{m['cmu_canonical_password']['macro_f1']:.4f}", pct(m["cmu_canonical_password"]["eer"]),
                       pct(m["clean_25_char_cross_session"]["top1_accuracy"]), pct(m["clean_25_char_cross_session"]["top3_accuracy"]))
                      for k,label in (("published_labels","공개 열 이름"),("release_label_swap","Train 물리식으로 보정"))
                      for m in [timestamps["matched_17ch_training_comparison"]["validation"][k]]]),
               "고정 가중치에 Test 입력 해석만 바꾸는 별도의 비교도 기록했습니다. [시간 열 검산과 민감도](models/timestamp_sensitivity.json)",
               "![채널·행동 특징 및 터치 모사 조건별 비교](assets/figures/fig8_feature_upgrade.png)",
               "### 5.6 사용자 지정 암호와 문장 배포 설정"]
    pw = features["password_user_strings"]; km = pw["synthesis_metadata"]["test"]; kt = pw["test"]["kb"]
    result += [
        "단어형 암호와 무작위 문자열을 실제 자유 타이핑의 키 쌍으로 합성했습니다. 피험자·문자열마다 같은 세션의 등록과 본인 질의에 쓰이는 원시 키 이벤트를 분리합니다. Validation의 보정 문장·길이표·대문자 기준을 먼저 확정하고 Test는 독립 난수로 생성합니다.",
        f"Test에서 단어 {len(pw['strings']['test']['word'])}종과 무작위 문자열 {len(pw['strings']['test']['random'])}종으로 입력 {km['generated_attempts']:,}개를 만들었습니다. 같은 세션의 본인 질의 {km['same_day_generated_inputs_verified']:,}개 모두에서 등록에 사용한 키 이벤트와의 교집합은 0개였습니다. 키보드 표본의 본인 비교는 {kt['num_genuine']:,}건, 타인 비교는 {kt['num_impostor']:,}건입니다. 하나의 입력을 여러 템플릿과 비교합니다.",
        table(["입력", "방식", "AUC", "EER", "타인 수락률", "본인 거부율"],
              [(m,method,f"{pw['test'][k][a]['roc_auc']:.4f}",pct(pw["test"][k][a]["eer"]),pct(pw["test"][k][a]["far"]),pct(pw["test"][k][a]["frr"]))
               for k,m in (("kb","키보드"),("touch","터치 모사")) for a,method in (("deployed_as_norm","9채널 AS-Norm"),("dual_raw","두 인코더 결합"))]),
        "실측 타건에서도 같은 보정을 평가했습니다. CMU 타인 비교에는 합성 입력 200건이 포함됩니다. [보정 비교 전체](models/password_calibration.json)",
        table(["실측 조건", "본인 / 타인 비교", "9채널 AUC", "듀얼 AUC", "듀얼 FAR / FRR"],
              [(label, f"{v['num_genuine']:,} / {v['num_impostor']:,}", f"{v['as_norm_auc']:.4f}", f"{v['dual_auc']:.4f}", f"{pct(v['dual']['far'])} / {pct(v['dual']['frr'])}")
               for k,label in (("cmu_test_kb","CMU"),("krfixed_test_kb","KeyRecs 고정문"),("words_same_day_kb","반복 단어 동일 세션"),("words_cross_day_kb","반복 단어 교차 세션"))
               for v in [calibration["scenarios"][k]]]),
    ]
    adapt = features["password_adaptation"]
    result += [f"암호 템플릿의 자동 갱신 정책은 키보드와 터치의 Validation 비용 합으로 결정했습니다. 최종 정책은 **{adapt['chosen_policy']}**, 선택 비용은 {adapt['validation_policy_costs'][adapt['chosen_policy']]:.4f}입니다. 다음 Test에서는 본인 입력으로만 템플릿을 갱신한 뒤 동일한 타인 입력을 비교했습니다.",
               table(["키보드 정책", "Test 타인 수락률", "2차 세션 말기 본인 거부율"],
                     [(label,pct(m["far"]),pct(m["frr_s2_reps_91_100"]))
                      for k,label in (("dual_no_adapt","고정 템플릿"),(adapt["chosen_policy"],"선택한 정책"))
                      for m in [adapt["test"]["kb"][k]]]),
               "두 오류율을 함께 고려해야 합니다. [모든 갱신 정책의 비교](models/evaluation_report.json)",
               "### 5.7 비밀번호 길이와 대문자: 12자 이상 + 길이별 기준 + 대문자 습관",
               "신규 암호의 최소 길이는 12자입니다. Validation 10명에서 8·10·12·14·16·20자 보정표를 만들고 중간 길이는 보간합니다. 기본 CMU 샘플 4개는 10자이며, 샘플 템플릿과 브라우저 입력 모두 제출용 Enter를 제외해 9개 전이를 사용합니다.",
               table(["길이", "키보드 AUC", "고정 FAR / FRR", "길이 보정 FAR / FRR", "터치 보정 FAR / FRR"],
                     [(length, f"{v['kb']['roc_auc']:.4f}", f"{pct(v['kb']['fixed_6_10']['far'])} / {pct(v['kb']['fixed_6_10']['frr'])}",
                       f"{pct(v['kb']['length_aware']['far'])} / {pct(v['kb']['length_aware']['frr'])}",
                       f"{pct(v['touch']['length_aware']['far'])} / {pct(v['touch']['length_aware']['frr'])}")
                      for length,v in features["password_length_calibration"]["test"].items()])]
    cap = features["password_capital_method"]
    result += [f"대문자 방식 표지를 추가한 12자 합성 시험에서 타인 수락률은 {pct(cap['test_without']['far'])} → {pct(cap['test_with']['far'])}, 본인 거부율은 {pct(cap['test_without']['frr'])} → {pct(cap['test_with']['frr'])}였습니다. 표지는 각 피험자의 세션별 Shift·CapsLock 사용 비율에서 얻었습니다.",
               "공개 데이터의 두 KeyRecs 세션은 날짜 간격이 알려져 있지 않습니다. 한글 IME와 모바일 입력 경로는 기능 시험으로 확인했으며, 한국어 화자와 실물 스마트폰의 식별률에는 별도 수집이 필요합니다. [입력 방식별 분석](models/typing_style_analysis.json)"]
    return {"goals":"\n\n".join(goals), "evaluation":"\n\n".join(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the README and submission.json without rewriting either.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def read_json(name):
        return json.loads((root / "models" / name).read_text(encoding="utf-8"))
    report = read_json("evaluation_report.json")
    blocks = result_block(report, read_json("trial_sensitivity.json"), read_json("password_calibration.json"), read_json("timestamp_sensitivity.json"))
    path = root / "README.md"
    original = path.read_text(encoding="utf-8")
    updated = original
    for key, body in blocks.items():
        start, end = f"<!-- {key}:start -->", f"<!-- {key}:end -->"
        if updated.count(start) != 1 or updated.count(end) != 1:
            raise ValueError(f"Expected one pair of {key} markers in README.md")
        prefix, rest = updated.split(start, 1)
        _, suffix = rest.split(end, 1)
        updated = prefix + start + "\n\n" + body + "\n\n" + end + suffix
    submission_path = root / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    value = report["models"]["KeyPrintNet"]["macro_f1"]
    if args.check:
        if updated != original or submission["primary_metric"]["value"] != value:
            raise SystemExit("README results or submission.json differ from the evaluated checkpoints")
    else:
        path.write_text(updated, encoding="utf-8")
        submission["primary_metric"]["value"] = value
        submission_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
