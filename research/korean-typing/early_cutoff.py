"""Reproduce the earlier FACT Korean two-fold cutoff experiment."""

import hashlib
from itertools import product

import numpy as np


CURRENT = (0.78, 0.12)
DEPLOYED = '현재 PR'
NO_MARGIN = '같은 모델에서 1·2위 차이 조건만 해제'
ERROR_BUDGET = '검증에서 기존 두 오수락 횟수 이내로 정답 수락 최대화'
EQUAL_WEIGHT = '등록자·미등록자 정답에 같은 가중치'


def observe(matrix, groups, pairs):
    strongest, leads, correct, outside_scores, outside_leads = [], [], [], [], []
    for index, roster in enumerate(groups):
        next_roster = groups[(index + 1) % len(groups)]
        for writer in roster:
            scores = matrix[writer, :pairs, :]
            known = scores[:, roster]
            rank = np.argsort(-known, axis=1, kind='stable')
            best = np.take_along_axis(known, rank[:, :2], axis=1)
            outside = np.sort(scores[:, next_roster], axis=1)
            strongest.extend(best[:, 0])
            leads.extend(best[:, 0] - best[:, 1])
            correct.extend(roster[rank[:, 0]] == writer)
            outside_scores.extend(outside[:, -1])
            outside_leads.extend(outside[:, -1] - outside[:, -2])
    return tuple(np.asarray(values) for values in (strongest, leads, correct, outside_scores, outside_leads))


def outcome(observations, threshold, margin):
    scores, leads, correct, outside_scores, outside_leads = observations
    known = (scores >= threshold) & (leads >= margin)
    unknown = (outside_scores >= threshold) & (outside_leads >= margin)
    return {
        'known_trials': len(known),
        'unknown_trials': len(unknown),
        'rank1_correct': int(correct.sum()),
        'accepted_correct': int((known & correct).sum()),
        'accepted_wrong': int((known & ~correct).sum()),
        'deferred_known': int((~known).sum()),
        'accepted_unknown': int(unknown.sum()),
    }


def choose_cutoffs(calibration):
    current = outcome(calibration, *CURRENT)
    grid = product((value / 100 for value in range(50, 101)), (value / 100 for value in range(16)))
    search = []
    for minimum, gap in grid:
        result = outcome(calibration, minimum, gap)
        distance = abs(minimum - CURRENT[0]) + abs(gap - CURRENT[1])
        search.append(((minimum, gap), result, -distance))

    eligible = [item for item in search if item[1]['accepted_wrong'] <= current['accepted_wrong']
                and item[1]['accepted_unknown'] <= current['accepted_unknown']]
    within_budget = max(eligible, key=lambda item: (
        item[1]['accepted_correct'], -item[1]['accepted_unknown'], -item[1]['accepted_wrong'], item[2],
    ))
    equal = max(search, key=lambda item: (
        item[1]['accepted_correct'] - item[1]['accepted_unknown'], -item[1]['accepted_wrong'],
        -item[1]['accepted_unknown'], item[2],
    ))
    return {DEPLOYED: CURRENT, ERROR_BUDGET: within_budget[0], EQUAL_WEIGHT: equal[0]}


def run(matrix, revision, published_report):
    matrix = np.asarray(matrix).reshape(50, 7, 50)
    assert np.isfinite(matrix).all()
    groups = np.arange(50).reshape(10, 5)
    all_typists = observe(matrix, groups, 5)
    baseline = outcome(all_typists, *CURRENT)
    original = {
        DEPLOYED: baseline,
        NO_MARGIN: outcome(all_typists, CURRENT[0], 0),
    }
    assert (baseline['rank1_correct'], baseline['accepted_correct'], baseline['accepted_unknown']) == (187, 19, 33)

    folds = []
    for parity in (0, 1):
        calibration = observe(matrix, groups[parity::2], 7)
        held_out = observe(matrix, groups[1-parity::2], 5)
        selected = choose_cutoffs(calibration)
        folds.append({
            label: {
                'minimum_score': minimum,
                'minimum_gap': gap,
                'calibration': outcome(calibration, minimum, gap),
                'held_out': outcome(held_out, minimum, gap),
            }
            for label, (minimum, gap) in selected.items()
        })
    cross = {
        label: {key: sum(fold[label]['held_out'][key] for fold in folds) for key in baseline}
        for label in (DEPLOYED, ERROR_BUDGET, EQUAL_WEIGHT)
    }
    return {
        '연구': 'FACT 공개 Galaxy S8 한국어 터치 간격, 76타, 등록 5개·질의 2개, 5인 후보',
        '판정_커밋': revision,
        '기존_공개_집계_SHA256': hashlib.sha256(published_report.encode()).hexdigest(),
        '방법': '가명 순서의 5인 조를 홀짝으로 나눠 25명씩 교차. 각 방향에서 기준 선택에만 25명 × 7쌍(기존 시험 5쌍 + 미사용 등록 기록 4개로 2쌍), 나머지 25명의 시험 5쌍씩으로 평가. 두 방향을 합쳐 같은 사람의 각 시험 입력을 한 번씩 집계. 올바른 등록자·오분류·보류의 분모는 250, 다른 조에 대한 미등록자 시험 분모도 250. 기존 연구 자료의 재분석이며 신규 외부 검증은 아님.',
        '미등록_조': '각 분할 안에서 다음 5인 조로 순환하여 평가. 원 공개 집계와 미등록 비교 조합이 달라 기존 기준 미등록 오수락이 33건에서 34건으로 달라짐.',
        '후보기준': '최소 일치도 0.50부터 1.00, 최소 1·2위 차이 0.00부터 0.15, 각각 0.01 간격. 제한 기준은 등록자 오인식과 미등록자 오수락이 모두 현재 PR의 튜닝 분할 관측값 이내인 후보 중 등록자 정답 수락 최대화. 선택에 평가 분할 점수 미사용.',
        '공개_시험_그대로의_대조': original,
        '교차평가': cross,
        '분할별_선택과_결과': folds,
    }
