"""Five-person inference across rosters, with writers as the units of analysis."""
from itertools import combinations
from math import comb
import numpy as np


METRICS = ('top1', 'accept_correct', 'accept_wrong', 'reject_known',
           'reject_known_below_similarity', 'reject_known_by_margin', 'accept_outsider')


def one(scores, true, roster, threshold, gap):
    # Stable tie order agrees with the original JS sort for a gallery sorted by index.
    first, second, *_ = sorted(roster, key=lambda i: (-scores[i], i))
    good_score = scores[first] >= threshold
    accepted = good_score and scores[first] - scores[second] >= gap
    return first == true, bool(accepted), not good_score


def fixed(scores, true, plan, threshold, gap):
    top, accepted, below = one(scores, true, plan['enrolled'], threshold, gap)
    _, outsider, _ = one(scores, true, plan['unknown'], threshold, gap)
    return np.array([top, top and accepted, not top and accepted, not accepted, below,
                     not accepted and not below, outsider], dtype=float)


def exact(scores, true, threshold, gap):
    # The four distractors come uniformly from the other N-1 people. For an outsider,
    # draw five from the same N-1. Count combinations, without treating them as independent trials.
    n = len(scores)
    order = np.argsort(-scores, kind='stable')
    ranking = scores[order]
    position = int(np.flatnonzero(order == true)[0])
    known_den, unknown_den = comb(n - 1, 4), comb(n - 1, 5)
    correct, wrong, unknown = 0, 0, 0
    # For each possible winner find the first lower-ranked rival that also clears the margin.
    # This index moves monotonically even when scores tie. Direct subtraction matches the web engine.
    start = 1
    for k, score in enumerate(ranking):
        if score < threshold:
            break
        start = max(k + 1, start)
        while start < n and score - ranking[start] < gap:
            start += 1
        weaker = n - start
        if k == position:
            correct = comb(weaker, 4)
        else:
            includes_true = position >= start
            wrong += comb(weaker - 1, 3) if includes_true else 0
            unknown += comb(weaker - int(includes_true), 4)
    top = comb(n - position - 1, 4) / known_den
    correct /= known_den
    wrong /= known_den
    rejected = 1.0 - correct - wrong
    below = (comb(int(np.sum(scores < threshold)) - 1, 4) / known_den) if scores[true] < threshold else 0.0
    return np.array([top, correct, wrong, max(0.0, rejected), below, max(0.0, rejected - below), unknown / unknown_den])


def verify_formulas():
    # Brute force every five-person gallery on small scores with ties and boundary scores.
    rng = np.random.default_rng(124)
    for scores in (np.array([0.4, 0.7, 0.7, 0.6, 0.8, 0.95, 0.1, 0.4]), rng.random(9)):
        for gap in (0, 0.12, 1):
            for threshold in (0.0, 0.7, 1.0):
                for idx in range(len(scores)):
                    others = [i for i in range(len(scores)) if i != idx]
                    known = [one(scores, idx, (idx, *r), threshold, gap) for r in combinations(others, 4)]
                    unknown = [one(scores, idx, r, threshold, gap) for r in combinations(others, 5)]
                    top, yes, below = np.array(known).T
                    expected = [top.mean(), (top & yes).mean(), (~top & yes).mean(), (~yes).mean(),
                                below.mean(), (~yes & ~below).mean(), np.mean([u[1] for u in unknown])]
                    np.testing.assert_allclose(exact(scores, idx, threshold, gap), expected, atol=1e-12)


def pack(values, names=METRICS):
    return {key: round(float(value), 6) for key, value in zip(names, values, strict=True)}


def bootstrap(values):
    # This interval is conditional on the observed candidate pool; people, not galleries or taps, are resampled.
    rng = np.random.default_rng(71518)
    n = len(values)
    means = np.empty((5000, values.shape[1]))
    for b in range(0, len(means), 100):
        idx = rng.integers(0, n, size=(100, n))
        means[b:b+100] = values[idx].mean(axis=1)
    return {k: [round(float(x), 6) for x in bounds]
            for k, bounds in zip(METRICS, np.quantile(means, [.025, .975], axis=0).T, strict=True)}


def summarize(job, speed):
    report = {}
    arrays = {}
    plan = job['plan']
    for name, release in [('main', job['main']), ('pr', job['pr']), ('pr_zero_margin_diagnostic', job['pr'])]:
        table = np.asarray(release['matrix'])
        n = len(plan)
        assert table.shape == (n, n) and np.isfinite(table).all()
        threshold = release['threshold']
        gap = 0 if name.endswith('diagnostic') else release['margin']
        rows = np.array([exact(scores, i, threshold, gap) for i, scores in enumerate(table)])
        same_groups = np.array([fixed(scores, i, plan[i], threshold, gap) for i, scores in enumerate(table)])
        assert rows.min() >= -1e-12 and rows.max() <= 1 + 1e-12
        np.testing.assert_allclose(rows[:, 1:4].sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(rows[:, 4:6].sum(axis=1), rows[:, 3], atol=1e-12)
        if not name.endswith('diagnostic'):
            # Verify independent five-person decisions against the actual released engine, once per writer.
            for i, (x, k, u) in enumerate(zip(same_groups, release['fixed']['genuine'], release['fixed']['unknown'], strict=True)):
                assert i == k['writerIndex'] == u['writerIndex']
                assert x[0] == k['top1'] and x[1]+x[2] == k['accepted']
                assert x[5] == k['marginRejection'] and x[6] == u['accepted']
        arrays[name] = rows
        report[name] = dict(default_similarity_threshold=threshold, applied_margin=gap,
                            mean_over_all_five_person_rosters=pack(rows.mean(axis=0)),
                            count_in_one_fixed_roster_per_writer={k: int(v) for k, v in zip(METRICS, same_groups.sum(axis=0), strict=True)})
        if not name.endswith('diagnostic'):
            report[name].update(writer_bootstrap_95_conditional_on_observed_gallery=bootstrap(rows),
                                revision=release['revision'], config_sha256=release['config_sha256'],
                                model_hash=release['model_hash'], onnx_runtime=release['runtime'])
    delta = arrays['pr'] - arrays['main']
    report['paired_pr_minus_main'] = dict(mean_rate_change=pack(delta.mean(axis=0)),
                                           writer_bootstrap_95_conditional_on_observed_gallery=bootstrap(delta))
    # A one-number comparator: distance in mean of per-question median log inter-tap time.
    speed_rows = []
    for i, (_, probe_speed) in enumerate(speed):
        scores = -np.abs(speed[:, 0] - probe_speed)
        speed_rows.append(exact(scores, i, -np.inf, 0)[0])
    ties = np.asarray(speed_rows)
    report['speed_only'] = dict(top1_mean_over_all_five_person_rosters=round(float(ties.mean()), 6))
    return report


def choose_cutoffs_on_development(matrix):
    """Exploratory cutoff fit. The separate VS matrix is never passed to this function."""
    limit = .10
    thresholds = np.arange(70, 101) / 100
    margins = np.arange(0, 21) / 100
    n = len(matrix)
    order = np.argsort(-matrix, axis=1, kind='stable')
    ranked = np.take_along_axis(matrix, order, axis=1)
    positions = np.argmax(order == np.arange(n)[:, None], axis=1)
    comb4 = np.array([comb(k, 4) for k in range(n + 1)], dtype=float)
    comb3 = np.array([comb(k, 3) for k in range(n + 1)], dtype=float)
    known_den, outsider_den = n * comb(n-1, 4), n * comb(n-1, 5)
    # Accumulate the weights of all galleries accepting each possible winner.
    masks = [(ranked >= threshold) for threshold in thresholds]
    best = None
    for gap in margins:
        correct, wrong, outsider = (np.zeros_like(ranked) for _ in range(3))
        for i, scores in enumerate(ranked):
            pos = positions[i]
            # Compute the safe lower-ranked rivals using the release's exact subtraction test.
            start = 1
            for k, score in enumerate(scores):
                start = max(start, k + 1)
                while start < n and score - scores[start] < gap:
                    start += 1
                weaker = n - start
                if pos == k:
                    correct[i, k] = comb4[weaker] / known_den
                else:
                    if pos >= start:
                        wrong[i, k] = comb3[weaker - 1] / known_den
                    outsider[i, k] = comb4[weaker - int(pos >= start)] / outsider_den
        for threshold, mask in zip(thresholds, masks, strict=True):
            far = outsider[mask].sum()
            if far > limit:
                continue
            key = (correct[mask].sum(), -wrong[mask].sum(), -far, threshold, gap)
            if best is None or key > best:
                best = key
    assert best is not None
    tar, _, far_negative, threshold, gap = best
    check = np.mean([exact(row, i, threshold, gap) for i, row in enumerate(matrix)], axis=0)
    np.testing.assert_allclose([tar, -far_negative], check[[1, 6]], atol=1e-10)
    return float(threshold), float(gap)


def calibration_report(train, validation):
    train_matrix = np.array(train['pr']['matrix'])
    threshold, gap = choose_cutoffs_on_development(train_matrix)
    summary = dict(development_outsider_acceptance_limit=.10, searched_similarity_values='0.70 to 1.00 in 0.01 increments',
                   searched_margin_values='0.00 to 0.20 in 0.01 increments',
                   objective='maximize correct acceptance, then minimize wrong known and outsider acceptance, then prefer larger cutoffs',
                   chosen_similarity_threshold=threshold, chosen_margin=gap)
    for part, matrix in [('TS', train_matrix), ('VS', np.asarray(validation['pr']['matrix']))]:
        values = np.array([exact(row, i, threshold, gap) for i, row in enumerate(matrix)])
        summary[part] = dict(writers=len(values), mean_over_all_five_person_rosters=pack(values.mean(axis=0)),
                             writer_bootstrap_95_conditional_on_observed_gallery=bootstrap(values))
    assert summary['TS']['mean_over_all_five_person_rosters']['accept_outsider'] <= .10
    return summary
