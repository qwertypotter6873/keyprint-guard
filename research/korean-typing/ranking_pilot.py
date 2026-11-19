"""Fit and test the candidate ranker with the frozen encoder scores."""

from math import comb

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


QUANTILES = [.10, .25, .50, .75, .90]
SEED = 71518
DRAWS = 32
PENALTY = .01

def timing_profiles(study):
    def fingerprint(captures):
        # Equal weight per answer, then equal weight per writer's probe.
        return np.mean([np.quantile(np.log(c['stroke']['dds']), QUANTILES) for c in captures], axis=0)

    refs, probes, owners = [], [], []
    for i, writer in enumerate(study['participants']):
        a, b = [s['captures'] for s in writer['sessions']]
        assert len(a) == 5 and len(b) % 2 == 0
        refs.append(fingerprint(a))
        for j in range(0, len(b), 2):
            probes.append(fingerprint(b[j:j+2]))
            owners.append(i)
    distances = -np.abs(np.asarray(probes)[:, None, :] - np.asarray(refs)[None, :, :])
    return distances, np.asarray(owners)


def assemble_features(study, scores, protocol):
    distances, owners = timing_profiles(study)
    deployed = np.asarray(scores['matrix'])
    raw = np.asarray(scores['rawMatrix'])
    if 'fixed' in scores:
        np.testing.assert_array_equal(owners, [row['writerIndex'] for row in scores['fixed']['genuine']])
    assert deployed.shape == raw.shape == distances.shape[:2]
    assert all(np.isfinite(values).all() for values in (deployed, raw, distances))
    return np.dstack((deployed, raw, distances)), owners, protocol


def fit_ranker(features):
    # Each training writer has one combined query. Every five-way training trial
    # contains its correct template and four different, uniformly drawn writers.
    n, targets, dims = features.shape
    assert n == targets
    mu = features.mean(axis=(0, 1))
    scale = np.maximum(features.std(axis=(0, 1)), 1e-8)
    rng = np.random.default_rng(SEED)
    all_indices = np.arange(n)
    groups = np.empty((n, DRAWS, 5), dtype=int)
    for i in range(n):
        groups[i, :, 0] = i
        others = all_indices[all_indices != i]
        for j in range(DRAWS):
            groups[i, j, 1:] = rng.choice(others, 4, replace=False)
    selected = features[all_indices[:, None, None], groups].reshape(-1, 5, dims)
    differences = (selected - selected[:, :1]) / scale

    def objective(w):
        scores = differences @ w
        denom = logsumexp(scores, axis=1)
        probabilities = np.exp(scores - denom[:, None])
        grad = np.einsum('ij,ijk->k', probabilities, differences) / len(differences)
        return denom.mean() + PENALTY * w.dot(w) / 2, grad + PENALTY * w

    solution = minimize(objective, np.zeros(dims), jac=True, method='L-BFGS-B',
                        options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 300})
    if not solution.success:
        raise RuntimeError(solution.message)
    return mu, scale, solution.x


def writer_top1(scores, owners, protocol):
    probs = []
    n = scores.shape[1]
    for row, author in zip(scores, owners, strict=True):
        if protocol == 'published_project_fixed_rosters':
            start = 5 * (author // 5)
            probs.append(float(start + int(np.argmax(row[start:start+5])) == author))
        else:
            # Reproduce stable index-based app tie breaking; uniform four distractors.
            after_true = int(np.sum(row < row[author]) + np.sum(row[author+1:] == row[author]))
            probs.append(comb(after_true, 4) / comb(n-1, 4))
    return np.bincount(owners, weights=probs, minlength=n) / np.bincount(owners, minlength=n)


def evaluate(data, fitted):
    x, owners, protocol = data
    mu, scale, learned = fitted
    z = (x - mu) / scale
    models = {
        'deployed_pr': x[:, :, 0],
        'same_frozen_encoder_before_cohort_normalization': x[:, :, 1],
        'timing_quantiles_without_fitted_weights': x[:, :, 2:].mean(axis=2),
        # Same signals as the learned scorer with equal weight for the embedding
        # and timing families and no writer-pair supervision. Scaling uses TS.
        'fixed_equal_family_weights': .5 * (z[:, :, :2].mean(axis=2) + z[:, :, 2:].mean(axis=2)),
        'frozen_encoder_plus_learned_candidate_weights': z @ learned,
    }
    return {key: writer_top1(values, owners, protocol) for key, values in models.items()}


def summarize(values):
    base = values['deployed_pr']
    raw = values['same_frozen_encoder_before_cohort_normalization']
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(base), size=(5000, len(base)))
    def rounded(number):
        return round(float(number), 6) + 0.0

    def interval(delta):
        return [rounded(v) for v in np.quantile(delta[samples].mean(axis=1), [.025,.975])]

    result = {}
    for model, by_writer in values.items():
        delta = by_writer - base
        diff_raw = by_writer - raw
        result[model] = dict(top1=rounded(by_writer.mean()), change_vs_pr=rounded(delta.mean()),
                             paired_95_ci_change=interval(delta), change_vs_raw=rounded(diff_raw.mean()),
                             paired_95_ci_change_vs_raw=interval(diff_raw))
    return result


def run(cases, hashes, original_touch_rule, revision):
    training, authors, gallery = cases['train']
    np.testing.assert_array_equal(authors, np.arange(333))
    fit = fit_ranker(training)
    results = {}
    for name in ['validation', 'nonoverlapping_prompts', 'fact_ko', 'fact_en']:
        features, subjects, protocol = cases[name]
        values = evaluate(cases[name], fit)
        results[name] = dict(writers=features.shape[1], probes=len(subjects), roster_protocol=protocol,
                             models=summarize(values))
    # Sanity check: use the original deployed ranks, point-for-point, on all tests.
    for key, expected in [('validation', .586100), ('nonoverlapping_prompts', .639310),
                          ('fact_ko', .748), ('fact_en', .720)]:
        assert results[key]['models']['deployed_pr']['top1'] == expected
    # Additional writer-disjoint out-of-fold comparison entirely inside TS.
    cv = {}
    for k in range(3):
        train = np.flatnonzero(authors % 3 != k)
        test = np.flatnonzero(authors % 3 == k)
        cv_model = fit_ranker(training[np.ix_(train, train)])
        comparison = evaluate((training[np.ix_(test, test)], np.arange(len(test)), gallery), cv_model)
        for key, value in comparison.items():
            cv.setdefault(key, []).extend(value)
    results['three_fold_writer_disjoint_TS'] = dict(writers=len(authors), probes=len(authors),
        roster_protocol='all five-person rosters within the held-out fold, with 111 eligible writers per fold',
        models=summarize({k: np.array(v) for k, v in cv.items()}))
    return dict(pr_5=revision, aihub_archives_sha256=hashes,
        fitted_on='333 AI Hub TS writers, Korean school subject, first 40 text taps; five earlier questions to enroll then two later questions in a joint probe; no VS or FACT data used in fitting.',
        primary_comparison='Frozen PR rank vs. a seven-weight scorer learned on TS for five-candidate ranking. No neural weight updates. Same paired probes and five-person rosters. No threshold or rejection applied in these top1 results.',
        features=['deployed normalized score', 'raw 0.65 centroid + 0.35 best exemplar cosine',
                  *[f'negative absolute difference in mean recording log-DD quantile {p}' for p in QUANTILES]],
        learning=dict(loss='five-way conditional log likelihood plus L2/2', l2=PENALTY, candidate_draws_per_writer=DRAWS,
                      draws_seed=SEED, hyperparameter_search=False),
        uncertainty='Bootstrap resamples writers, paired, conditional on the gallery, training fit and data. AI Hub secondary prompt-holdout may include primary VS writers; FACT Korean and English share people and use 76 taps.',
        original_touch_rule=original_touch_rule, comparisons=results)
