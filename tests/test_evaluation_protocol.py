"""Evaluation regressions: held-out attacks, prefix inputs and paired policies."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation_protocol import balanced_impostor_indices, paired_person_auc_ci
from preprocessing import prefix_sequences
from train_and_evaluate import build_verification_trials, TEST_BOT_SEED
from password_synth import select_adaptation_policy, synthetic_input_counts, adaptation_eval


class EvaluationProtocolTests(unittest.TestCase):
    def test_test_bots_change_without_changing_human_trials(self):
        split = SimpleNamespace(subjects=["a", "b"], subject_ids=np.repeat([0, 1], 64),
                                sessions=np.tile(np.repeat(np.arange(1, 9), 8), 2),
                                reps=np.tile(np.arange(1, 9), 16))
        old = build_verification_trials(split, np.zeros(17), np.ones(17))
        new = build_verification_trials(split, np.zeros(17), np.ones(17), bot_seed=TEST_BOT_SEED)
        for a, b in zip(old["groups"], new["groups"]):
            for key in ("enroll_idx", "gen_idx", "imp_idx", "bot_idx"):
                np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse({r.tobytes() for r in old["bot_flat"]} & {r.tobytes() for r in new["bot_flat"]})

    def test_prefix_cannot_observe_suffix(self):
        h = np.array([[.1, .12, .13, .2]], np.float32)
        d = np.array([[.2, .3, .8]], np.float32)
        bounds = {"H": (.01, 1), "DD": (.01, 2), "UD": (-1, 2)}
        def enc(h, d):
            return prefix_sequences(h, d, d-h[:, :-1], np.zeros((7, 3)), 2, bounds, np.zeros(17), np.ones(17))
        expected = enc(h, d)
        h[:, -1] = .9
        d[:, -1] = 1.5
        np.testing.assert_array_equal(expected, enc(h, d))
        self.assertAlmostEqual(float(expected[0, 5, 0]), .8, places=6)

    def test_balanced_sample_covers_all_people_and_is_order_invariant(self):
        index = {(p, 1, k): i*10+k for i, p in enumerate("abcd") for k in range(6)}
        a = balanced_impostor_indices(index, "a", "word")
        b = balanced_impostor_indices(dict(reversed(list(index.items()))), "a", "word")
        self.assertEqual(a, b)
        self.assertEqual(Counter(i//10 for i in a), {1: 2, 2: 2, 3: 2})

    def test_paired_identity_bootstrap_has_zero_difference_for_same_model(self):
        targets = np.repeat(list("abc"), 3)
        queries = np.tile(list("abc"), 3)
        y = (targets == queries).astype(int)
        scores = np.array([.8, .5, .3, .2, .4, .6, .1, .7, .9])
        ci = paired_person_auc_ci(y, scores, scores, targets, queries, n_boot=30)
        self.assertEqual((ci["ci95_low"], ci["ci95_high"]), (0, 0))
        self.assertEqual(ci["num_people"], 3)

    def test_disabled_adaptation_can_win_validation(self):
        metrics = {"dual_no_adapt": {"frr_one": .2, "far": .2},
                   "dual_adapt_margin_1.5": {"frr_one": .01, "far": .8}}
        chosen, _ = select_adaptation_policy({"kb": metrics, "touch": metrics}, [1.5])
        self.assertEqual(chosen, "dual_no_adapt")

    def test_input_counts_deduplicate_modes_and_template_comparisons(self):
        row = {"label": 0, "attempt_id": "query1", "enroll_attempt_ids": ["enroll1"]}
        counts = synthetic_input_counts([row, row, row])
        self.assertEqual(counts["comparison_trials"], 3)
        self.assertEqual(counts["generated_attempts"], 2)
        self.assertEqual(counts["generated_impostor_query_inputs"], 1)

    def test_adaptation_uses_paired_attack_queries(self):
        n = 3*2*100
        h = np.full((n, 4), .1, dtype=np.float32)
        d = np.full((n, 3), .2, dtype=np.float32)
        p = np.repeat(list("abc"), 200)
        sessions = np.tile(np.repeat([1, 2], 100), 3)
        reps = np.tile(np.arange(100), 6)
        def embed(tensors):
            e = np.zeros((len(tensors), 64)); e[:, 0] = 1
            return e
        prm = {"mu9": 0, "sd9": 1, "mu17": 0, "sd17": 1, "tau": 1}
        result = adaptation_eval(h, d, p, sessions, reps, "word", embed, embed,
                                 {"H": (.01, 1), "DD": (.01, 2), "UD": (-1, 2)},
                                 np.zeros(17), np.ones(17), .1, {"kb": prm}, {"kb": .5},
                                 {"kb": np.eye(64)[:12]}, [0, 1.5], np.random.default_rng(42), "kb")
        self.assertEqual(len({v["paired_impostor_sha256"] for v in result.values()}), 1)
        self.assertTrue(all(v["num_impostor_trials"] == 120 for v in result.values()))


if __name__ == "__main__":
    unittest.main()
