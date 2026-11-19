import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_trial_sensitivity import exhaustive_human_trials, trial_counts


class ExhaustiveTrialCountTests(unittest.TestCase):
    def test_cmu_full_counts_and_genuine_session_separation(self):
        split = SimpleNamespace(subject_ids=np.repeat(np.arange(10), 400),
                                sessions=np.tile(np.repeat(np.arange(1, 9), 50), 10))
        groups = [{"uid": uid, "enroll_session": session, "enroll_idx": np.arange(5)}
                  for uid in range(10) for session in (1, 3, 5, 7)]
        full = exhaustive_human_trials(split, {"groups": groups})
        self.assertEqual(trial_counts(full), {"enrollment_templates": 40, "genuine_comparisons": 14000,
                                             "human_impostor_comparisons": 144000, "bot_comparisons": 0})
        for g in full["groups"]:
            self.assertTrue(np.all(split.subject_ids[g["gen_idx"]] == g["uid"]))
            self.assertTrue(np.all(split.sessions[g["gen_idx"]] != g["enroll_session"]))
            self.assertTrue(np.all(split.subject_ids[g["imp_idx"]] != g["uid"]))


if __name__ == "__main__":
    unittest.main()
