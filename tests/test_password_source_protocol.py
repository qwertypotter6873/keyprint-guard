"""Regressions for password source events and held-out calibration."""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from password_synth import (  # noqa: E402
    CALIB_LENGTHS,
    covered,
    covered_disjoint_s1,
    digraph_tables,
    partition_tab_disjoint,
    run_password_synthesis,
    synthesise,
)
from preprocessing import N_CHANNELS  # noqa: E402


def events(tokens, seed):
    rng = np.random.default_rng(seed)
    t = 0.0
    result = []
    for token in tokens:
        for label in list(token) + ["Space"]:
            result.append((label, t, float(rng.uniform(.045, .15))))
            t += float(rng.uniform(.09, .30))
    return result


def endpoints(occurrences):
    return {(person, session, i + k) for person, session, i in occurrences for k in (0, 1)}


class PasswordSourceProtocolTests(unittest.TestCase):
    def test_adjacent_digraphs_and_final_hold_use_disjoint_source_events(self):
        sessions = {("val", day): events(["abc", "abc"], day) for day in (1, 2)}
        sessions[("test", 1)] = events(["abc"], 3)
        tabs = digraph_tables(sessions, ["test", "val"])
        tab = tabs[("val", 1)]
        self.assertEqual(tab, digraph_tables(sessions, ["val"])[("val", 1)])
        ab = tab[("a", "b")][0].start
        bc = tab[("b", "c")][0].start
        self.assertNotEqual(ab, bc)
        self.assertEqual(endpoints([ab]) & endpoints([bc]), {bc})

        for seed in range(20):
            rng = np.random.default_rng(seed)
            enrollment, query = partition_tab_disjoint(tab, "abc", rng)
            enrollment_sources = endpoints([occ.start for pool in enrollment.values() for occ in pool])
            query_sources = endpoints([occ.start for pool in query.values() for occ in pool])
            self.assertTrue(enrollment_sources.isdisjoint(query_sources))
            for attempts in (enrollment, query):
                holds, _, starts = synthesise(attempts, "abc", rng)
                raw = sessions[("val", 1)]
                expected = [raw[i][2] for _, _, i in starts] + [raw[starts[-1][2] + 1][2]]
                np.testing.assert_allclose(holds, expected)

    def test_impossible_source_split_is_detected_without_discarding_a_feasible_one(self):
        sessions = {("p", day): events(["aaaaa"], day) for day in (1, 2)}
        tabs = digraph_tables(sessions, ["p"])
        self.assertTrue(covered(tabs[("p", 1)], "aaa", min_per_occ=2))
        self.assertFalse(covered_disjoint_s1(tabs, "p", "aaa"))
        with self.assertRaisesRegex(ValueError, "source-disjoint"):
            partition_tab_disjoint(tabs[("p", 1)], "aaa", np.random.default_rng(42))

        sessions[("p", 1)] = events(["aaaaaa"], 1)
        tabs = digraph_tables(sessions, ["p"])
        self.assertTrue(covered_disjoint_s1(tabs, "p", "aaa"))
        a, b = partition_tab_disjoint(tabs[("p", 1)], "aaa", np.random.default_rng(42))
        self.assertEqual(len(a[("a", "a")]), 2)
        self.assertEqual(len(b[("a", "a")]), 2)
        self.assertTrue(endpoints([occ.start for occ in a[("a", "a")]]).isdisjoint(endpoints([occ.start for occ in b[("a", "a")]])))

    def test_all_validation_stages_are_invariant_to_held_out_people_and_source_order(self):
        split = {"train": ["r0", "r1"], "val": [f"v{i}" for i in range(6)], "test": [f"t{i}" for i in range(7)]}
        alphabet = "abcde"
        rng = np.random.default_rng(7)
        vocab = sorted({"".join(word) for word in rng.choice(list(alphabet), size=(95, 8))})
        held_out_vocab = sorted({"".join(word) for word in rng.choice(list(alphabet), size=(20, 8))} - set(vocab))
        pairs = [a + b for a, b in itertools.product(alphabet, repeat=2)] * 8
        capitals = [("Shift", "A")] * 6 + [("CapsLock", "A", "A", "CapsLock")] * 3
        sessions = {}
        for i, p in enumerate(split["train"] + split["val"] + split["test"]):
            tokens = pairs + capitals + (vocab if p in split["train"] else [])
            if p == "t0":
                tokens += held_out_vocab * 2
            for day in (1, 2):
                sessions[(p, day)] = events(tokens, 2 * i + day)

        def unit(a):
            return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)

        def encoder(channels):
            projection = rng.normal(size=(channels, 8))
            return lambda tensors: unit(tensors[:, :channels].mean(axis=-1) @ projection)

        encode9, encode17 = encoder(9), encoder(N_CHANNELS)
        cohort = unit(rng.normal(size=(16, 8)))

        def run(raw, people):
            return run_password_synthesis(
                raw, people, encode9, encode17, {"H": (.01, .5), "DD": (.01, 1.5), "UD": (-.4, 1.4)},
                np.zeros(N_CHANNELS), np.ones(N_CHANNELS), .09, {"kb": cohort, "touch": cohort}, {"kb": .5, "touch": .5},
                seed=42, frr_target=.10, min_length=12,
            )

        full = run(sessions, split)
        remaining_sessions = {key: value for key, value in reversed(list(sessions.items())) if key[0] != "t0"}
        reduced = run(remaining_sessions, {**split, "test": split["test"][1:][::-1]})
        for stage in full:
            self.assertEqual(full[stage]["strings"]["val"], reduced[stage]["strings"]["val"])
            self.assertEqual(full[stage]["synthesis_metadata"]["val"], reduced[stage]["synthesis_metadata"]["val"])

        base = full["password_user_strings"]
        length = full["password_length_calibration"]
        capital = full["password_capital_method"]
        for words in base["strings"].values():
            self.assertTrue(set(words["word"]).isdisjoint(held_out_vocab))
        self.assertEqual(base["dual_calibration"], reduced["password_user_strings"]["dual_calibration"])
        self.assertEqual(length["table"], reduced["password_length_calibration"]["table"])
        for field in ("weight", "tau_offset", "caps_share", "validation_far_by_weight", "validation_frr_by_weight"):
            self.assertEqual(capital[field], reduced["password_capital_method"][field])

        self.assertEqual(base["synthesis_metadata"]["test"]["num_participants_scored"], 7)
        self.assertEqual(reduced["password_user_strings"]["synthesis_metadata"]["test"]["num_participants_scored"], 6)
        self.assertEqual({int(k) for k in length["strings"]["val"]}, set(CALIB_LENGTHS))
        all_words = {}
        for sp, capital_count in (("val", 12), ("test", 15)):
            self.assertEqual({k: len(v) for k, v in base["strings"][sp].items()}, {"word": 30, "random": 30})
            self.assertTrue(all(len(words) == 15 for words in length["strings"][sp].values()))
            self.assertEqual(len(capital["strings"][sp]), capital_count)
            all_words[sp] = (
                set().union(*base["strings"][sp].values(), *length["strings"][sp].values(), capital["strings"][sp])
            )
            for audit in (base["synthesis_metadata"][sp], *length["synthesis_metadata"][sp].values(), capital["synthesis_metadata"][sp]):
                self.assertTrue(audit["disjoint_source_events_enforced"])
                self.assertGreater(audit["same_day_generated_inputs_verified"], 0)
        self.assertTrue(all_words["val"].isdisjoint(all_words["test"]))


if __name__ == "__main__":
    unittest.main()
