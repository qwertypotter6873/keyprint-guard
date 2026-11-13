"""
Adversarial regression suite for Track A signal processing, split isolation,
controlled baselines, disjoint synthesis, schema v4 provenance, and clean-room
from-scratch training & non-retraining notebook execution.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavior_profile import Event, SESSION_BREAK_GAP_SEC, kind_of, load_keyrecs_sessions, parse_keyrecs_free_csv  # noqa: E402
from capture_sim import (  # noqa: E402
    RecorderSim,
    Stroke,
    _jsonable_stroke,
    build_input_tensor,
    extract_clean_stroke,
    extract_naive_unrepaired_stroke,
)
from keyboard_layout import code_for_label  # noqa: E402
from feature_eval import capture_stats, fixed_stroke_capture, stroke_payload  # noqa: E402
from password_synth import (  # noqa: E402
    capital_fusion,
    capital_method_rates,
    capital_trials,
    choose_strings,
    digraph_tables,
    string_trials,
    synthesis_metadata,
)
from preprocessing import (  # noqa: E402
    N_CHANNELS,
    SEED,
    augment_keystroke_batch,
    load_keyrecs_fixed_split,
    parse_keyrecs_sentence_corpus,
    prepare_full_pipeline,
    touchify_timings,
)
from train_and_evaluate import KEYPRINT_DEFAULT_EPOCHS, VANILLA_DEFAULT_EPOCHS  # noqa: E402


SCRATCH_VERIFY_DIR = Path(
    "/Users/bookend/.gemini/antigravity-cli/brain/d27dc495-df96-439d-8c05-28dba720a294/scratch/track_a_verify"
)


class TestLosslessKeyRecsParserAndWindows(unittest.TestCase):
    def test_real_keyrecs_free_csv_exact_counts(self) -> None:
        csv_path = REPO_ROOT / "data" / "raw" / "KeyRecs-free-text.csv"
        rows_by_ps, stats = parse_keyrecs_free_csv(csv_path)
        self.assertEqual(stats["total_lines"], 562596)
        self.assertEqual(stats["valid_transitions"], 562398)
        self.assertEqual(stats["session_end_strokes"], 198)
        self.assertEqual(stats["recovered_quote_rows"], 26)
        self.assertEqual(stats["corrupt_rows"], 0)
        self.assertEqual(len(rows_by_ps), 198)
        self.assertEqual(sum(len(v) for v in rows_by_ps.values()), 562596)

    def test_synthetic_quotes_session_end_and_corrupt_mid_session_break(self) -> None:
        header = (
            "participant,session,key1,key2,DU.key1.key1,DD.key1.key2,"
            "DU.key1.key2,UD.key1.key2,UU.key1.key2,\n"
        )
        # Session 1 & Session 2: includes unescaped `"`, quoted comma `","`, a mid-session corrupt row,
        # another clean run, and a session-end trailing stroke (`k2==""`, `dd==""`).
        lines = [
            header,
            'p001,1,a,",0.080,0.150,0.220,0.070,0.140,\n',
            'p001,1,",",",0.070,0.160,0.250,0.090,0.180,\n',
            'p001,1,",",b,0.090,0.140,0.220,0.050,0.130,\n',
            "p001,1,b,c,CORRUPT_HOLD,0.120,0.200,0.040,0.120,\n",
            "p001,1,x,y,0.085,0.130,0.210,0.045,0.125,\n",
            "p001,1,y,,0.095,,,,,\n",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "synthetic_free.csv"
            p.write_text("".join(lines), encoding="utf-8")

            rows, stats = parse_keyrecs_free_csv(p)
            self.assertEqual(stats["total_lines"], 6)
            self.assertEqual(stats["valid_transitions"], 4)
            self.assertEqual(stats["session_end_strokes"], 1)
            self.assertEqual(stats["recovered_quote_rows"], 2)
            self.assertEqual(stats["corrupt_rows"], 1)

            sessions = load_keyrecs_sessions(p)
            s1 = sessions[("p001", 1)]
            # Pre-corrupt keys: a (t=0.0), " (t=0.15), , (t=0.31)
            # Corrupt line b->c dropped + SESSION_BREAK_GAP_SEC inserted!
            # Post-corrupt keys: x (t = 0.31 + 0.14 + GAP), y (session-end stroke at +0.13 with hold=0.095)
            self.assertEqual([k for k, _, _ in s1], ["a", '"', ",", "x", "y"])
            self.assertAlmostEqual(s1[0][1], 0.0, places=6)
            self.assertAlmostEqual(s1[1][1], 0.150, places=6)
            self.assertAlmostEqual(s1[2][1], 0.310, places=6)
            self.assertGreaterEqual(s1[3][1] - s1[2][1], SESSION_BREAK_GAP_SEC)
            self.assertAlmostEqual(s1[4][1] - s1[3][1], 0.130, places=6)
            self.assertAlmostEqual(s1[4][2], 0.095, places=6)

            # RecorderSim / extract_clean_stroke must NOT treat across-corrupt-row transition `, -> x` as clean/unstitched
            rec = RecorderSim(touch=False)
            for label, t, h in s1:
                kd, _ = kind_of(label)
                code = "Backspace" if kd == "bs" else ("Space" if kd == "space" else code_for_label(label))
                ch = " " if kd == "space" else (label if kd == "char" else "")
                rec.feed(code, ch, t, h)
            stroke = rec.extract()
            self.assertNotIn([code_for_label(","), code_for_label("x")], stroke["pair_keys"])
            # Transition `x -> y` follows skipped `, -> x`, so it is marked stitched (`stitch_flags[-1] == True`)
            self.assertEqual(stroke["stitch_flags"], [False, False, True])

            # Also verify parse_keyrecs_sentence_corpus breaks `curr = []` on a corrupt row (never bridging `,` directly to `x`)
            s2_lines = [f"p001,2,a,b,0.080,0.140,0.220,0.060,0.140,\n" for _ in range(18)] + ["p001,2,b,,0.080,,,,,\n"]
            s1_padded = lines[1:] + [f"p001,1,m,n,0.080,0.140,0.220,0.060,0.140,\n" for _ in range(16)] + ["p001,1,n,,0.080,,,,,\n"]
            p2 = Path(tmp) / "synthetic_free_2s.csv"
            p2.write_text(header + "".join(s1_padded + s2_lines), encoding="utf-8")
            _, filtered_runs = parse_keyrecs_sentence_corpus(
                p2,
                bounds={"H": (0.01, 0.50), "DD": (0.01, 1.50), "UD": (-0.40, 1.40)},
                window_keys=2,
                stride_keys=2,
            )
            s1_run_texts = ["".join(k for k, _, _ in run) for run in filtered_runs[("p001", 1)]]
            self.assertEqual(s1_run_texts[0], 'a",')
            self.assertEqual(s1_run_texts[1], "xy")

    def test_non_overlapping_25_key_sentence_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, meta = prepare_full_pipeline(
                REPO_ROOT / "data" / "raw" / "DSL-StrongPasswordData.csv",
                Path(tmp),
                seed=SEED,
            )
        sent_splits = meta["_sentence_splits"]
        self.assertEqual(meta["keyrecs_sentence_corpus"]["sentence_window_keys"], 25)
        self.assertEqual(meta["keyrecs_sentence_corpus"]["sentence_stride_keys"], 25)
        for sp_name, sp in sent_splits.items():
            for uid in range(len(sp.subjects)):
                for sess in (1, 2):
                    idx = np.where((sp.subject_ids == uid) & (sp.sessions == sess))[0]
                    if len(idx) < 2:
                        continue
                    order = idx[np.argsort(sp.chunk_ids[idx])]
                    kr = sp.key_idx_ranges[order]
                    # Every chunk has >= 25 distinct CSV keystroke rows and strictly disjoint row intervals
                    for k in range(len(kr)):
                        self.assertGreaterEqual(kr[k, 1] - kr[k, 0], 24)
                    for k in range(len(kr) - 1):
                        self.assertLess(
                            kr[k, 1],
                            kr[k + 1, 0],
                            msg=f"Keystroke overlap in {sp_name} uid={uid} sess={sess}: {kr[k]} vs {kr[k+1]}",
                        )


class TestRawHDDPreservationAndAugmentationOrder(unittest.TestCase):
    def test_boundary_rollover_preserved_before_independent_clipping(self) -> None:
        bounds = {"H": (0.02, 0.30), "DD": (0.02, 0.80), "UD": (-0.25, 0.70)}
        # raw_h0 = 0.50 (> H_hi=0.30), raw_dd = 0.32 (< DD_hi=0.80)
        # True raw_ud = 0.32 - 0.50 = -0.18 (rollover!), whereas clipped dd - clipped h = 0.32 - 0.30 = +0.02!
        raw_holds = np.array([[0.50, 0.09, 0.08]], dtype=np.float32)
        raw_dds = np.array([[0.32, 0.20]], dtype=np.float32)
        kb = np.zeros((1, 7, 2), dtype=np.float32)
        mean = np.zeros(N_CHANNELS, dtype=np.float32)
        std = np.ones(N_CHANNELS, dtype=np.float32)

        # 1) warp=False, touch_prob=0.0
        rng = np.random.RandomState(123)
        seq_id = augment_keystroke_batch(
            raw_holds, raw_dds, None, bounds, mean, std, rng, key_block=kb, warp=False, touch_prob=0.0, touch_hold=0.09
        )
        self.assertAlmostEqual(float(seq_id[0, 0, 0]), 0.30, places=6)  # H_curr clipped to 0.30
        self.assertAlmostEqual(float(seq_id[0, 2, 0]), 0.32, places=6)  # DD = 0.32
        self.assertAlmostEqual(float(seq_id[0, 3, 0]), -0.18, places=6)  # UD computed BEFORE H clipping!
        self.assertAlmostEqual(float(seq_id[0, 7, 0]), 0.18, places=6)  # Rollover_Mag = max(0, -UD) = 0.18
        self.assertAlmostEqual(float(seq_id[0, 16, 0]), 1.0, places=6)  # Hold_Valid = 1.0

        # 2) warp=True, touch_prob=0.0 -> matches exact step-by-step order on identical RandomState seed
        rng1 = np.random.RandomState(999)
        seq_warp = augment_keystroke_batch(
            raw_holds, raw_dds, None, bounds, mean, std, rng1, key_block=kb, warp=True, touch_prob=0.0, touch_hold=0.09
        )
        rng2 = np.random.RandomState(999)
        tempo = rng2.uniform(0.80, 1.25, size=(1, 1)).astype(np.float32)
        phase = rng2.uniform(0, 2 * np.pi, size=(1, 1)).astype(np.float32)
        t_grid = np.linspace(0.0, np.pi, 2, dtype=np.float32)[None, :]
        smooth_wave = 1.0 + 0.06 * np.sin(t_grid + phase)
        local_h = rng2.normal(1.0, 0.035, size=raw_holds.shape).astype(np.float32)
        local_d = rng2.normal(1.0, 0.040, size=raw_dds.shape).astype(np.float32)
        jitter_h = rng2.normal(0.0, 0.0018, size=raw_holds.shape).astype(np.float32)
        jitter_dd = rng2.normal(0.0, 0.0020, size=raw_dds.shape).astype(np.float32)
        ah = (raw_holds * (0.65 + 0.35 * tempo) * local_h + jitter_h).astype(np.float32)
        ad = (raw_dds * tempo * smooth_wave * local_d + jitter_dd).astype(np.float32)
        expected_ud0 = float(np.clip(ad[0, 0] - ah[0, 0], *bounds["UD"]))
        self.assertAlmostEqual(float(seq_warp[0, 3, 0]), expected_ud0, places=6)

        # 3) touch_prob=1.0 -> matches touchify_timings with Hold_Valid = 0.0
        touch_hold = 0.095
        seq_touch = augment_keystroke_batch(
            raw_holds, raw_dds, None, bounds, mean, std, np.random.RandomState(7), key_block=kb, warp=False, touch_prob=1.0, touch_hold=touch_hold
        )
        h_t, u_t = touchify_timings(raw_holds, raw_dds, touch_hold, bounds)
        self.assertTrue(np.allclose(seq_touch[0, 0, :], h_t[0, :-1], atol=1e-6))
        self.assertTrue(np.allclose(seq_touch[0, 3, :], u_t[0, :], atol=1e-6))
        self.assertTrue(np.allclose(seq_touch[0, 16, :], 0.0, atol=1e-6))

    def test_splits_preserve_unclipped_raw_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            splits, meta = prepare_full_pipeline(
                REPO_ROOT / "data" / "raw" / "DSL-StrongPasswordData.csv",
                Path(tmp),
                seed=SEED,
            )
        self.assertGreater(float(splits["train"].raw_dds.max()), float(meta["outlier_bounds"]["DD"][1]))
        kr_fixed = meta["_keyrecs_fixed_train"]
        raw_h, raw_d, raw_u, seq_scaled, _sids = kr_fixed
        self.assertEqual(raw_h.shape[0], seq_scaled.shape[0])
        self.assertEqual(raw_d.shape, raw_u.shape)
        self.assertGreater(float(raw_d.max()), float(meta["outlier_bounds"]["DD"][1]))


class TestStitchAndTensorContract(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = {"H": (0.01, 0.50), "DD": (0.01, 1.50), "UD": (-0.40, 1.40)}
        self.mean = np.zeros(N_CHANNELS, dtype=np.float32)
        self.std = np.ones(N_CHANNELS, dtype=np.float32)
        self.impute_hold = 0.09

    def test_hand_calculated_backspace_stitch_ab_break_cd(self) -> None:
        # Type A (h=0.08), B (h=0.06), typo X, Backspace, C (h=0.20), D (h=0.09)
        # RecorderSim pops X on Backspace and marks C.clean_from_prev=False, so B->C is skipped,
        # leaving A->B and C->D with stitch_flags == [False, True]!
        raw_seq = [
            ("a", 0.00, 0.08),
            ("b", 0.20, 0.06),
            ("x", 0.40, 0.07),
            ("Backspace", 0.60, 0.05),
            ("c", 0.85, 0.20),
            ("d", 1.15, 0.09),
        ]
        rec = RecorderSim(touch=False)
        for label, t, h in raw_seq:
            kd, _ = kind_of(label)
            code = "Backspace" if kd == "bs" else code_for_label(label)
            ch = label if kd == "char" else ""
            rec.feed(code, ch, t, h)
        stroke = rec.extract()
        self.assertEqual(stroke["pair_keys"], [["KeyA", "KeyB"], ["KeyC", "KeyD"]])
        self.assertEqual(stroke["holds"], [0.08, 0.06, 0.20, 0.09])
        self.assertEqual(stroke["h_curr"], [0.08, 0.20])
        self.assertEqual(stroke["h_next"], [0.06, 0.09])
        self.assertEqual(stroke["stitch_flags"], [False, True])

        T = build_input_tensor(stroke, self.bounds, self.impute_hold, self.mean, self.std)
        self.assertEqual(T.shape, (17, 2))
        # H_curr must be [0.08, 0.20] (NOT [0.08, 0.06]!)
        self.assertTrue(np.allclose(T[0], [0.08, 0.20], atol=1e-6))
        self.assertTrue(np.allclose(T[1], [0.06, 0.09], atol=1e-6))
        # mean_h = (0.08 + 0.06 + 0.20 + 0.09) / 4 = 0.1075 -> Rel_H_curr[1] = 0.20 / 0.1075
        expected_mean_h = (0.08 + 0.06 + 0.20 + 0.09) / 4.0
        self.assertAlmostEqual(float(T[4, 1]), 0.20 / expected_mean_h, places=5)
        # Local_Accel must be 0.0 at t=0 AND 0.0 at stitched t=1!
        self.assertTrue(np.allclose(T[8], [0.0, 0.0], atol=1e-6))

        # Contrast with naive unrepaired helper used for Case 4 ablation
        naive = extract_naive_unrepaired_stroke(raw_seq, touch=False)
        self.assertGreater(len(naive["dds"]), len(stroke["dds"]))

    def test_skipped_max_hold_outlier_sets_stitch_flag_and_zeros_accel(self) -> None:
        # Stack: A(h=0.08), B(h=0.06), X(h=1.50 > MAX_HOLD=1.20), C(h=0.20), D(h=0.09)
        # Transitions B->X and X->C are skipped by MAX_HOLD filter; C->D must be marked stitched!
        rec = RecorderSim(touch=False)
        for code, ch, t, h in (
            ("KeyA", "a", 0.00, 0.08),
            ("KeyB", "b", 0.18, 0.06),
            ("KeyX", "x", 0.35, 1.50),
            ("KeyC", "c", 0.60, 0.20),
            ("KeyD", "d", 0.85, 0.09),
        ):
            rec.feed(code, ch, t, h)
        stroke = rec.extract()
        self.assertEqual(stroke["pair_keys"], [["KeyA", "KeyB"], ["KeyC", "KeyD"]])
        self.assertEqual(stroke["stitch_flags"], [False, True])
        self.assertEqual(stroke["h_curr"], [0.08, 0.20])
        self.assertEqual(stroke["h_next"], [0.06, 0.09])
        T = build_input_tensor(stroke, self.bounds, self.impute_hold, self.mean, self.std)
        self.assertAlmostEqual(float(T[8, 1]), 0.0, places=6)

    def test_mixed_none_holds_and_empty_stroke(self) -> None:
        # Mixed keyboard/touch: A(h=0.10) -> B(h=None) -> C(h=0.20) -> D(h=0.14)
        sim = RecorderSim(touch=False)
        sim.feed("KeyA", "a", 0.00, 0.10)
        sim.feed("KeyB", "b", 0.25, None)
        sim.feed("KeyC", "c", 0.50, 0.20)
        sim.feed("KeyD", "d", 0.90, 0.14)
        stroke = sim.extract()

        self.assertEqual(stroke["hv_curr"], [True, False, True])
        self.assertEqual(stroke["hv_next"], [False, True, True])
        T = build_input_tensor(stroke, self.bounds, self.impute_hold, self.mean, self.std)
        # Hold_Valid = 1.0 iff BOTH hv_curr[t] and hv_next[t] are True -> [0.0, 0.0, 1.0]
        self.assertTrue(np.allclose(T[16], [0.0, 0.0, 1.0], atol=1e-6))
        # Model mean_h uses imputed+clipped unique_holds = [0.10, 0.09, 0.20, 0.14]
        expected_mean_h = (0.10 + self.impute_hold + 0.20 + 0.14) / 4.0
        self.assertAlmostEqual(float(T[4, 2]), 0.20 / expected_mean_h, places=5)

        # capture_stats holdMs uses ONLY observed raw unique_holds [0.10, 0.20, 0.14] -> round(146.67) == 147ms
        st = capture_stats(stroke)
        self.assertEqual(st["holdMs"], round(1000.0 * (0.10 + 0.20 + 0.14) / 3.0))

        # Pure touch (all holds None) -> capture_stats holdMs == 0
        sim_touch = RecorderSim(touch=True)
        for ch, t in (("a", 0.00), ("b", 0.25), ("c", 0.50), ("d", 0.90)):
            sim_touch.feed(None, ch, t, None)
        stroke_touch = sim_touch.extract()
        self.assertEqual(capture_stats(stroke_touch)["holdMs"], 0)

        # Payload keys check
        js_s = _jsonable_stroke(stroke)
        for k in ("h_curr", "h_next", "hv_curr", "hv_next"):
            self.assertIn(k, js_s)
        sp = stroke_payload(fixed_stroke_capture([0.08, 0.09], [0.15], [0.07], ["KeyA", "KeyB"]))
        for k in ("hCurr", "hNext", "hvCurr", "hvNext"):
            self.assertIn(k, sp)

        # Empty stroke (m == 0) -> (17, 0) float32
        empty_stroke = extract_clean_stroke([Stroke("KeyA", 0.0, 0.08, True)], [Event("char", 0.0, 0.08)], 0)
        T_empty = build_input_tensor(empty_stroke, self.bounds, self.impute_hold, self.mean, self.std)
        self.assertEqual(T_empty.shape, (17, 0))
        self.assertEqual(T_empty.dtype, np.float32)


class TestDisjointSynthesisAndCapitalThresholdParity(unittest.TestCase):
    def test_disjoint_occ_id_enforced_and_capital_threshold_parity(self) -> None:
        raw_sessions = load_keyrecs_sessions(REPO_ROOT / "data" / "raw" / "KeyRecs-free-text.csv")
        subjs = sorted({p for (p, s) in raw_sessions if s == 1})[:20]
        tabs = digraph_tables(raw_sessions, subjs)

        # Every occurrence across all (person, session, digraph) has a unique integer occ_id at index 4
        all_occ_ids = [occ[4] for tab in tabs.values() for lst in tab.values() for occ in lst]
        self.assertEqual(len(all_occ_ids), len(set(all_occ_ids)))

        val_p, test_p = subjs[:10], subjs[10:20]
        rng = np.random.default_rng(SEED)
        chosen_val = choose_strings(raw_sessions, tabs, val_p, n_words=3, n_random=2, rng=rng, vocab_people=val_p)

        bounds = {"H": (0.01, 0.45), "DD": (0.01, 1.50), "UD": (-0.35, 1.40)}
        mean = np.zeros(N_CHANNELS, dtype=np.float32)
        std = np.ones(N_CHANNELS, dtype=np.float32)

        # Deterministic normalized mock encoders & mock cohorts
        rng_proj = np.random.default_rng(42)
        W9 = rng_proj.standard_normal((9, 64))
        W17 = rng_proj.standard_normal((17, 64))
        coh_mock = rng_proj.standard_normal((16, 64))
        coh_mock /= np.linalg.norm(coh_mock, axis=1, keepdims=True)
        cohorts9 = {"kb": coh_mock, "touch": coh_mock}

        def _enc(W):
            def fn(tensors):
                out = []
                for x in tensors:
                    v = x[: W.shape[0]].mean(axis=1) @ W
                    out.append(v / max(1e-8, np.linalg.norm(v)))
                return np.asarray(out, dtype=np.float64)
            return fn

        rows = string_trials(
            raw_sessions,
            tabs,
            val_p,
            chosen_val,
            _enc(W9),
            _enc(W17),
            bounds,
            mean,
            std,
            0.09,
            cohorts9,
            np.random.default_rng(SEED),
        )
        same_rows = [r for r in rows if r["tag"] == "same"]
        self.assertGreater(len(same_rows), 0)
        for r in same_rows:
            self.assertTrue(set(r["enroll_occ_ids"]).isdisjoint(set(r["query_occ_ids"])))

        synth_meta = synthesis_metadata(tabs, val_p, chosen_val, rows)
        self.assertTrue(synth_meta["disjoint_occ_enforced"])
        self.assertNotIn("unique_syntheses", json.dumps(synth_meta))
        self.assertIn("generated_attempts", synth_meta)

        # Verify capital_trials internal disjointness assertion & capital_fusion deployed threshold parity + val-only isolation
        rates = capital_method_rates(raw_sessions)
        prm = {"mu9": 0.25, "sd9": 0.30, "mu17": 0.30, "sd17": 0.30, "tau": 2.3456}
        cap_val = capital_trials(
            tabs, rates, val_p, 8, 3, _enc(W9), _enc(W17), bounds, mean, std, 0.09, prm, np.random.default_rng(SEED)
        )
        cap_test = capital_trials(
            tabs, rates, test_p, 8, 3, _enc(W9), _enc(W17), bounds, mean, std, 0.09, prm, np.random.default_rng(SEED + 1)
        )
        self.assertGreater(len(cap_val), 0)
        self.assertGreater(len(cap_test), 0)

        base_tau = 2.3456
        res1 = capital_fusion(cap_val, cap_test, caps_share=0.25, base_tau=base_tau)
        self.assertEqual(res1["deployed_threshold_no_capitals"], round(base_tau, 4))
        self.assertEqual(res1["deployed_threshold_with_capitals"], round(base_tau + res1["tau_offset"], 4))

        # Mutate test_rows drastically -> selected weight, tau_offset, and deployed thresholds MUST NOT change!
        cap_test_mutated = [(y, -10.0 if y == 1 else 10.0, qc, nc) for (y, _z, qc, nc) in cap_test]
        res2 = capital_fusion(cap_val, cap_test_mutated, caps_share=0.25, base_tau=base_tau)
        self.assertEqual(res1["weight"], res2["weight"])
        self.assertEqual(res1["tau_offset"], res2["tau_offset"])
        self.assertEqual(res1["deployed_threshold_with_capitals"], res2["deployed_threshold_with_capitals"])


class TestCanonicalDefaultsAndCleanDirectoryPipeline(unittest.TestCase):
    def test_canonical_epoch_defaults_and_non_retraining_notebook_spec(self) -> None:
        self.assertEqual(VANILLA_DEFAULT_EPOCHS, 35)
        self.assertEqual(KEYPRINT_DEFAULT_EPOCHS, 35)
        old_smoke = os.environ.pop("KEYPRINT_SMOKE_EPOCHS", None)
        try:
            import visualize_and_run
            importlib.reload(visualize_and_run)
            self.assertEqual(visualize_and_run.VANILLA_EPOCHS, 35)
            self.assertEqual(visualize_and_run.KEYPRINT_EPOCHS, 35)
        finally:
            if old_smoke is not None:
                os.environ["KEYPRINT_SMOKE_EPOCHS"] = old_smoke

    def test_clean_empty_directory_from_scratch_pipeline_and_notebook_execution(self) -> None:
        if SCRATCH_VERIFY_DIR.exists():
            shutil.rmtree(SCRATCH_VERIFY_DIR)
        SCRATCH_VERIFY_DIR.mkdir(parents=True, exist_ok=True)
        # Guarantee ZERO pre-existing .pt or .onnx files before running
        self.assertEqual(list(SCRATCH_VERIFY_DIR.rglob("*.pt")), [])
        self.assertEqual(list(SCRATCH_VERIFY_DIR.rglob("*.onnx")), [])

        env = {
            **os.environ,
            "KEYPRINT_OUTPUT_ROOT": str(SCRATCH_VERIFY_DIR),
            "KEYPRINT_REPO_ROOT": str(REPO_ROOT),
            "KEYPRINT_SMOKE_EPOCHS": "1",
        }
        env.pop("KEYPRINT_REUSE_WEIGHTS", None)
        py = sys.executable

        # 1) Full from-scratch training & export (`visualize_and_run.py`)
        subprocess.run([py, str(REPO_ROOT / "src" / "visualize_and_run.py")], env=env, check=True)

        # Verify all 4 checkpoints + 4 ONNX files + JSON artifacts exist
        for rel in (
            "models/vanilla_1dcnn.pt",
            "models/vanilla_1dcnn_aug.pt",
            "models/legacy/keyprint_net_9ch.pt",
            "models/keyprint_net.pt",
            "models/keyprint_net.onnx",
            "models/legacy/keyprint_net_9ch.onnx",
            "web/public/models/keyprint_net.onnx",
            "web/public/models/keyprint_net_password.onnx",
            "web/public/models/model_config.json",
            "web/scripts/fixtures/feature_parity.json",
            "models/evaluation_report.json",
        ):
            self.assertTrue((SCRATCH_VERIFY_DIR / rel).exists(), msg=f"Missing {rel}")

        report = json.loads((SCRATCH_VERIFY_DIR / "models" / "evaluation_report.json").read_text(encoding="utf-8"))
        web_cfg = json.loads((SCRATCH_VERIFY_DIR / "web" / "public" / "models" / "model_config.json").read_text(encoding="utf-8"))
        parity_fx = json.loads((SCRATCH_VERIFY_DIR / "web" / "scripts" / "fixtures" / "feature_parity.json").read_text(encoding="utf-8"))

        # Schema v4 & provenance checks
        self.assertEqual(web_cfg["schema_version"], 4)
        self.assertEqual(web_cfg["preprocess_version"], "kp-preproc-v4")
        sha17 = hashlib.sha256((SCRATCH_VERIFY_DIR / "web/public/models/keyprint_net.onnx").read_bytes()).hexdigest()
        sha9 = hashlib.sha256((SCRATCH_VERIFY_DIR / "web/public/models/keyprint_net_password.onnx").read_bytes()).hexdigest()
        expected_hash = hashlib.sha256(f"{sha17}:{sha9}".encode("utf-8")).hexdigest()[:16]
        self.assertEqual(web_cfg["onnx_sha256"], {"sentence_17ch": sha17, "password_9ch": sha9})
        self.assertEqual(web_cfg["model_hash"], expected_hash)
        for sp in web_cfg["mvp_sample_profiles"]:
            self.assertEqual(sp["schemaVersion"], 4)
            self.assertEqual(sp["preprocessVersion"], "kp-preproc-v4")
            self.assertEqual(sp["modelHash"], expected_hash)
            self.assertEqual(sp["createdAt"], 1700000000000)
            self.assertEqual(sp["updatedAt"], 1700000000000)

        for case in parity_fx["cases"]:
            for k in ("h_curr", "h_next", "hv_curr", "hv_next"):
                self.assertIn(k, case["expected"]["stroke"])

        # Controlled baselines & human(1600) vs bot(200) breakdown
        for m_name in (
            "ScaledManhattan",
            "Vanilla1DCNN",
            "Vanilla1DCNN_Aug",
            "Vanilla1DCNN_Aug_ASNorm",
            "KeyPrintNet_9ch",
            "KeyPrintNet",
        ):
            self.assertIn(m_name, report["models"])
            ib = report["models"][m_name]["impostor_breakdown"]
            self.assertEqual(ib["human_only"]["num_impostor"], 1600)
            self.assertEqual(ib["bot_only"]["num_impostor"], 200)
            self.assertEqual(ib["combined"]["num_impostor"], 1800)

        # 5 separated silhouettes + histories + troubleshooting ablations + provenance
        for sil_k in (
            "silhouette_raw4_cosine",
            "silhouette_clipped4_unscaled_cosine",
            "silhouette_clipped4_scaled_cosine",
            "silhouette_bio9_cosine",
            "silhouette_bio17_cosine",
        ):
            self.assertIn(sil_k, report["preprocessing"])
        for h_k in ("Vanilla1DCNN", "Vanilla1DCNN_Aug", "KeyPrintNet_9ch", "KeyPrintNet"):
            self.assertEqual(len(report["histories"][h_k]), 1)
        self.assertIn("tempo_normalization", report["troubleshooting_ablations"])
        self.assertIn("backspace_stitch_subset", report["troubleshooting_ablations"])
        self.assertIn("provenance", report)
        self.assertIn("password_transfer_val", report["feature_upgrade"])
        self.assertIn("password_transfer_other_content", report["feature_upgrade"])

        # 2) Auxiliary evaluations (`eval_password_calibration.py` & `eval_typing_style.py`)
        subprocess.run([py, str(REPO_ROOT / "src" / "eval_password_calibration.py")], env=env, check=True)
        subprocess.run([py, str(REPO_ROOT / "src" / "eval_typing_style.py")], env=env, check=True)

        pw_cal = json.loads((SCRATCH_VERIFY_DIR / "models" / "password_calibration.json").read_text(encoding="utf-8"))
        self.assertFalse(pw_cal["split_isolation"]["val_test_pooled"])
        style_rep = json.loads((SCRATCH_VERIFY_DIR / "models" / "typing_style_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(style_rep["participant_counts"]["raw_corpus_participants"], 99)
        self.assertEqual(style_rep["participant_counts"]["clean_chunk_participants"], 96)

        # 3) Generate & execute canonical non-retraining notebook (`build_notebook.py`)
        subprocess.run([py, str(REPO_ROOT / "src" / "build_notebook.py")], env=env, check=True)
        nb_path = SCRATCH_VERIFY_DIR / "notebooks" / "keyprint_training_pipeline.ipynb"
        nb_text = nb_path.read_text(encoding="utf-8")
        self.assertNotIn("train_vanilla_baseline", nb_text)
        self.assertNotIn("train_keyprint_net", nb_text)

        nb = nbformat.read(nb_path, as_version=4)
        code_cells = [c["source"] for c in nb.cells if c.cell_type == "code"]
        runner = (
            "import json, matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import builtins\n"
            "builtins.display = lambda *a, **k: None\n"
            "ns = {}\n"
            f"cells = {json.dumps(code_cells)}\n"
            "for idx, src in enumerate(cells):\n"
            "    exec(compile(src, f'<nb_cell_{idx}>', 'exec'), ns)\n"
            "print('NOTEBOOK_EXEC_OK')\n"
        )
        res = subprocess.run(
            [py, "-c", runner],
            cwd=str(nb_path.parent),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("NOTEBOOK_EXEC_OK", res.stdout)

        # 4) Verify `KEYPRINT_REUSE_WEIGHTS=1` preserves histories & exact metrics without retraining
        env_reuse = {**env, "KEYPRINT_REUSE_WEIGHTS": "1"}
        subprocess.run([py, str(REPO_ROOT / "src" / "visualize_and_run.py")], env=env_reuse, check=True)
        report_reused = json.loads((SCRATCH_VERIFY_DIR / "models" / "evaluation_report.json").read_text(encoding="utf-8"))
        for h_k in ("Vanilla1DCNN", "Vanilla1DCNN_Aug", "KeyPrintNet_9ch", "KeyPrintNet"):
            self.assertEqual(len(report_reused["histories"][h_k]), 1)
        self.assertAlmostEqual(
            report["models"]["KeyPrintNet"]["eer"],
            report_reused["models"]["KeyPrintNet"]["eer"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
