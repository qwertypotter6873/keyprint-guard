"""Unit & end-to-end regression tests for string cohort calibration, hold calibration, and field-study JSONL analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze_field_study import analyze_jsonl_files, format_korean_summary  # noqa: E402
from string_calibration import (  # noqa: E402
    PUBLIC_EXPERIMENT_STRINGS,
    STRING_FEATURE_NAMES,
    apply_hold_calibration,
    build_calibration_parity_cases,
    extract_string_features,
    predict_string_dual_params,
)


class CalibrationAndFieldStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_config = json.loads(
            (REPO_ROOT / "web" / "public" / "models" / "model_config.json").read_text(encoding="utf-8")
        )
        cls.calib_bundle = json.loads(
            (REPO_ROOT / "web" / "public" / "models" / "string_cohort_calibration.json").read_text(encoding="utf-8")
        )
        cls.parity_fixture = json.loads(
            (REPO_ROOT / "web" / "scripts" / "fixtures" / "calibration_parity.json").read_text(encoding="utf-8")
        )

    def test_public_experiment_strings_stratification(self) -> None:
        self.assertEqual(len(PUBLIC_EXPERIMENT_STRINGS), 24)
        for length in (12, 16, 20):
            sub = [s for s in PUBLIC_EXPERIMENT_STRINGS if s["length"] == length]
            self.assertEqual(len(sub), 8)
            self.assertEqual(sum(1 for s in sub if s["tier"] == "core"), 2)
            for s in sub:
                self.assertEqual(len(str(s["text"])), length)

        rep_fx = extract_string_features("BookkeeperRoom77", self.calib_bundle["digraph_table"])
        alt_fx = extract_string_features("formalsignalwork", self.calib_bundle["digraph_table"])
        self.assertGreater(rep_fx["features"][0], 0.15)  # repeat_frac
        self.assertEqual(alt_fx["features"][0], 0.0)
        self.assertGreater(alt_fx["features"][2], rep_fx["features"][2])  # alt_hand_frac

    def test_parity_fixture_matches_live_python(self) -> None:
        live = build_calibration_parity_cases(self.model_config, self.calib_bundle)
        self.assertEqual(live, self.parity_fixture)

    def test_shrinkage_falls_back_to_length_baseline_on_unseen_transitions(self) -> None:
        by_len = self.model_config["password_dual"]["by_length"]
        empty_bundle = {
            **self.calib_bundle,
            "digraph_table": {},
            "tau_offset": {"keyboard": 0.0, "touch": 0.0},
        }
        prm = predict_string_dual_params("CoffeeTree88", "touch", by_len, empty_bundle)
        self.assertEqual(prm["support"], 0.0)
        self.assertEqual(prm["shrink_weight"], 0.0)

        supported = predict_string_dual_params("CoffeeTree88", "touch", by_len, self.calib_bundle)
        self.assertGreater(supported["support"], 0.5)
        self.assertGreater(supported["shrink_weight"], 0.5)
        # Repeat-heavy touch passwords have higher impostor baseline mu17 and tighter sd17
        self.assertGreater(supported["mu17"], prm["mu17"])
        self.assertLess(supported["sd17"], prm["sd17"])

    def test_physics_consistent_hold_calibration_preserves_ud_and_touch_nulls(self) -> None:
        stroke = {
            "holds": [0.120, None, 0.100],
            "holdValid": [True, False, True],
            "hCurr": [0.120, None],
            "hNext": [None, 0.100],
            "hvCurr": [True, False],
            "hvNext": [False, True],
            "dds": [0.200, 0.230],
            "uds": [0.080, None],
            "stitchFlags": [False, False],
            "pairKeys": [["KeyA", "KeyB"], ["KeyB", "KeyC"]],
        }
        cal = apply_hold_calibration(stroke, delta_sec=0.030, recompute_ud=True)
        self.assertAlmostEqual(cal["holds"][0], 0.090, places=7)
        self.assertIsNone(cal["holds"][1])
        self.assertAlmostEqual(cal["holds"][2], 0.070, places=7)
        # UD[0] = DD[0] (0.200) - new_hCurr[0] (0.090) = 0.110
        self.assertAlmostEqual(cal["uds"][0], 0.110, places=7)
        self.assertIsNone(cal["uds"][1])

        naive = apply_hold_calibration(stroke, delta_sec=0.030, recompute_ud=False)
        self.assertAlmostEqual(naive["uds"][0], 0.080, places=7)  # broken UD != DD - H_curr

    def test_analyze_field_study_end_to_end_multi_device_merge_and_rescore(self) -> None:
        # Build realistic multi-device JSONL files (Device A: mechanical_linear, Device B: pantograph_external, Device C: touch_phone)
        unit0 = [1.0] + [0.0] * 63
        unit1 = [0.0, 1.0] + [0.0] * 62

        raw_stroke_mech = {
            "cpm": 320,
            "holdMs": 82,
            "rolloverPct": 22,
            "backspaceCount": 0,
            "cleanDigraphCount": 8,
            "inputMode": "keyboard",
            "capitalMethods": ["shift"],
            "behavior": {"bs_rate": [0.0, 9]},
            "holds": [0.082] * 9,
            "holdValid": [True] * 9,
            "hCurr": [0.082] * 8,
            "hNext": [0.082] * 8,
            "hvCurr": [True] * 8,
            "hvNext": [True] * 8,
            "dds": [0.175, 0.160, 0.190, 0.155, 0.180, 0.170, 0.165, 0.185],
            "uds": [d - 0.082 for d in [0.175, 0.160, 0.190, 0.155, 0.180, 0.170, 0.165, 0.185]],
            "stitchFlags": [False] * 8,
            "pairKeys": [
                ["KeyG", "KeyA"], ["KeyA", "KeyC"], ["KeyC", "KeyH"], ["KeyH", "KeyO"],
                ["KeyO", "KeyN"], ["KeyN", "Digit2"], ["Digit2", "Digit0"], ["Digit0", "Digit2"],
            ],
            "pairKeysRedacted": False,
        }
        # Pantograph shifted by +28ms hold on identical DD rhythm
        raw_stroke_panto = {
            **raw_stroke_mech,
            "holdMs": 110,
            "holds": [0.110] * 9,
            "hCurr": [0.110] * 8,
            "hNext": [0.110] * 8,
            "uds": [d - 0.110 for d in raw_stroke_mech["dds"]],
        }

        enr_mech = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_enr_mech_1",
            "eventType": "enroll",
            "recordedAt": "2026-11-19T01:00:00Z",
            "domain": "arena",
            "lang": "en",
            "roomCode": "STUDY1",
            "profileId": "u_p01",
            "profileName": "참가자1",
            "shotCount": 5,
            "env": {
                "participantId": "P01",
                "os": "macOS",
                "browser": "Chrome",
                "timerResolutionUs": 5,
                "keyboardType": "mechanical_linear",
                "hangulLayout": "pc_2beolsik",
                "layoutMismatch2Beolsik": False,
                "posture": "desk_keyboard",
                "sessionSlot": "day1_slot1",
                "detectedInputMode": "keyboard",
            },
            "templateSummary": {
                "count": 5,
                "holdMs": 82,
                "inputMode": "keyboard",
                "centroid": unit0,
                "exemplars": [unit0],
                "behavior": {"bs_rate": [0.02, 50]},
            },
            "rawCaptures": [raw_stroke_mech],
        }
        enr_panto = {
            **enr_mech,
            "recordId": "ev_enr_panto_2",
            "recordedAt": "2026-11-19T01:05:00Z",
            "profileId": "u_p02",
            "profileName": "참가자2",
            "env": {**enr_mech["env"], "keyboardType": "pantograph_external", "sessionSlot": "day1_slot2"},
            "templateSummary": {
                "count": 5,
                "holdMs": 110,
                "inputMode": "keyboard",
                "centroid": unit1,
                "exemplars": [unit1],
                "behavior": {"bs_rate": [0.08, 50]},
            },
            "rawCaptures": [raw_stroke_panto],
        }
        q_cross = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_q_cross_1",
            "eventType": "query",
            "domain": "arena",
            "lang": "en",
            "roomCode": "STUDY1",
            "recordedAt": "2026-11-19T01:10:00Z",
            "queryEnv": {
                **enr_mech["env"],
                "keyboardType": "pantograph_external",
                "sessionSlot": "day1_slot2",
            },
            "groundTruth": {"role": "enrolled", "profileId": "u_p01", "profileName": "참가자1", "enrollEnv": None},
            "crossDevice": True,
            "candidateCount": 2,
            "verdict": {
                "winnerId": "u_p01",
                "top1Score": 0.61,
                "leadGap": 0.14,
                "openSetThreshold": 0.42,
                "marginThreshold": 0.03,
                "accepted": True,
                "rejectedByMargin": False,
                "top1Correct": True,
                "decisionCorrect": True,
            },
            "rankings": [
                {"id": "u_p01", "name": "참가자1", "rank": 1, "similarity": 0.61, "embeddingSimilarity": 0.61, "behaviorLlr": 0.4},
                {"id": "u_p02", "name": "참가자2", "rank": 2, "similarity": 0.47, "embeddingSimilarity": 0.47, "behaviorLlr": -0.3},
            ],
            "optInDiagnostics": {"rawCaptures": [raw_stroke_panto]},
        }

        # Touch enrollment + touch query with DeviceMotion + private custom password record
        enr_touch = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_enr_touch_1",
            "eventType": "enroll",
            "recordedAt": "2026-11-19T02:00:00Z",
            "domain": "arena",
            "lang": "ko",
            "roomCode": "STUDY1",
            "profileId": "u_p01",
            "profileName": "참가자1",
            "shotCount": 5,
            "env": {
                "participantId": "P01",
                "os": "iOS",
                "browser": "Safari",
                "timerResolutionUs": 100,
                "keyboardType": "touch_phone",
                "hangulLayout": "mobile_qwerty_2beolsik",
                "layoutMismatch2Beolsik": False,
                "posture": "hand_two_thumbs",
                "sessionSlot": "day1_slot3",
                "detectedInputMode": "touch",
            },
            "templateSummary": {"count": 5, "holdMs": 0, "inputMode": "touch", "centroid": unit0, "exemplars": [unit0]},
            "deviceMotion": {
                "available": True,
                "effectiveHz": 60.0,
                "medianIntervalMs": 16.67,
                "tapImpulseMean": 0.62,
                "linearAccelRms": 0.31,
                "gyroRms": 5.4,
                "pitchDeg": 34.0,
                "inferredPosture": "in_hand",
            },
        }
        q_touch = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_q_touch_1",
            "eventType": "query",
            "domain": "arena",
            "lang": "ko",
            "roomCode": "STUDY1",
            "recordedAt": "2026-11-19T02:05:00Z",
            "queryEnv": enr_touch["env"],
            "groundTruth": {"role": "enrolled", "profileId": "u_p01", "profileName": "참가자1"},
            "verdict": {
                "winnerId": "u_p01",
                "top1Score": 0.52,
                "leadGap": 0.08,
                "openSetThreshold": 0.45,
                "marginThreshold": 0.03,
                "accepted": True,
                "rejectedByMargin": False,
                "top1Correct": True,
                "decisionCorrect": True,
            },
            "rankings": [
                {"id": "u_p01", "name": "참가자1", "rank": 1, "similarity": 0.52, "embeddingSimilarity": 0.43, "behaviorLlr": 1.8},
                {"id": "u_p02", "name": "참가자2", "rank": 2, "similarity": 0.44, "embeddingSimilarity": 0.44, "behaviorLlr": 0.0},
            ],
            "deviceMotion": {
                "available": True,
                "effectiveHz": 59.8,
                "medianIntervalMs": 16.72,
                "tapImpulseMean": 0.65,
                "linearAccelRms": 0.33,
                "gyroRms": 5.1,
                "pitchDeg": 33.2,
                "inferredPosture": "in_hand",
            },
        }
        q_pw_private_gen = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_q_pw_1",
            "eventType": "query",
            "domain": "password",
            "roomCode": "STUDY1",
            "recordedAt": "2026-11-19T02:10:00Z",
            "queryEnv": enr_touch["env"],
            "targetAccount": {"id": "u_p01", "name": "참가자1"},
            "groundTruth": {"role": "genuine", "isGenuine": True},
            "passwordMeta": {
                "length": 12,
                "publicExperimentId": None,
                "publicText": None,
                "passwordHash12": "a1b2c3d4e5f6",
                "isPublicAllowlist": False,
            },
            "verdict": {
                "accepted": True,
                "adaptedAfterVerdict": False,
                "z": 1.45,
                "lengthBaseline": {"z": 1.10, "similarity": 0.48, "accepted": False},
                "stringCalibrated": {"z": 1.45, "similarity": 0.57, "accepted": True, "latencyMs": 0.04},
            },
            "optInDiagnostics": {
                "rawCapture": {
                    **raw_stroke_mech,
                    "pairKeys": None,
                    "pairKeysRedacted": True,
                    "capitalMethods": [],
                }
            },
        }
        q_pw_pub_imp = {
            **q_pw_private_gen,
            "recordId": "ev_q_pw_2",
            "recordedAt": "2026-11-19T02:12:00Z",
            "groundTruth": {"role": "impostor", "isGenuine": False},
            "passwordMeta": {
                "length": 12,
                "publicExperimentId": "EXP12_REP_A",
                "publicCategory": "repeat_heavy",
                "publicText": "CoffeeTree88",
                "passwordHash12": "112233445566",
                "isPublicAllowlist": True,
            },
            "verdict": {
                "accepted": False,
                "adaptedAfterVerdict": False,
                "z": -0.40,
                "lengthBaseline": {"z": 0.35, "similarity": 0.51, "accepted": True},
                "stringCalibrated": {"z": -0.40, "similarity": 0.39, "accepted": False, "latencyMs": 0.03},
            },
            "optInDiagnostics": {"rawCapture": raw_stroke_mech},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            f_pc = Path(tmpdir) / "pc.jsonl"
            f_phone = Path(tmpdir) / "phone.jsonl"
            # Include a duplicate record across files to verify deduplication
            f_pc.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in [enr_mech, enr_panto, q_cross]) + "\n",
                encoding="utf-8",
            )
            f_phone.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in [enr_mech, enr_touch, q_touch, q_pw_private_gen, q_pw_pub_imp]) + "\n",
                encoding="utf-8",
            )

            report = analyze_jsonl_files([f_pc, f_phone], repo_root=REPO_ROOT)

        self.assertEqual(report["meta"]["total_records"], 7)
        self.assertEqual(report["meta"]["duplicates_skipped"], 1)
        integ = report["environment_and_errors"]["protocol_integrity"]
        self.assertTrue(integ["custom_password_redaction_ok"])
        self.assertTrue(integ["template_lock_ok"])

        # Within-subject delta: pantograph (110ms) - mechanical (82ms) = +28.0ms
        deltas = report["hold_calibration"]["within_subject_switch_deltas_ms"]
        self.assertAlmostEqual(deltas["pantograph_external - mechanical_linear"]["mean_delta_ms"], 28.0, places=2)
        self.assertTrue(report["hold_calibration"]["rescoring"]["available"])
        self.assertEqual(
            report["hold_calibration"]["rescoring"]["arena_cross_switch"]["physics_switch_plus_shrink"]["n_cross_switch_queries"],
            1,
        )

        # Touch beta=0 fails to accept (embeddingSimilarity 0.43 < 0.45), whereas beta=0.20 fuses LLR +1.8 and accepts!
        bs = report["touch_and_motion"]["beta_sweep"]
        self.assertEqual(bs["0.00"]["genuine_accept_rate"], 0.0)
        self.assertEqual(bs["0.20"]["genuine_accept_rate"], 1.0)
        self.assertEqual(report["touch_and_motion"]["device_motion_audit"]["posture_inference_agreement"], 1.0)
        self.assertEqual(
            report["touch_and_motion"]["sensor_fusion_ab"]["in_hand"]["keystroke_plus_motion"]["n_sensor_paired_queries"],
            1,
        )

        # Password string calibration beats length baseline on the paired trials
        pw_cmp = report["password_string_calibration"]["overall"]
        self.assertEqual(pw_cmp["string_calibrated"]["genuine_accept_rate"], 1.0)
        self.assertEqual(pw_cmp["string_calibrated"]["impostor_far"], 0.0)
        self.assertEqual(pw_cmp["length_baseline"]["impostor_far"], 1.0)

        summary_txt = format_korean_summary(report)
        self.assertIn("비공개 암호 마스킹=정상", summary_txt)
        self.assertIn("물리 스위치 홀드 편차", summary_txt)

    def test_password_cross_switch_rescore_and_same_participant_gt(self) -> None:
        unit0 = [1.0] + [0.0] * 63
        unit1 = [0.0, 1.0] + [0.0] * 62
        raw_stroke = {
            "cpm": 260,
            "holdMs": 72,
            "rolloverPct": 15,
            "inputMode": "keyboard",
            "holds": [0.072] * 9,
            "holdValid": [True] * 9,
            "hCurr": [0.072] * 8,
            "hNext": [0.072] * 8,
            "hvCurr": [True] * 8,
            "hvNext": [True] * 8,
            "dds": [0.18, 0.21, 0.16, 0.22, 0.19, 0.20, 0.17, 0.23],
            "uds": [0.108, 0.138, 0.088, 0.148, 0.118, 0.128, 0.098, 0.158],
            "stitchFlags": [False] * 8,
            "pairKeys": [
                ["KeyG", "KeyA"], ["KeyA", "KeyC"], ["KeyC", "KeyH"], ["KeyH", "KeyO"],
                ["KeyO", "KeyN"], ["KeyN", "Digit2"], ["Digit2", "Digit0"], ["Digit0", "Digit2"],
            ],
            "pairKeysRedacted": False,
        }
        enr_p01_a = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_pw_enr_a",
            "eventType": "enroll",
            "recordedAt": "2026-11-19T01:00:00Z",
            "domain": "password",
            "roomCode": "STUDY1",
            "profileId": "u_p01_a",
            "profileName": "참가자1_A",
            "env": {"participantId": "P01", "keyboardType": "mechanical_tactile", "detectedInputMode": "keyboard"},
            "templateSummary": {
                "count": 4,
                "holdMs": 72,
                "centroid": unit0,
                "exemplars": [unit0],
                "centroid17": unit0,
                "exemplars17": [unit0],
            },
            "rawCaptures": [raw_stroke],
        }
        enr_p01_b = {
            **enr_p01_a,
            "recordId": "ev_pw_enr_b",
            "profileId": "u_p01_b",
            "profileName": "참가자1_B",
        }
        enr_p02 = {
            **enr_p01_a,
            "recordId": "ev_pw_enr_p02",
            "profileId": "u_p02",
            "profileName": "참가자2",
            "env": {"participantId": "P02", "keyboardType": "mechanical_tactile", "detectedInputMode": "keyboard"},
            "templateSummary": {
                "count": 4,
                "holdMs": 95,
                "centroid": unit1,
                "exemplars": [unit1],
                "centroid17": unit1,
                "exemplars17": [unit1],
            },
        }
        # P01 logs into their own second account (u_p01_b) on mechanical_linear while dropdown stayed on enrolled:u_p01_a
        q_same_subj = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_pw_q_same",
            "eventType": "query",
            "domain": "password",
            "roomCode": "STUDY1",
            "recordedAt": "2026-11-19T01:05:00Z",
            "queryEnv": {"participantId": "P01", "keyboardType": "mechanical_linear", "detectedInputMode": "keyboard"},
            "targetAccount": {"id": "u_p01_b", "name": "참가자1_B", "enrollEnv": enr_p01_b["env"]},
            "groundTruth": {"role": "impostor", "isGenuine": False, "selectedRoleRaw": "enrolled:u_p01_a"},
            "passwordMeta": {"length": 12, "publicExperimentId": "EXP12_ALT_A", "isPublicAllowlist": True},
            "verdict": {
                "accepted": True,
                "z": 2.4,
                "lengthBaseline": {"z": 2.2, "accepted": True},
                "stringCalibrated": {"z": 2.4, "accepted": True, "latencyMs": 0.02},
            },
            "optInDiagnostics": {"rawCapture": {**raw_stroke, "holdMs": 88, "holds": [0.088] * 9, "hCurr": [0.088] * 8, "hNext": [0.088] * 8}},
        }
        # P01 attacks P02's account -> must remain impostor
        q_diff_subj = {
            **q_same_subj,
            "recordId": "ev_pw_q_diff",
            "recordedAt": "2026-11-19T01:06:00Z",
            "targetAccount": {"id": "u_p02", "name": "참가자2", "enrollEnv": enr_p02["env"]},
            "verdict": {
                "accepted": False,
                "z": -1.5,
                "lengthBaseline": {"z": -1.4, "accepted": False},
                "stringCalibrated": {"z": -1.5, "accepted": False, "latencyMs": 0.02},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pw_cross.jsonl"
            p.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in [enr_p01_a, enr_p01_b, enr_p02, q_same_subj, q_diff_subj]) + "\n",
                encoding="utf-8",
            )
            rep = analyze_jsonl_files([p], repo_root=REPO_ROOT)

        pw_all = rep["environment_and_errors"]["password"]["overall"]
        self.assertEqual(pw_all["n_genuine"], 1)
        self.assertEqual(pw_all["n_impostor"], 1)
        self.assertEqual(pw_all["genuine_accept_rate"], 1.0)
        self.assertEqual(pw_all["impostor_far"], 0.0)
        pw_rescore = rep["hold_calibration"]["rescoring"]["password_cross_switch"]["physics_switch_plus_shrink"]
        self.assertEqual(pw_rescore["n_cross_switch_queries"], 2)
        self.assertEqual(pw_rescore["n_genuine"], 1)
        self.assertEqual(pw_rescore["n_impostor"], 1)

    def test_posture_gated_motion_fusion_protects_cross_posture_queries(self) -> None:
        enr_hand = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_m_enr",
            "eventType": "enroll",
            "recordedAt": "2026-11-19T03:00:00Z",
            "domain": "arena",
            "lang": "ko",
            "roomCode": "STUDY1",
            "profileId": "u_m1",
            "profileName": "모바일참가자",
            "env": {"participantId": "P01", "keyboardType": "touch_phone", "posture": "hand_two_thumbs", "detectedInputMode": "touch"},
            "deviceMotion": {
                "available": True,
                "effectiveHz": 58.8,
                "medianIntervalMs": 17.0,
                "tapImpulseMean": 0.68,
                "linearAccelRms": 0.55,
                "gyroRms": 11.2,
                "pitchDeg": -55.0,
                "rollDeg": 8.5,
                "gravityJitterRms": 0.58,
                "inferredPosture": "in_hand",
            },
        }
        q_hand = {
            "schema": "kp-eval-log-v1",
            "recordId": "ev_m_q_hand",
            "eventType": "query",
            "recordedAt": "2026-11-19T03:02:00Z",
            "domain": "arena",
            "lang": "ko",
            "roomCode": "STUDY1",
            "queryEnv": {"participantId": "P01", "keyboardType": "touch_phone", "posture": "hand_two_thumbs", "detectedInputMode": "touch"},
            "groundTruth": {"role": "enrolled", "profileId": "u_m1"},
            "verdict": {"openSetThreshold": 0.72, "marginThreshold": 0.03},
            "rankings": [
                {"id": "u_m1", "name": "모바일참가자", "rank": 1, "similarity": 0.74, "embeddingSimilarity": 0.70, "behaviorLlr": 0.8},
                {"id": "u_m2", "name": "비교군", "rank": 2, "similarity": 0.42, "embeddingSimilarity": 0.42, "behaviorLlr": 0.0},
            ],
            "deviceMotion": {
                "available": True,
                "effectiveHz": 58.8,
                "medianIntervalMs": 17.0,
                "tapImpulseMean": 0.69,
                "linearAccelRms": 0.53,
                "gyroRms": 11.8,
                "pitchDeg": -52.0,
                "rollDeg": 9.0,
                "gravityJitterRms": 0.56,
                "inferredPosture": "in_hand",
            },
        }
        # Phone placed flat on desk (including slight camera-bump rocking within flat threshold):
        # keystroke similarity 0.74 > threshold 0.72, naive motion penalty drops it below 0.72, posture-gated preserves 0.74!
        q_desk = {
            **q_hand,
            "recordId": "ev_m_q_desk",
            "recordedAt": "2026-11-19T03:05:00Z",
            "queryEnv": {"participantId": "P01", "keyboardType": "touch_phone", "posture": "desk_phone_flat", "detectedInputMode": "touch"},
            "deviceMotion": {
                "available": True,
                "effectiveHz": 59.9,
                "medianIntervalMs": 16.7,
                "tapImpulseMean": 0.07,
                "linearAccelRms": 0.05,
                "gyroRms": 7.2,
                "pitchDeg": -1.6,
                "rollDeg": 176.0,
                "gravityJitterRms": 0.61,
                "inferredPosture": "in_hand",  # stale browser field overridden by physical rule -> on_desk_flat
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "motion_posture.jsonl"
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [enr_hand, q_hand, q_desk]) + "\n", encoding="utf-8")
            rep = analyze_jsonl_files([p], repo_root=REPO_ROOT)

        tm = rep["touch_and_motion"]
        self.assertEqual(tm["device_motion_audit"]["posture_inference_agreement"], 1.0)
        self.assertGreater(
            tm["sensor_fusion_ab"]["in_hand"]["keystroke_plus_motion_posture_gated"]["mean_top1_score"],
            tm["sensor_fusion_ab"]["in_hand"]["keystroke_only"]["mean_top1_score"],
        )
        self.assertEqual(tm["sensor_fusion_ab"]["on_desk_flat"]["keystroke_plus_motion"]["genuine_accept_rate"], 0.0)
        self.assertEqual(tm["sensor_fusion_ab"]["on_desk_flat"]["keystroke_plus_motion_posture_gated"]["genuine_accept_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
