"""
Reproducible offline evaluation for the 3-pillar follow-up study:
  1) [Model] User-chosen password string-specific lightweight cohort calibration
             (Train 68 people x stratified Train strings -> Val 10 people x disjoint Val strings
              -> Test 18 people x disjoint Test strings + real repeated words + KeyRecs fixed text + CMU).
  2) [Data/Signal] Physics-consistent cross-switch Hold calibration (`UD = DD - H_curr` preserved
             vs naive H-only adjustment vs uncalibrated under controlled switch hold shifts).
  3) [Data/Mobile] Smartphone touch behaviour fusion (`beta = 0`, `0.2`, tuned `beta*`, and
             touch-recalibrated `(sigma2, delta2, tau2)` on KeyRecs touch 5-enroll + 2-query Arena).

Outputs:
  - models/calibration_and_cross_device_study.json
  - web/public/models/string_cohort_calibration.json

Usage:
  .venv/bin/python src/eval_calibration_and_cross_device.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_profile import BehaviorModel, Event, fit_behavior_model, kind_of, load_keyrecs_sessions  # noqa: E402
from capture_sim import build_input_tensor, make_template, ms_rounded  # noqa: E402
from eval_password_calibration import build_scenarios as build_real_password_scenarios  # noqa: E402
from feature_eval import (  # noqa: E402
    BETA_GRID,
    attach_tensors,
    build_capture_bank,
    embed_bank,
    encode_variable,
    open_set_eval,
    open_set_matrix,
)
from models import KeyPrintNet  # noqa: E402
from password_synth import (  # noqa: E402
    CALIB_LENGTHS,
    digraph_tables,
    dual_z,
    params_for_length,
    password_rng,
    raw_similarity,
    string_trials,
)
from preprocessing import (  # noqa: E402
    N_CHANNELS,
    N_TIMING_CHANNELS,
    SEED,
    prepare_full_pipeline,
    touchify_timings,
    apply_channel_scaler,
    extract_sequences,
)
from string_calibration import (  # noqa: E402
    PUBLIC_EXPERIMENT_STRINGS,
    STRING_FEATURE_NAMES,
    apply_hold_calibration,
    build_digraph_summary,
    extract_string_features,
    fit_ridge_residual_model,
    generate_stratified_strings,
    predict_string_dual_params,
)
from train_and_evaluate import compute_cohort_centroids, compute_eer, encode_sequences  # noqa: E402


def _frr_at_fixed_far(y: np.ndarray, scores: np.ndarray, target_far: float) -> Tuple[float, float]:
    """Return `(frr, threshold)` at the highest sensitivity achieving impostor FAR <= `target_far`."""
    imp = scores[y == 0]
    gen = scores[y == 1]
    if len(imp) == 0 or len(gen) == 0:
        return float("nan"), 0.0
    # Strict quantile threshold so empirical FAR <= target_far (up to ties)
    thr = float(np.quantile(imp, max(0.0, min(1.0, 1.0 - target_far))))
    # Nudge slightly if ties exceed target_far
    if float((imp >= thr).mean()) > target_far + 1e-9:
        thr = math.nextafter(thr, math.inf)
    frr = float((gen < thr).mean())
    return frr, thr


def _paired_cluster_bootstrap_frr_diff(
    rows: Sequence[Dict],
    s_base: np.ndarray,
    s_cand: np.ndarray,
    target_far: float,
    n_boot: int = 1000,
    seed: int = 20261119,
) -> Dict[str, object]:
    """Bootstrap by target participant `(cand_frr - base_frr)` at matched `target_far`."""
    y = np.asarray([r["label"] for r in rows], dtype=np.int32)
    targets = np.asarray([r["target"] for r in rows])
    uniq_t = sorted(set(targets))
    idx_by_t = {t: np.where(targets == t)[0] for t in uniq_t}

    frr_b, _ = _frr_at_fixed_far(y, s_base, target_far)
    frr_c, _ = _frr_at_fixed_far(y, s_cand, target_far)
    point_diff = frr_c - frr_b

    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq_t, size=len(uniq_t), replace=True)
        boot_idx = np.concatenate([idx_by_t[t] for t in picked])
        yb = y[boot_idx]
        if yb.sum() == 0 or (yb == 0).sum() == 0:
            continue
        fb, _ = _frr_at_fixed_far(yb, s_base[boot_idx], target_far)
        fc, _ = _frr_at_fixed_far(yb, s_cand[boot_idx], target_far)
        diffs.append(fc - fb)

    ci = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))] if diffs else [point_diff, point_diff]
    return {
        "target_far": round(target_far, 4),
        "baseline_frr": round(frr_b, 4),
        "candidate_frr": round(frr_c, 4),
        "diff_cand_minus_base": round(point_diff, 4),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
    }


def _string_dispersion_metrics(rows: Sequence[Dict], scores: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    """Measure inter-string FAR/FRR dispersion when applying each row's deployed threshold."""
    y = np.asarray([r["label"] for r in rows], dtype=np.int32)
    acc = scores >= thresholds
    by_word_imp: Dict[str, List[bool]] = defaultdict(list)
    by_word_gen: Dict[str, List[bool]] = defaultdict(list)
    for r, a in zip(rows, acc):
        if r["label"] == 0:
            by_word_imp[r["word"]].append(bool(a))
        else:
            by_word_gen[r["word"]].append(not bool(a))
    str_fars = [float(np.mean(v)) for v in by_word_imp.values() if len(v) >= 5]
    str_frrs = [float(np.mean(v)) for v in by_word_gen.values() if len(v) >= 4]
    return {
        "pooled_far": round(float(acc[y == 0].mean()), 4),
        "pooled_frr": round(float((~acc[y == 1]).mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "eer": round(float(compute_eer(y, scores)[0]), 4),
        "per_string_far_std": round(float(np.std(str_fars)), 4) if str_fars else 0.0,
        "per_string_far_p90": round(float(np.quantile(str_fars, 0.90)), 4) if str_fars else 0.0,
        "per_string_far_max": round(float(np.max(str_fars)), 4) if str_fars else 0.0,
        "per_string_frr_std": round(float(np.std(str_frrs)), 4) if str_frrs else 0.0,
    }


def _score_rows_baseline_and_string(
    rows: Sequence[Dict],
    mode: str,
    by_length_table: List[Dict],
    calib_bundle: Dict,
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sub = [r for r in rows if r["mode"] == mode]
    z_base = np.zeros(len(sub), dtype=np.float64)
    t_base = np.zeros(len(sub), dtype=np.float64)
    z_cal = np.zeros(len(sub), dtype=np.float64)
    t_cal = np.zeros(len(sub), dtype=np.float64)

    cache_base: Dict[int, Dict[str, float]] = {}
    cache_cal: Dict[str, Dict[str, float]] = {}
    for i, r in enumerate(sub):
        w = r["word"]
        L = len(w)
        if L not in cache_base:
            cache_base[L] = params_for_length(by_length_table, L, mode)
        if w not in cache_cal:
            cache_cal[w] = predict_string_dual_params(w, mode, by_length_table, calib_bundle)
        pb = cache_base[L]
        pc = cache_cal[w]
        z_base[i] = (r["raw9"] - pb["mu9"]) / pb["sd9"] + (r["raw17"] - pb["mu17"]) / pb["sd17"]
        t_base[i] = pb["tau"]
        z_cal[i] = (r["raw9"] - pc["mu9"]) / pc["sd9"] + (r["raw17"] - pc["mu17"]) / pc["sd17"]
        t_cal[i] = pc["tau"]
    return sub, z_base, t_base, z_cal, t_cal


def _recalibrate_tau_offset(
    val_rows: Sequence[Dict],
    by_length_table: List[Dict],
    calib_partial: Dict,
    frr_target: float = 0.10,
    min_len: int = 12,
) -> Dict[str, float]:
    """Fit a single 1-DOF scalar `tau_offset` per mode on Validation genuine trials (`L >= min_len`)
    so the 68-subject per-length `tau(L)` curve from `model_config.json` is preserved without 10-subject quantile noise."""
    offsets: Dict[str, float] = {}
    zero_bundle = {**calib_partial, "tau_offset": {"keyboard": 0.0, "touch": 0.0}}
    for mode, m_key in (("kb", "keyboard"), ("touch", "touch")):
        r_sub = [x for x in val_rows if x["mode"] == mode and x["label"] == 1 and len(x["word"]) >= min_len]
        if len(r_sub) < 10:
            offsets[m_key] = 0.0
            continue
        margins = []
        for x in r_sub:
            pc = predict_string_dual_params(x["word"], mode, by_length_table, zero_bundle)
            z = (x["raw9"] - pc["mu9"]) / pc["sd9"] + (x["raw17"] - pc["mu17"]) / pc["sd17"]
            margins.append(z - pc["tau"])
        offsets[m_key] = round(float(np.quantile(margins, frr_target)), 4)
    return offsets


def _touch_sessions_only(raw_sessions: Dict[Tuple[str, int], Sequence]) -> Dict[Tuple[str, int], List[Event]]:
    """Convert raw KeyRecs sessions to touch-equivalent Event streams (no hold, no Shift/CapsLock)."""
    out: Dict[Tuple[str, int], List[Event]] = {}
    for key, stream in raw_sessions.items():
        evs: List[Event] = []
        for label, t, _ in ms_rounded(stream):
            kind, _ = kind_of(label)
            if kind in ("char", "space", "bs"):
                evs.append(Event(kind=kind, t=float(t), hold=None, side=None, rep=False))
        out[key] = evs
    return out


def _top1_from_matrix(mat: Dict) -> float:
    S, P = mat["S"], mat["person"]
    preds = np.argmax(S, axis=1)
    return round(float(np.mean(preds == P)), 4)


def main() -> None:
    t_start = time.time()
    repo_root = Path(os.environ.get("KEYPRINT_REPO_ROOT", Path(__file__).resolve().parents[1]))
    out_root = Path(os.environ.get("KEYPRINT_OUTPUT_ROOT", repo_root))
    device = torch.device("cpu")

    shipped_cfg = json.loads((repo_root / "web/public/models/model_config.json").read_text(encoding="utf-8"))
    shipped_eval = json.loads((repo_root / "models/evaluation_report.json").read_text(encoding="utf-8"))
    by_length_table = shipped_cfg["password_dual"]["by_length"]
    frr_target = float(shipped_cfg["password_dual"].get("frr_target", 0.10))

    print("[1/4] Preparing canonical subject-disjoint splits & loading frozen encoders...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        splits, meta = prepare_full_pipeline(
            repo_root / "data/raw/DSL-StrongPasswordData.csv", Path(tmp), seed=SEED
        )
    kr = meta["_keyrecs_splits_subjs"]
    mean = np.asarray(meta["channel_mean"], dtype=np.float32)
    std = np.asarray(meta["channel_std"], dtype=np.float32)
    bounds = {k: tuple(v) for k, v in meta["outlier_bounds"].items()}
    touch_hold = float(meta["touch_hold_impute_sec"])

    model17 = KeyPrintNet(in_channels=N_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18).to(device)
    model17.load_state_dict(torch.load(out_root / "models/keyprint_net.pt", map_location="cpu", weights_only=True))
    model17.eval()

    model9 = KeyPrintNet(in_channels=N_TIMING_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18).to(device)
    model9.load_state_dict(torch.load(out_root / "models/legacy/keyprint_net_9ch.pt", map_location="cpu", weights_only=True))
    model9.eval()

    embed9 = lambda x: encode_sequences(model9, x[:, :N_TIMING_CHANNELS], device)  # noqa: E731
    embed17 = lambda x: encode_sequences(model17, x[:, :N_CHANNELS], device)  # noqa: E731

    tr_h, tr_d = splits["train"].raw_holds, splits["train"].raw_dds
    t_h, t_u = touchify_timings(tr_h, np.clip(tr_d, *bounds["DD"]).astype(np.float32), touch_hold, bounds)
    tr_touch_seq = apply_channel_scaler(
        extract_sequences(t_h, np.clip(tr_d, *bounds["DD"]).astype(np.float32), t_u, splits["train"].key_block, np.zeros(len(tr_d), dtype=np.float32)),
        mean,
        std,
    )
    cohorts9 = {
        "kb": compute_cohort_centroids(embed9(splits["train"].seq_scaled), splits["train"].subject_ids),
        "touch": compute_cohort_centroids(embed9(tr_touch_seq), splits["train"].subject_ids),
    }

    raw_sessions = load_keyrecs_sessions(repo_root / "data/raw/KeyRecs-free-text.csv")
    ms_sessions = {k: ms_rounded(v) for k, v in raw_sessions.items()}

    # -----------------------------------------------------------------------------------
    # PILLAR 1 (⑤ + ⑥): User-chosen password string-specific cohort calibration
    # -----------------------------------------------------------------------------------
    print("[2/4] Fitting Train-only digraph summary & evaluating string-specific calibration...", flush=True)
    tabs_train = digraph_tables(ms_sessions, kr["train"])
    tabs_val = digraph_tables(ms_sessions, kr["val"])
    tabs_test = digraph_tables(ms_sessions, kr["test"])

    digraph_summary = build_digraph_summary(tabs_train, kr["train"], min_count=5)

    # Exclude any string already used in canonical evaluation_report.json to keep 100% clean separation
    existing_words = set()
    fu = shipped_eval["feature_upgrade"]
    for sp in ("val", "test"):
        for g in fu["password_user_strings"]["strings"][sp].values():
            existing_words.update(g)
        for g in fu["password_length_calibration"]["strings"][sp].values():
            existing_words.update(g)
        existing_words.update(fu["password_capital_method"]["strings"][sp])

    rng_tr = np.random.default_rng(np.random.SeedSequence([SEED, 901, 0]))
    rng_va = np.random.default_rng(np.random.SeedSequence([SEED, 901, 1]))
    rng_te = np.random.default_rng(np.random.SeedSequence([SEED, 901, 2]))

    str_train = generate_stratified_strings(tabs_train, kr["train"], CALIB_LENGTHS, per_length=28, rng=rng_tr, exclude=existing_words, min_share=0.55)
    used_tr = {item["word"] for items in str_train.values() for item in items}

    str_val = generate_stratified_strings(tabs_val, kr["val"], CALIB_LENGTHS, per_length=12, rng=rng_va, exclude=existing_words | used_tr, min_share=0.70)
    used_va = {item["word"] for items in str_val.values() for item in items}

    str_test = generate_stratified_strings(tabs_test, kr["test"], CALIB_LENGTHS, per_length=16, rng=rng_te, exclude=existing_words | used_tr | used_va, min_share=0.65)
    used_te = {item["word"] for items in str_test.values() for item in items}

    assert used_tr.isdisjoint(used_va) and used_tr.isdisjoint(used_te) and used_va.isdisjoint(used_te)
    assert set(kr["train"]).isdisjoint(kr["val"]) and set(kr["train"]).isdisjoint(kr["test"]) and set(kr["val"]).isdisjoint(kr["test"])

    def _collect_split_rows(tabs_sp, people_sp, str_by_len, seed_tag: int) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
        all_r: List[Dict] = []
        by_l: Dict[int, List[Dict]] = {}
        for L, items in sorted(str_by_len.items()):
            by_cat: Dict[str, List[str]] = defaultdict(list)
            for it in items:
                by_cat[it["stratum"]].append(it["word"])
            r_rng = np.random.default_rng(np.random.SeedSequence([SEED, 902, seed_tag, L]))
            r_len = string_trials(
                ms_sessions, tabs_sp, people_sp, dict(by_cat), embed9, embed17, bounds, mean, std, touch_hold, cohorts9, r_rng
            )
            by_l[L] = r_len
            all_r.extend(r_len)
        return all_r, by_l

    train_rows, _ = _collect_split_rows(tabs_train, kr["train"], str_train, 0)
    val_rows, val_rows_by_len = _collect_split_rows(tabs_val, kr["val"], str_val, 1)
    test_rows, test_rows_by_len = _collect_split_rows(tabs_test, kr["test"], str_test, 2)

    val_ge12 = [r for r in val_rows if len(r["word"]) >= 12]
    test_ge12 = [r for r in test_rows if len(r["word"]) >= 12]

    # Select (ridge_alpha, support_kappa, shrink_scale) strictly on Validation (L >= 12 deployed Guard domain)!
    best_cfg_by_mode: Dict[str, Tuple[float, float, float, float]] = {}  # mode_key -> (obj, alpha, kappa, shrink)
    val_sweep_log: List[Dict] = []
    for r_alpha in (4.0, 12.0, 30.0):
        for kappa in (25.0, 60.0):
            cand_models = fit_ridge_residual_model(train_rows, by_length_table, digraph_summary, ridge_alpha=r_alpha, support_kappa=kappa)
            for shrink in (0.0, 0.4, 0.7, 1.0):
                partial = {
                    "digraph_table": digraph_summary,
                    "support_kappa": kappa,
                    "shrink_scale": {"keyboard": shrink, "touch": shrink},
                    "models": cand_models,
                }
                tau_off = _recalibrate_tau_offset(val_rows, by_length_table, partial, frr_target=frr_target, min_len=12)
                full_b = {**partial, "tau_offset": tau_off}
                for mode, m_key in (("kb", "keyboard"), ("touch", "touch")):
                    sub_v, zb, tb, zc, tc = _score_rows_baseline_and_string(val_ge12, mode, by_length_table, full_b)
                    yv = np.asarray([x["label"] for x in sub_v], dtype=np.int32)
                    frr5, _ = _frr_at_fixed_far(yv, zc, 0.05)
                    frr10, _ = _frr_at_fixed_far(yv, zc, 0.10)
                    disp = _string_dispersion_metrics(sub_v, zc, tc)
                    # Objective: primary FRR @ FAR=5% & 10% (constant-FAR `z`) + deployed per-string FAR std penalty
                    obj = 0.5 * (frr5 + frr10) + 0.25 * disp["per_string_far_std"]
                    val_sweep_log.append({
                        "mode": m_key, "ridge_alpha": r_alpha, "support_kappa": kappa, "shrink_scale": shrink,
                        "val_frr_at_far5": round(frr5, 4), "val_frr_at_far10": round(frr10, 4),
                        "val_per_string_far_std": disp["per_string_far_std"], "val_objective": round(obj, 5),
                    })
                    prev = best_cfg_by_mode.get(m_key)
                    if prev is None or (round(obj, 5), shrink) < (round(prev[0], 5), prev[3]):
                        best_cfg_by_mode[m_key] = (obj, r_alpha, kappa, shrink)

    # Assemble chosen Validation model per mode
    chosen_kappa = 40.0
    final_models: Dict[str, Dict] = {}
    chosen_shrink: Dict[str, float] = {}
    for m_key in ("keyboard", "touch"):
        _, best_a, best_k, best_s = best_cfg_by_mode[m_key]
        m_fit = fit_ridge_residual_model(train_rows, by_length_table, digraph_summary, ridge_alpha=best_a, support_kappa=best_k)
        final_models[m_key] = {**m_fit[m_key], "ridge_alpha": best_a, "support_kappa": best_k}
        chosen_shrink[m_key] = best_s
        chosen_kappa = best_k

    calib_bundle = {
        "schema": "kp-string-calib-v1",
        "feature_names": list(STRING_FEATURE_NAMES),
        "support_kappa": chosen_kappa,
        "shrink_scale": chosen_shrink,
        "models": final_models,
        "digraph_table": digraph_summary,
    }
    calib_bundle["tau_offset"] = _recalibrate_tau_offset(
        val_rows, by_length_table, calib_bundle, frr_target=frr_target, min_len=12
    )

    # Evaluate on Validation, deployed Guard domain (`L >= 12`), all lengths (`8..20`), and `12-char` subset
    string_eval: Dict[str, Dict] = {
        "validation_ge12": {},
        "test_ge12": {},
        "test_all_lengths": {},
        "test_len12": {},
        "real_non_synthetic_test": {},
    }
    for split_name, rows_src in (
        ("validation_ge12", val_ge12),
        ("test_ge12", test_ge12),
        ("test_all_lengths", test_rows),
        ("test_len12", [r for r in test_rows if len(r["word"]) == 12]),
    ):
        for mode, m_key in (("kb", "keyboard"), ("touch", "touch")):
            sub, zb, tb, zc, tc = _score_rows_baseline_and_string(rows_src, mode, by_length_table, calib_bundle)
            base_disp = _string_dispersion_metrics(sub, zb, tb)
            cand_disp = _string_dispersion_metrics(sub, zc, tc)
            boot5 = _paired_cluster_bootstrap_frr_diff(sub, zb, zc, 0.05)
            boot10 = _paired_cluster_bootstrap_frr_diff(sub, zb, zc, 0.10)
            boot_margin_dep = _paired_cluster_bootstrap_frr_diff(
                sub, zb - tb, zc - tc, max(0.01, base_disp["pooled_far"])
            )
            string_eval[split_name][m_key] = {
                "num_trials": len(sub),
                "num_strings": len({x["word"] for x in sub}),
                "num_people": len({x["target"] for x in sub}),
                "length_only_baseline": base_disp,
                "string_calibrated": cand_disp,
                "frr_at_far_5pct": boot5,
                "frr_at_far_10pct": boot10,
                "margin_frr_at_baseline_deployed_far": boot_margin_dep,
            }

    # Also test on non-synthetic real recordings (`krfixed`, `words_cross_day`, `cmu_test`)!
    scen9, _ = build_real_password_scenarios(repo_root, out_root, device, "9")
    scen17, meta17 = build_real_password_scenarios(repo_root, out_root, device, "17")
    real_words_test = meta17["words"]

    for scen_key, fixed_str in (
        ("cmu_test", ".tie5Roanl"),
        ("krfixed_test", "vpwjkeurkb"),
        ("words_cross_day", None),
        ("words_same_day", None),
    ):
        string_eval["real_non_synthetic_test"][scen_key] = {}
        for mode, m_key in (("kb", "keyboard"), ("touch", "touch")):
            full_name = f"{scen_key}_{mode}"
            if full_name not in scen9:
                continue
            g9_list = scen9[full_name]
            g17_list = scen17[full_name]
            real_rows: List[Dict] = []
            # Map each group back to its word (`fixed_str` or natural word list)
            for gi, ((E9, G9, I9), (E17, G17, I17)) in enumerate(zip(g9_list, g17_list)):
                t9, t17 = make_template(np.asarray(E9, dtype=np.float64)), make_template(np.asarray(E17, dtype=np.float64))
                w_guess = fixed_str if fixed_str is not None else real_words_test[min(gi, len(real_words_test) - 1)]
                for Q9, Q17, lab in ((G9, G17, 1), (I9, I17, 0)):
                    r9 = raw_similarity(np.asarray(Q9, dtype=np.float64), t9)
                    r17 = raw_similarity(np.asarray(Q17, dtype=np.float64), t17)
                    for v9, v17 in zip(r9, r17):
                        real_rows.append({
                            "word": w_guess, "mode": mode, "target": f"g{gi}", "label": lab,
                            "raw9": float(v9), "raw17": float(v17),
                        })
            sub, zb, tb, zc, tc = _score_rows_baseline_and_string(real_rows, mode, by_length_table, calib_bundle)
            y_r = np.asarray([x["label"] for x in sub], dtype=np.int32)
            frr10_b, _ = _frr_at_fixed_far(y_r, zb, 0.10)
            frr10_c, _ = _frr_at_fixed_far(y_r, zc, 0.10)
            string_eval["real_non_synthetic_test"][scen_key][m_key] = {
                "num_genuine": int(y_r.sum()),
                "num_impostor": int((y_r == 0).sum()),
                "baseline_auc": round(float(roc_auc_score(y_r, zb)), 4),
                "calibrated_auc": round(float(roc_auc_score(y_r, zc)), 4),
                "baseline_frr_at_far10": round(frr10_b, 4),
                "calibrated_frr_at_far10": round(frr10_c, 4),
            }

    # Adoption decision per mode on the deployed Guard domain (`test_ge12`, L >= 12):
    # require Validation shrink > 0 AND Test FRR@FAR10 <= baseline AND per-string FAR std reduced
    adopted_string: Dict[str, bool] = {}
    for m_key in ("keyboard", "touch"):
        te = string_eval["test_ge12"][m_key]
        val_chose_positive = chosen_shrink[m_key] > 0.0
        frr_improved = te["frr_at_far_10pct"]["diff_cand_minus_base"] <= 0.0
        disp_improved = te["string_calibrated"]["per_string_far_std"] <= te["length_only_baseline"]["per_string_far_std"]
        adopted_string[m_key] = bool(val_chose_positive and frr_improved and disp_improved)

    calib_bundle["adopted"] = adopted_string

    # -----------------------------------------------------------------------------------
    # PILLAR 2 (③): Physics-consistent Cross-Switch Hold Calibration
    # -----------------------------------------------------------------------------------
    print("[3/4] Evaluating physics-consistent Hold calibration under cross-switch offsets...", flush=True)
    kb_val_bank = build_capture_bank(raw_sessions, kr["val"], touch=False, target_chars=68)
    kb_test_bank = build_capture_bank(raw_sessions, kr["test"], touch=False, target_chars=68)
    attach_tensors(kb_val_bank, bounds, touch_hold, mean, std)
    attach_tensors(kb_test_bank, bounds, touch_hold, mean, std)
    embed_bank(kb_val_bank, model17, device, N_CHANNELS, "emb")
    embed_bank(kb_test_bank, model17, device, N_CHANNELS, "emb")

    sent_cohort_kb = np.asarray(shipped_cfg["sentence_cohort_centroids"], dtype=np.float64)
    desktop_beh_model = BehaviorModel({
        f["name"]: {"sigma2": f["sigma2"], "delta2": f["delta2"], "tau2": f["tau2"]}
        for f in shipped_cfg["behavior_profile"]["features"] if f.get("scored") and "sigma2" in f
    })
    kb_beta = float(shipped_cfg["behavior_profile"]["beta"]["sentence_keyboard"])
    kb_rule = {"threshold": float(shipped_cfg["arena_open_set_threshold"]), "margin": float(shipped_cfg["arena_margin"])}

    def _eval_shifted_query_bank(
        base_bank: Dict,
        applied_shift_sec: float,
        mode_kind: str,
        scale_lambda: float = 0.0,
    ) -> Dict[str, float]:
        """Simulate Session-1 enrollment on Switch A (`delta=0`) and Session-2 query on Switch B (`+applied_shift_sec`)."""
        bank = copy.deepcopy(base_bank)
        for p in bank:
            enr_holds = [h for cap in bank[p][1][:5] for h, ok in zip(cap["holds"], cap["hold_valid"]) if ok and h is not None]
            tmpl_mean_h = float(np.mean(enr_holds)) if enr_holds else 0.095
            for idx, cap in enumerate(bank[p][2]):
                # Physical reality on Switch B: holds change by `+applied_shift_sec`, so raw UD = DD - H_curr changes by `-applied_shift_sec`!
                shifted = apply_hold_calibration(cap, delta_sec=-applied_shift_sec, recompute_ud=True)
                if mode_kind == "uncalibrated":
                    proc = shifted
                elif mode_kind == "naive_hold_only":
                    # Shifts H back by `applied_shift_sec` WITHOUT recomputing UD -> violates UD = DD - H_curr!
                    proc = apply_hold_calibration(shifted, delta_sec=applied_shift_sec, recompute_ud=False)
                elif mode_kind == "physics_switch_prior":
                    # Shifts H back AND recomputes UD = DD - H_curr
                    proc = apply_hold_calibration(shifted, delta_sec=applied_shift_sec, recompute_ud=True)
                elif mode_kind == "blind_partial_shrink":
                    # No prior knowledge of `applied_shift_sec`; shrinks query mean hold partially toward template mean hold
                    proc = apply_hold_calibration(
                        shifted, delta_sec=0.0, scale_lambda=scale_lambda, target_mean_hold_sec=tmpl_mean_h, recompute_ud=True
                    )
                else:
                    raise ValueError(mode_kind)
                proc["tensor"] = build_input_tensor(proc, bounds, touch_hold, mean, std)
                bank[p][2][idx] = proc
        embed_bank(bank, model17, device, N_CHANNELS, "emb")
        mat = open_set_matrix(bank, "emb", sent_cohort_kb, desktop_beh_model, kb_beta, 5, 2)
        os5 = open_set_eval(mat, 5, **kb_rule)
        return {
            "top1_18way": _top1_from_matrix(mat),
            "gallery5_correct_accept": os5["correct_accept"],
            "gallery5_outsider_far": os5["outsider_accepted"],
        }

    # Tune blind partial shrink `scale_lambda` on Validation (`+25 ms` cross-switch vs `0 ms` same-switch outsider FAR)
    lambda_grid_log = []
    best_lambda = 0.0
    best_l_score = -1e9
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        same_sw = _eval_shifted_query_bank(kb_val_bank, 0.0, "blind_partial_shrink", scale_lambda=lam)
        cross_sw = _eval_shifted_query_bank(kb_val_bank, 0.025, "blind_partial_shrink", scale_lambda=lam)
        score_val = cross_sw["top1_18way"] + cross_sw["gallery5_correct_accept"] - 1.5 * same_sw["gallery5_outsider_far"]
        lambda_grid_log.append({
            "scale_lambda": lam,
            "val_same_switch_outsider_far": same_sw["gallery5_outsider_far"],
            "val_cross_25ms_top1": cross_sw["top1_18way"],
            "val_cross_25ms_accept": cross_sw["gallery5_correct_accept"],
            "val_score": round(score_val, 4),
        })
        if score_val > best_l_score + 1e-6:
            best_l_score = score_val
            best_lambda = lam

    hold_study: Dict[str, object] = {
        "selected_blind_scale_lambda": best_lambda,
        "validation_lambda_grid": lambda_grid_log,
        "test_same_switch_baseline": _eval_shifted_query_bank(kb_test_bank, 0.0, "uncalibrated"),
        "test_same_switch_with_blind_shrink": _eval_shifted_query_bank(kb_test_bank, 0.0, "blind_partial_shrink", scale_lambda=best_lambda),
        "test_cross_switch_offsets": {},
    }
    for shift_ms in (-25, -15, 15, 25, 35):
        sh_sec = shift_ms / 1000.0
        hold_study["test_cross_switch_offsets"][f"{shift_ms:+d}ms"] = {
            "uncalibrated": _eval_shifted_query_bank(kb_test_bank, sh_sec, "uncalibrated"),
            "naive_hold_only_broken_ud": _eval_shifted_query_bank(kb_test_bank, sh_sec, "naive_hold_only"),
            "blind_partial_shrink_physics": _eval_shifted_query_bank(kb_test_bank, sh_sec, "blind_partial_shrink", scale_lambda=best_lambda),
            "switch_prior_physics": _eval_shifted_query_bank(kb_test_bank, sh_sec, "physics_switch_prior"),
        }

    # -----------------------------------------------------------------------------------
    # PILLAR 3 (④): Mobile Touch `beta` & Touch-Specific `(sigma2, delta2, tau2)`
    # -----------------------------------------------------------------------------------
    print("[4/4] Evaluating smartphone touch behaviour beta & touch-specific LLR variances...", flush=True)
    touch_raw_sessions = _touch_sessions_only(raw_sessions)
    touch_beh_model = fit_behavior_model(touch_raw_sessions, kr["train"], window=70)

    touch_val_bank = build_capture_bank(raw_sessions, kr["val"], touch=True, target_chars=68)
    touch_test_bank = build_capture_bank(raw_sessions, kr["test"], touch=True, target_chars=68)
    attach_tensors(touch_val_bank, bounds, touch_hold, mean, std)
    attach_tensors(touch_test_bank, bounds, touch_hold, mean, std)
    embed_bank(touch_val_bank, model17, device, N_CHANNELS, "emb")
    embed_bank(touch_test_bank, model17, device, N_CHANNELS, "emb")

    sent_cohort_touch = np.asarray(shipped_cfg["sentence_cohort_centroids_touch"], dtype=np.float64)
    touch_rule = {
        "threshold": float(shipped_cfg["arena_open_set_threshold_touch"]),
        "margin": float(shipped_cfg["arena_margin_touch"]),
    }

    def _eval_touch_variant(bank: Dict, beh_model: Optional[BehaviorModel], beta_val: float) -> Dict[str, float]:
        mat = open_set_matrix(bank, "emb", sent_cohort_touch, beh_model, beta_val, 5, 2)
        os5 = open_set_eval(mat, 5, **touch_rule)
        return {
            "top1": _top1_from_matrix(mat),
            "correct_accept": os5["correct_accept"],
            "misidentified": os5["misidentified"],
            "outsider_far": os5["outsider_accepted"],
        }

    # Select best `beta` on Validation subject to open-set outsider FAR <= 2.0%
    best_beta_desk, best_score_desk = 0.2, -1e9
    best_beta_touch, best_score_touch = 0.2, -1e9
    touch_val_grid = []
    for b_cand in BETA_GRID:
        vd = _eval_touch_variant(touch_val_bank, desktop_beh_model, float(b_cand))
        vt = _eval_touch_variant(touch_val_bank, touch_beh_model, float(b_cand))
        sd_val = (vd["top1"] + vd["correct_accept"] - 1.5 * vd["outsider_far"]) if vd["outsider_far"] <= 0.02 else -10.0 - vd["outsider_far"]
        st_val = (vt["top1"] + vt["correct_accept"] - 1.5 * vt["outsider_far"]) if vt["outsider_far"] <= 0.02 else -10.0 - vt["outsider_far"]
        touch_val_grid.append({
            "beta": float(b_cand),
            "val_desktop_llr": vd,
            "val_touch_llr": vt,
        })
        if (round(sd_val, 4), -b_cand) > (round(best_score_desk, 4), -best_beta_desk):
            best_score_desk, best_beta_desk = sd_val, float(b_cand)
        if (round(st_val, 4), -b_cand) > (round(best_score_touch, 4), -best_beta_touch):
            best_score_touch, best_beta_touch = st_val, float(b_cand)

    touch_study = {
        "deployed_open_set_rule_touch": touch_rule,
        "selected_beta_desktop_llr": best_beta_desk,
        "selected_beta_touch_llr": best_beta_touch,
        "touch_fitted_feature_count": len(touch_beh_model.params),
        "touch_fitted_features": sorted(touch_beh_model.params.keys()),
        "validation_comparison": {
            "beta_0_no_behavior": _eval_touch_variant(touch_val_bank, desktop_beh_model, 0.0),
            "beta_0_2_deployed_desktop_llr": _eval_touch_variant(touch_val_bank, desktop_beh_model, 0.2),
            "tuned_beta_desktop_llr": _eval_touch_variant(touch_val_bank, desktop_beh_model, best_beta_desk),
            "tuned_beta_touch_recalibrated_llr": _eval_touch_variant(touch_val_bank, touch_beh_model, best_beta_touch),
        },
        "test_comparison": {
            "beta_0_no_behavior": _eval_touch_variant(touch_test_bank, desktop_beh_model, 0.0),
            "beta_0_2_deployed_desktop_llr": _eval_touch_variant(touch_test_bank, desktop_beh_model, 0.2),
            "tuned_beta_desktop_llr": _eval_touch_variant(touch_test_bank, desktop_beh_model, best_beta_desk),
            "tuned_beta_touch_recalibrated_llr": _eval_touch_variant(touch_test_bank, touch_beh_model, best_beta_touch),
        },
        "validation_grid": touch_val_grid,
    }

    te_base_t = touch_study["test_comparison"]["beta_0_2_deployed_desktop_llr"]
    te_cand_t = touch_study["test_comparison"]["tuned_beta_touch_recalibrated_llr"]
    adopted_touch_beh = bool(
        te_cand_t["top1"] >= te_base_t["top1"]
        and te_cand_t["correct_accept"] >= te_base_t["correct_accept"]
        and te_cand_t["outsider_far"] <= te_base_t["outsider_far"]
        and (te_cand_t["top1"] > te_base_t["top1"] or te_cand_t["correct_accept"] > te_base_t["correct_accept"] or te_cand_t["outsider_far"] < te_base_t["outsider_far"])
    )
    touch_study["adopted_touch_recalibrated_llr"] = adopted_touch_beh

    # Export browser companion JSON (`web/public/models/string_cohort_calibration.json`)
    web_calib_payload = {
        **calib_bundle,
        "touch_behavior_profile": {
            **touch_beh_model.to_json(),
            "recommended_beta": best_beta_touch,
            "adopted": adopted_touch_beh,
        },
        "hold_calibration": {
            "recommended_blind_scale_lambda": best_lambda,
            "note": "Cross-switch delta_sec should be estimated from paired multi-device field logs; blind lambda=0 preserves same-switch impostor FAR until device switch type is declared.",
        },
        "public_experiment_strings": PUBLIC_EXPERIMENT_STRINGS,
    }
    web_out_path = out_root / "web/public/models/string_cohort_calibration.json"
    web_out_path.write_text(json.dumps(web_calib_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    report = {
        "schema": "kp-followup-calibration-study-v1",
        "elapsed_sec": round(time.time() - t_start, 2),
        "split_counts": {
            "train_people": len(kr["train"]),
            "val_people": len(kr["val"]),
            "test_people": len(kr["test"]),
            "train_strings": len(used_tr),
            "val_strings": len(used_va),
            "test_strings": len(used_te),
            "digraph_table_entries": len(digraph_summary),
            "web_payload_bytes": web_out_path.stat().st_size,
        },
        "string_cohort_calibration": {
            "chosen_shrink_scale": chosen_shrink,
            "adopted": adopted_string,
            "tau_offset": calib_bundle["tau_offset"],
            "validation_sweep": val_sweep_log,
            **string_eval,
        },
        "hold_calibration_study": hold_study,
        "touch_behavior_study": touch_study,
    }
    report_path = out_root / "models/calibration_and_cross_device_study.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] Wrote {report_path} and {web_out_path} ({web_out_path.stat().st_size} bytes) in {report['elapsed_sec']}s", flush=True)
    print("String calibration adopted:", adopted_string, flush=True)
    print("Test>=12 String Calib (KB):", json.dumps(string_eval["test_ge12"]["keyboard"]["frr_at_far_10pct"]), flush=True)
    print("Test>=12 String Calib (Touch):", json.dumps(string_eval["test_ge12"]["touch"]["frr_at_far_10pct"]), flush=True)
    print("Touch Behavior Test:", json.dumps(touch_study["test_comparison"]), flush=True)


if __name__ == "__main__":
    main()
