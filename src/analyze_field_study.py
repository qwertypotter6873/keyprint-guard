"""
Offline analyzer for multi-device & smartphone `.jsonl` evaluation logs exported from the browser
(`EvalStudyManager.toJsonl()` in `web/src/evalStudy.js`).

Merges logs collected across Mechanical, External Pantograph, Laptop Pantograph, and Smartphone
sessions (including cross-device enrollments/queries synced via room codes or merged `.jsonl` files)
and evaluates all 4 experimental hypotheses:
  1) Environment & Cross-Device Error Matrix (device pair, OS/browser, Hangul 2-beolsik mismatch, posture)
  2) Within-Subject Switch Hold Bias (`Δh`) & Physics-Consistent Hold Calibration Re-Scoring
     (`uncalibrated` vs `naive_h_only` vs `physics_switch_prior` vs `physics_switch_plus_shrink`)
  3) Smartphone Touch Behaviour Weight `β` Sweep (`0`, `0.2`, grid `β*`) & `DeviceMotion` Sensor A/B
     (sampling Hz, `in_hand` vs `on_desk_flat`, keystroke-only vs keystroke + tap-impulse/tilt)
  4) User-Chosen Password String-Specific Cohort Calibration (`lengthBaseline` vs `stringCalibrated`,
     12-char & touch subsets, per-string FAR dispersion, and browser `< 1 ms` runtime overhead)

Usage:
  .venv/bin/python src/analyze_field_study.py logs/*.jsonl --out models/field_study_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_sim import as_norm_similarity, build_input_tensor, fuse  # noqa: E402
from password_synth import dual_z, params_for_length, raw_similarity  # noqa: E402
from string_calibration import apply_hold_calibration  # noqa: E402


PHYSICAL_KB_GROUPS = {
    "mechanical_linear": "mechanical",
    "mechanical_tactile": "mechanical",
    "mechanical_clicky": "mechanical",
    "mechanical_other": "mechanical",
    "pantograph_external": "pantograph_ext",
    "pantograph_laptop": "pantograph_laptop",
    "membrane": "membrane",
}

TOUCH_KB_TYPES = {"touch_phone", "touch_tablet"}
BETA_SWEEP_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def _open_set_accept(scores_desc: Sequence[float], threshold: float, margin: float) -> bool:
    if not scores_desc:
        return False
    second = scores_desc[1] if len(scores_desc) > 1 else -math.inf
    return bool(scores_desc[0] >= threshold and (scores_desc[0] - second) >= margin)


def _safe_rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den > 0 else None


def _safe_mean(vals: Sequence[float], digits: int = 4) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(clean)), digits) if clean else None


def _safe_median(vals: Sequence[float], digits: int = 4) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.median(clean)), digits) if clean else None


def _frr_at_far(labels: Sequence[int], scores: Sequence[float], target_far: float) -> Optional[float]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    imp = s[y == 0]
    gen = s[y == 1]
    if len(imp) == 0 or len(gen) == 0:
        return None
    thr = float(np.quantile(imp, max(0.0, min(1.0, 1.0 - target_far))))
    if float((imp >= thr).mean()) > target_far + 1e-9:
        thr = math.nextafter(thr, math.inf)
    return round(float((gen < thr).mean()), 4)


def _roc_auc_safe(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(labels, dtype=np.int32)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, np.asarray(scores, dtype=np.float64))), 4)


def load_and_merge_jsonl(paths: Sequence[Path]) -> Dict[str, object]:
    """Load and deduplicate records by `recordId` across one or more `.jsonl` files."""
    records_by_id: Dict[str, Dict] = {}
    duplicate_count = 0
    invalid_lines = 0
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if not isinstance(obj, dict) or not obj.get("recordId") or not obj.get("eventType"):
                    invalid_lines += 1
                    continue
                rid = str(obj["recordId"])
                if rid in records_by_id:
                    duplicate_count += 1
                    continue
                records_by_id[rid] = obj

    records = sorted(records_by_id.values(), key=lambda r: str(r.get("recordedAt", "")))
    return {
        "records": records,
        "files_loaded": [str(p) for p in paths],
        "total_records": len(records),
        "duplicates_skipped": duplicate_count,
        "invalid_lines": invalid_lines,
    }


def _index_enrollments(records: Sequence[Dict]) -> Dict[Tuple[str, str, str], Dict]:
    """Index latest enrollment record by `(roomCode, profileId, domKey)` and fallback `('*', profileId, domKey)`."""
    idx: Dict[Tuple[str, str, str], Dict] = {}
    for rec in records:
        if rec.get("eventType") != "enroll":
            continue
        room = str(rec.get("roomCode") or "LOCAL")
        pid = str(rec.get("profileId") or "")
        if not pid:
            continue
        dom = "password" if rec.get("domain") == "password" else f"arena_{rec.get('lang') or 'en'}"
        idx[(room, pid, dom)] = rec
        idx[("*", pid, dom)] = rec
    return idx


def _lookup_enroll(
    enroll_idx: Dict[Tuple[str, str, str], Dict],
    room: str,
    profile_id: Optional[str],
    dom_key: str,
) -> Optional[Dict]:
    if not profile_id:
        return None
    return enroll_idx.get((room, profile_id, dom_key)) or enroll_idx.get(("*", profile_id, dom_key))


# ---------------------------------------------------------------------------------------
# Section 1: Environment & Cross-Device Error Matrix
# ---------------------------------------------------------------------------------------
def _summarize_arena_subset(queries: Sequence[Dict]) -> Dict[str, object]:
    enrolled = [q for q in queries if q.get("groundTruth", {}).get("role") == "enrolled"]
    outsider = [q for q in queries if q.get("groundTruth", {}).get("role") == "outsider"]
    unspec = [q for q in queries if q.get("groundTruth", {}).get("role") not in ("enrolled", "outsider")]
    return {
        "n_total": len(queries),
        "n_enrolled": len(enrolled),
        "n_outsider": len(outsider),
        "n_unspecified": len(unspec),
        "top1_acc": _safe_rate(sum(1 for q in enrolled if q.get("verdict", {}).get("top1Correct") is True), len(enrolled)),
        "genuine_accept_rate": _safe_rate(
            sum(1 for q in enrolled if q.get("verdict", {}).get("decisionCorrect") is True), len(enrolled)
        ),
        "margin_reject_rate": _safe_rate(
            sum(1 for q in enrolled if q.get("verdict", {}).get("rejectedByMargin") is True), len(enrolled)
        ),
        "outsider_far": _safe_rate(sum(1 for q in outsider if q.get("verdict", {}).get("accepted") is True), len(outsider)),
        "mean_top1_score": _safe_mean([q.get("verdict", {}).get("top1Score") for q in queries]),
        "mean_lead_gap": _safe_mean([q.get("verdict", {}).get("leadGap") for q in queries]),
    }


def _summarize_password_subset(queries: Sequence[Dict]) -> Dict[str, object]:
    genuine = [q for q in queries if q.get("groundTruth", {}).get("isGenuine") is True]
    impostor = [q for q in queries if q.get("groundTruth", {}).get("isGenuine") is False]
    labels = [1 if q.get("groundTruth", {}).get("isGenuine") else 0 for q in genuine + impostor]
    scores = [float(q.get("verdict", {}).get("z", 0.0)) for q in genuine + impostor]
    return {
        "n_total": len(queries),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "genuine_accept_rate": _safe_rate(sum(1 for q in genuine if q.get("verdict", {}).get("accepted") is True), len(genuine)),
        "genuine_frr": _safe_rate(sum(1 for q in genuine if not q.get("verdict", {}).get("accepted")), len(genuine)),
        "impostor_far": _safe_rate(sum(1 for q in impostor if q.get("verdict", {}).get("accepted") is True), len(impostor)),
        "roc_auc": _roc_auc_safe(labels, scores),
        "frr_at_far5pct": _frr_at_far(labels, scores, 0.05),
        "frr_at_far10pct": _frr_at_far(labels, scores, 0.10),
    }


def analyze_environment_and_errors(records: Sequence[Dict], enroll_idx: Dict[Tuple[str, str, str], Dict]) -> Dict[str, object]:
    enrolls = [r for r in records if r.get("eventType") == "enroll"]
    arena_q = [r for r in records if r.get("eventType") == "query" and r.get("domain") == "arena"]
    pw_q = [r for r in records if r.get("eventType") == "query" and r.get("domain") == "password"]

    participants = sorted(
        {
            str((r.get("env") or r.get("queryEnv") or {}).get("participantId"))
            for r in records
            if (r.get("env") or r.get("queryEnv") or {}).get("participantId")
        }
    )
    devices = sorted(
        {
            str((r.get("env") or r.get("queryEnv") or {}).get("keyboardType"))
            for r in records
            if (r.get("env") or r.get("queryEnv") or {}).get("keyboardType")
        }
    )
    os_browsers = sorted(
        {
            f"{(r.get('env') or r.get('queryEnv') or {}).get('os', '?')}/{(r.get('env') or r.get('queryEnv') or {}).get('browser', '?')}"
            for r in records
        }
    )
    timer_us = [
        (r.get("env") or r.get("queryEnv") or {}).get("timerResolutionUs")
        for r in records
        if (r.get("env") or r.get("queryEnv") or {}).get("timerResolutionUs") is not None
    ]

    # Verify privacy & experimental control invariants
    custom_pw_records = [
        r
        for r in records
        if r.get("domain") == "password" and not (r.get("passwordMeta") or {}).get("isPublicAllowlist", False)
    ]
    redaction_violations = 0
    for r in custom_pw_records:
        if (r.get("passwordMeta") or {}).get("publicText") is not None:
            redaction_violations += 1
        raw_caps = r.get("rawCaptures") or []
        if r.get("optInDiagnostics") and r["optInDiagnostics"].get("rawCapture"):
            raw_caps = [*raw_caps, r["optInDiagnostics"]["rawCapture"]]
        for rc in raw_caps:
            if rc and (rc.get("pairKeys") is not None or rc.get("pairKeysRedacted") is not True):
                redaction_violations += 1

    adapted_during_eval = sum(1 for q in pw_q if q.get("verdict", {}).get("adaptedAfterVerdict") is True)

    # Enrich cross-device pair keys using merged enroll_idx when enrollment came from another device's JSONL
    arena_by_pair: Dict[str, List[Dict]] = defaultdict(list)
    arena_same, arena_cross = [], []
    arena_by_layout: Dict[str, List[Dict]] = defaultdict(list)
    arena_by_posture: Dict[str, List[Dict]] = defaultdict(list)

    for q in arena_q:
        q_env = q.get("queryEnv") or {}
        q_dev = q_env.get("keyboardType") or "unknown"
        gt = q.get("groundTruth") or {}
        dom_key = f"arena_{q.get('lang') or 'en'}"
        enr_rec = _lookup_enroll(enroll_idx, str(q.get("roomCode") or "LOCAL"), gt.get("profileId"), dom_key)
        e_env = gt.get("enrollEnv") or (enr_rec.get("env") if enr_rec else None) or {}
        e_dev = e_env.get("keyboardType")
        if e_dev:
            pair = f"{e_dev} -> {q_dev}"
            arena_by_pair[pair].append(q)
            if e_dev == q_dev:
                arena_same.append(q)
            else:
                arena_cross.append(q)
        else:
            arena_by_pair[f"(outsider/unlinked) -> {q_dev}"].append(q)

        mismatch = bool(q_env.get("layoutMismatch2Beolsik")) or bool(e_env.get("layoutMismatch2Beolsik"))
        layout_tag = f"{q_env.get('hangulLayout', 'unknown')}{' [mismatch]' if mismatch else ''}"
        arena_by_layout[layout_tag].append(q)
        arena_by_posture[str(q_env.get("posture") or "unknown")].append(q)

    pw_by_pair: Dict[str, List[Dict]] = defaultdict(list)
    pw_same, pw_cross = [], []
    pw_by_slot: Dict[str, List[Dict]] = defaultdict(list)
    for q in pw_q:
        q_env = q.get("queryEnv") or {}
        q_dev = q_env.get("keyboardType") or "unknown"
        tgt = q.get("targetAccount") or {}
        enr_rec = _lookup_enroll(enroll_idx, str(q.get("roomCode") or "LOCAL"), tgt.get("id"), "password")
        e_env = tgt.get("enrollEnv") or (enr_rec.get("env") if enr_rec else None) or {}
        e_dev = e_env.get("keyboardType")
        if e_dev:
            pair = f"{e_dev} -> {q_dev}"
            pw_by_pair[pair].append(q)
            if e_dev == q_dev:
                pw_same.append(q)
            else:
                pw_cross.append(q)
        else:
            pw_by_pair[f"(sample/unlinked) -> {q_dev}"].append(q)
        pw_by_slot[str(q_env.get("sessionSlot") or "unknown")].append(q)

    return {
        "counts": {
            "total_records": len(records),
            "enrollments": len(enrolls),
            "arena_queries": len(arena_q),
            "password_queries": len(pw_q),
            "participants": participants,
            "devices": devices,
            "os_browsers": os_browsers,
            "median_timer_resolution_us": _safe_median(timer_us, 1),
        },
        "protocol_integrity": {
            "custom_password_records": len(custom_pw_records),
            "custom_password_redaction_violations": redaction_violations,
            "custom_password_redaction_ok": redaction_violations == 0,
            "password_queries_adapted_during_eval": adapted_during_eval,
            "template_lock_ok": adapted_during_eval == 0,
        },
        "arena": {
            "overall": _summarize_arena_subset(arena_q),
            "same_device": _summarize_arena_subset(arena_same),
            "cross_device": _summarize_arena_subset(arena_cross),
            "by_device_pair": {k: _summarize_arena_subset(v) for k, v in sorted(arena_by_pair.items())},
            "by_hangul_layout": {k: _summarize_arena_subset(v) for k, v in sorted(arena_by_layout.items())},
            "by_posture": {k: _summarize_arena_subset(v) for k, v in sorted(arena_by_posture.items())},
        },
        "password": {
            "overall": _summarize_password_subset(pw_q),
            "same_device": _summarize_password_subset(pw_same),
            "cross_device": _summarize_password_subset(pw_cross),
            "by_device_pair": {k: _summarize_password_subset(v) for k, v in sorted(pw_by_pair.items())},
            "by_session_slot": {k: _summarize_password_subset(v) for k, v in sorted(pw_by_slot.items())},
        },
    }


# ---------------------------------------------------------------------------------------
# Section 2: Within-Subject Switch Hold Bias & Physics-Consistent Re-Scoring
# ---------------------------------------------------------------------------------------
def _mean_valid_hold_sec(raw_cap: Dict) -> Optional[float]:
    holds = raw_cap.get("holds") or []
    hv = raw_cap.get("holdValid") or []
    vals = [float(h) for h, ok in zip(holds, hv) if ok and h is not None]
    if vals:
        return float(np.mean(vals))
    hm = raw_cap.get("holdMs")
    if hm and float(hm) > 0:
        return float(hm) / 1000.0
    return None


def _load_frozen_encoders(repo_root: Path) -> Optional[Dict[str, object]]:
    """Load frozen PyTorch checkpoints + model_config.json when available for rawCapture re-scoring."""
    cfg_path = repo_root / "web" / "public" / "models" / "model_config.json"
    pt17_path = repo_root / "models" / "keyprint_net.pt"
    pt9_candidates = [
        repo_root / "models" / "legacy" / "keyprint_net_9ch.pt",
        repo_root / "models" / "keyprint_net_9ch.pt",
    ]
    if not (cfg_path.exists() and pt17_path.exists()):
        return None

    import torch
    from models import KeyPrintNet

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ck17 = torch.load(pt17_path, map_location="cpu", weights_only=True)
    sd17 = ck17["model_state_dict"] if isinstance(ck17, dict) and "model_state_dict" in ck17 else ck17
    in17 = int(cfg.get("in_channels", 17))
    hid17 = int(sd17["stem.0.weight"].shape[0]) if "stem.0.weight" in sd17 else 80
    emb17 = int(cfg.get("embed_dim", 64))
    net17 = KeyPrintNet(in_channels=in17, hidden_channels=hid17, embed_dim=emb17, dropout=0.18)
    net17.load_state_dict(sd17)
    net17.eval()

    net9 = None
    pt9_path = next((p for p in pt9_candidates if p.exists()), None)
    if pt9_path is not None:
        ck9 = torch.load(pt9_path, map_location="cpu", weights_only=True)
        sd9 = ck9["model_state_dict"] if isinstance(ck9, dict) and "model_state_dict" in ck9 else ck9
        in9 = int((cfg.get("password_model") or {}).get("in_channels", 9))
        hid9 = int(sd9["stem.0.weight"].shape[0]) if "stem.0.weight" in sd9 else 80
        net9 = KeyPrintNet(in_channels=in9, hidden_channels=hid9, embed_dim=emb17, dropout=0.18)
        net9.load_state_dict(sd9)
        net9.eval()

    bounds = {k: tuple(v) for k, v in cfg["outlier_bounds"].items()}
    mean = np.asarray(cfg["channel_mean"], dtype=np.float64)
    std = np.asarray(cfg["channel_std"], dtype=np.float64)
    impute_hold = float(cfg.get("touch_hold_impute_sec", 0.09))
    cohort_kb = np.asarray(cfg.get("sentence_cohort_centroids") or cfg["cohort_centroids"], dtype=np.float64)

    return {
        "cfg": cfg,
        "net17": net17,
        "net9": net9,
        "bounds": bounds,
        "mean": mean,
        "std": std,
        "impute_hold": impute_hold,
        "cohort_kb": cohort_kb,
        "torch": torch,
    }


def _embed_stroke_pt(raw_cap: Dict, bundle: Dict[str, object], *, use_9ch: bool = False) -> Optional[np.ndarray]:
    if not raw_cap or not raw_cap.get("dds"):
        return None
    stroke = dict(raw_cap)
    if stroke.get("pairKeys") is None:
        if not use_9ch:
            return None
        stroke["pairKeys"] = [["?", "?"] for _ in stroke["dds"]]

    tensor = build_input_tensor(stroke, bundle["bounds"], bundle["impute_hold"], bundle["mean"], bundle["std"])
    if tensor.shape[1] == 0:
        return None
    torch = bundle["torch"]
    with torch.no_grad():
        if use_9ch and bundle["net9"] is not None:
            x = torch.from_numpy(tensor[:9, :][None, :, :])
            emb = bundle["net9"](x).cpu().numpy()[0]
        else:
            x = torch.from_numpy(tensor[None, :, :])
            emb = bundle["net17"](x).cpu().numpy()[0]
    norm = float(np.linalg.norm(emb)) or 1e-8
    return emb.astype(np.float64) / norm


def analyze_hold_calibration(
    records: Sequence[Dict],
    enroll_idx: Dict[Tuple[str, str, str], Dict],
    repo_root: Path,
) -> Dict[str, object]:
    """Estimate within-subject physical switch hold shifts and re-score opt-in rawCaptures."""
    # 1. Collect per-(participant, keyboardType) observed hold measurements from enrollment & query captures
    enr_holds_by_subj_dev: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    all_holds_by_subj_dev: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for rec in records:
        env = rec.get("env") or rec.get("queryEnv") or {}
        pid = str(env.get("participantId") or "UNSET")
        kb = str(env.get("keyboardType") or "")
        if kb not in PHYSICAL_KB_GROUPS:
            continue
        if rec.get("eventType") == "enroll":
            raw_caps = rec.get("rawCaptures") or []
            vals = [_mean_valid_hold_sec(c) for c in raw_caps if c]
            vals = [v for v in vals if v is not None]
            if not vals and (rec.get("templateSummary") or {}).get("holdMs", 0) > 0:
                vals = [float(rec["templateSummary"]["holdMs"]) / 1000.0]
            enr_holds_by_subj_dev[(pid, kb)].extend(vals)
            all_holds_by_subj_dev[(pid, kb)].extend(vals)
        elif rec.get("eventType") == "query":
            opt = rec.get("optInDiagnostics") or {}
            raw_caps = opt.get("rawCaptures") or ([opt["rawCapture"]] if opt.get("rawCapture") else [])
            for c in raw_caps:
                if c:
                    v = _mean_valid_hold_sec(c)
                    if v is not None:
                        all_holds_by_subj_dev[(pid, kb)].append(v)

    by_participant_ms: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (pid, kb), vals in sorted(all_holds_by_subj_dev.items()):
        if vals:
            by_participant_ms[pid][kb] = round(float(np.mean(vals)) * 1000.0, 2)

    # Within-subject pairwise deltas (in ms) across devices typed by the same participant
    pairwise_deltas_ms: Dict[str, List[float]] = defaultdict(list)
    for pid, dev_map in by_participant_ms.items():
        devs = sorted(dev_map)
        for i in range(len(devs)):
            for j in range(i + 1, len(devs)):
                a, b = devs[i], devs[j]
                pairwise_deltas_ms[f"{b} - {a}"].append(dev_map[b] - dev_map[a])

    switch_bias_summary = {
        pair: {
            "n_participants": len(diffs),
            "mean_delta_ms": _safe_mean(diffs, 2),
            "std_delta_ms": round(float(np.std(diffs)), 2) if len(diffs) >= 2 else 0.0,
        }
        for pair, diffs in sorted(pairwise_deltas_ms.items())
    }

    # Global enrollment-only mean hold per keyboardType (used as non-leaky switch prior)
    dev_enr_mean_sec: Dict[str, float] = {}
    by_dev_pool: Dict[str, List[float]] = defaultdict(list)
    for (_, kb), vals in enr_holds_by_subj_dev.items():
        by_dev_pool[kb].extend(vals)
    for kb, vals in by_dev_pool.items():
        if vals:
            dev_enr_mean_sec[kb] = float(np.mean(vals))

    bundle = _load_frozen_encoders(repo_root)
    if bundle is None:
        return {
            "per_participant_mean_hold_ms": dict(by_participant_ms),
            "within_subject_switch_deltas_ms": switch_bias_summary,
            "rescoring": {"available": False, "reason": "frozen PyTorch checkpoints not found"},
        }

    # 2. Re-score physical cross-device Arena queries that have opt-in rawCaptures & enrolled templates
    cfg = bundle["cfg"]
    arena_thr = float(cfg.get("arena_open_set_threshold", 0.42))
    arena_margin = float(cfg.get("arena_margin", 0.03))
    beta_kb = float((cfg.get("behavior_profile") or {}).get("beta", {}).get("sentence_keyboard", 0.0))

    modes = ("uncalibrated", "naive_h_only", "physics_switch_prior", "physics_switch_plus_shrink")
    arena_eval_rows: Dict[str, List[Dict]] = {m: [] for m in modes}

    for q in records:
        if q.get("eventType") != "query" or q.get("domain") != "arena":
            continue
        q_env = q.get("queryEnv") or {}
        q_dev = str(q_env.get("keyboardType") or "")
        if q_dev not in PHYSICAL_KB_GROUPS:
            continue
        opt = q.get("optInDiagnostics") or {}
        raw_caps = [c for c in (opt.get("rawCaptures") or []) if c and c.get("pairKeys")]
        if not raw_caps:
            continue

        room = str(q.get("roomCode") or "LOCAL")
        dom_key = f"arena_{q.get('lang') or 'en'}"
        rankings = q.get("rankings") or []

        # Collect enrolled candidate templates from enroll records
        gallery: List[Dict] = []
        for r_cand in rankings:
            cid = str(r_cand.get("id") or "")
            enr = _lookup_enroll(enroll_idx, room, cid, dom_key)
            ts = (enr or {}).get("templateSummary") or {}
            if not ts.get("centroid"):
                continue
            c_vec = np.asarray(ts["centroid"], dtype=np.float64)
            c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-8)
            exs = (
                np.asarray(ts["exemplars"], dtype=np.float64)
                if ts.get("exemplars")
                else c_vec[None, :]
            )
            e_dev = str(((enr or {}).get("env") or {}).get("keyboardType") or q_dev)
            e_hold_sec = float(ts.get("holdMs") or 90.0) / 1000.0
            gallery.append(
                {
                    "id": cid,
                    "centroid": c_vec,
                    "exemplars": exs,
                    "enroll_dev": e_dev,
                    "enroll_hold_sec": e_hold_sec,
                    "behavior_llr": float(r_cand.get("behaviorLlr") or 0.0),
                }
            )
        if not gallery:
            continue

        gt = q.get("groundTruth") or {}
        role = gt.get("role")
        actual_id = gt.get("profileId")
        actual_cand = next((g for g in gallery if g["id"] == actual_id), None)
        is_cross = bool(actual_cand and actual_cand["enroll_dev"] != q_dev) or (
            role == "outsider" and any(g["enroll_dev"] != q_dev for g in gallery)
        )
        if not is_cross:
            continue

        for mode in modes:
            cand_scores: List[Tuple[float, str]] = []
            for cand in gallery:
                e_dev = cand["enroll_dev"]
                delta_prior = dev_enr_mean_sec.get(q_dev, 0.0) - dev_enr_mean_sec.get(e_dev, 0.0)
                embs = []
                for rc in raw_caps:
                    if mode == "uncalibrated":
                        cal_rc = rc
                    elif mode == "naive_h_only":
                        cal_rc = apply_hold_calibration(
                            rc,
                            delta_sec=delta_prior,
                            scale_lambda=0.5,
                            target_mean_hold_sec=cand["enroll_hold_sec"],
                            recompute_ud=False,
                        )
                    elif mode == "physics_switch_prior":
                        cal_rc = apply_hold_calibration(rc, delta_sec=delta_prior, recompute_ud=True)
                    else:
                        cal_rc = apply_hold_calibration(
                            rc,
                            delta_sec=delta_prior,
                            scale_lambda=0.5,
                            target_mean_hold_sec=cand["enroll_hold_sec"],
                            recompute_ud=True,
                        )
                    emb = _embed_stroke_pt(cal_rc, bundle, use_9ch=False)
                    if emb is not None:
                        embs.append(emb)
                if not embs:
                    continue
                q_emb = np.sum(embs, axis=0)
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                sim = as_norm_similarity(q_emb, cand, bundle["cohort_kb"], top_m=16)
                if beta_kb > 0:
                    sim = fuse(sim, cand["behavior_llr"], beta_kb)
                cand_scores.append((sim, cand["id"]))

            if not cand_scores:
                continue
            cand_scores.sort(key=lambda x: x[0], reverse=True)
            scores_desc = [s for s, _ in cand_scores]
            win_id = cand_scores[0][1]
            accepted = _open_set_accept(scores_desc, arena_thr, arena_margin)
            arena_eval_rows[mode].append(
                {
                    "role": role,
                    "top1_ok": (win_id == actual_id) if role == "enrolled" else None,
                    "accept_ok": (accepted and win_id == actual_id) if role == "enrolled" else None,
                    "outsider_fa": accepted if role == "outsider" else None,
                    "top1_score": scores_desc[0],
                }
            )

    arena_rescore_summary = {}
    for mode, rows in arena_eval_rows.items():
        enr_r = [r for r in rows if r["role"] == "enrolled"]
        out_r = [r for r in rows if r["role"] == "outsider"]
        arena_rescore_summary[mode] = {
            "n_cross_switch_queries": len(rows),
            "n_enrolled": len(enr_r),
            "n_outsider": len(out_r),
            "top1_acc": _safe_rate(sum(1 for r in enr_r if r["top1_ok"]), len(enr_r)),
            "genuine_accept_rate": _safe_rate(sum(1 for r in enr_r if r["accept_ok"]), len(enr_r)),
            "outsider_far": _safe_rate(sum(1 for r in out_r if r["outsider_fa"]), len(out_r)),
            "mean_top1_score": _safe_mean([r["top1_score"] for r in rows]),
        }

    # 3. Also re-score physical cross-device Password queries (public strings with pairKeys & 9ch+17ch templates)
    pw_eval_rows: Dict[str, List[Dict]] = {m: [] for m in modes}
    by_len_table = (cfg.get("password_dual") or {}).get("by_length") or []
    slope = float((cfg.get("password_dual") or {}).get("slope", 1.0))

    if bundle["net9"] is not None and by_len_table:
        for q in records:
            if q.get("eventType") != "query" or q.get("domain") != "password":
                continue
            q_env = q.get("queryEnv") or {}
            q_dev = str(q_env.get("keyboardType") or "")
            if q_dev not in PHYSICAL_KB_GROUPS:
                continue
            rc = (q.get("optInDiagnostics") or {}).get("rawCapture")
            if not rc or not rc.get("pairKeys"):
                continue
            room = str(q.get("roomCode") or "LOCAL")
            tgt = q.get("targetAccount") or {}
            enr = _lookup_enroll(enroll_idx, room, tgt.get("id"), "password")
            ts = (enr or {}).get("templateSummary") or {}
            if not ts.get("centroid") or not ts.get("centroid17"):
                continue
            e_dev = str(((enr or {}).get("env") or {}).get("keyboardType") or q_dev)
            if e_dev == q_dev:
                continue
            is_gen = (q.get("groundTruth") or {}).get("isGenuine")
            if is_gen is None:
                continue

            c9 = np.asarray(ts["centroid"], dtype=np.float64)
            c9 = c9 / (np.linalg.norm(c9) + 1e-8)
            ex9 = np.asarray(ts["exemplars"], dtype=np.float64) if ts.get("exemplars") else c9[None, :]
            c17 = np.asarray(ts["centroid17"], dtype=np.float64)
            c17 = c17 / (np.linalg.norm(c17) + 1e-8)
            ex17 = np.asarray(ts["exemplars17"], dtype=np.float64) if ts.get("exemplars17") else c17[None, :]
            tmpl9 = {"centroid": c9, "exemplars": ex9}
            tmpl17 = {"centroid": c17, "exemplars": ex17}

            pw_len = int((q.get("passwordMeta") or {}).get("length") or 12)
            prm = params_for_length(by_len_table, pw_len, "kb")
            cap_llr_val = float((q.get("verdict") or {}).get("capitalLlr") or 0.0)
            cm_cfg = (cfg.get("password_dual") or {}).get("capital_method") or {}
            cap_w = float(cm_cfg.get("weight", 0.0)) if cap_llr_val != 0.0 else 0.0
            tau_eff = prm["tau"] + (float(cm_cfg.get("tau_offset", 0.0)) if cap_llr_val != 0.0 else 0.0)

            e_hold_sec = float(ts.get("holdMs") or 90.0) / 1000.0
            delta_prior = dev_enr_mean_sec.get(q_dev, 0.0) - dev_enr_mean_sec.get(e_dev, 0.0)

            for mode in modes:
                if mode == "uncalibrated":
                    cal_rc = rc
                elif mode == "naive_h_only":
                    cal_rc = apply_hold_calibration(
                        rc,
                        delta_sec=delta_prior,
                        scale_lambda=0.5,
                        target_mean_hold_sec=e_hold_sec,
                        recompute_ud=False,
                    )
                elif mode == "physics_switch_prior":
                    cal_rc = apply_hold_calibration(rc, delta_sec=delta_prior, recompute_ud=True)
                else:
                    cal_rc = apply_hold_calibration(
                        rc,
                        delta_sec=delta_prior,
                        scale_lambda=0.5,
                        target_mean_hold_sec=e_hold_sec,
                        recompute_ud=True,
                    )
                e9 = _embed_stroke_pt(cal_rc, bundle, use_9ch=True)
                e17 = _embed_stroke_pt(cal_rc, bundle, use_9ch=False)
                if e9 is None or e17 is None:
                    continue
                r9 = raw_similarity(e9, tmpl9)
                r17 = raw_similarity(e17, tmpl17)
                z = dual_z(r9, r17, prm) + cap_w * cap_llr_val
                sim = 1.0 / (1.0 + math.exp(-slope * (z - tau_eff)))
                pw_eval_rows[mode].append({"label": 1 if is_gen else 0, "z": z, "accepted": sim >= 0.5})

    pw_rescore_summary = {}
    for mode, rows in pw_eval_rows.items():
        gen_r = [r for r in rows if r["label"] == 1]
        imp_r = [r for r in rows if r["label"] == 0]
        pw_rescore_summary[mode] = {
            "n_cross_switch_queries": len(rows),
            "n_genuine": len(gen_r),
            "n_impostor": len(imp_r),
            "genuine_accept_rate": _safe_rate(sum(1 for r in gen_r if r["accepted"]), len(gen_r)),
            "impostor_far": _safe_rate(sum(1 for r in imp_r if r["accepted"]), len(imp_r)),
            "mean_genuine_z": _safe_mean([r["z"] for r in gen_r]),
            "roc_auc": _roc_auc_safe([r["label"] for r in rows], [r["z"] for r in rows]),
        }

    return {
        "per_participant_mean_hold_ms": dict(by_participant_ms),
        "within_subject_switch_deltas_ms": switch_bias_summary,
        "rescoring": {
            "available": True,
            "arena_cross_switch": arena_rescore_summary,
            "password_cross_switch": pw_rescore_summary,
        },
    }


# ---------------------------------------------------------------------------------------
# Section 3: Smartphone Touch `beta` Sweep & `DeviceMotion` Verification
# ---------------------------------------------------------------------------------------
def _motion_feature_vec(dm: Optional[Dict]) -> Optional[np.ndarray]:
    if not dm or not dm.get("available"):
        return None
    vals = [
        dm.get("tapImpulseMean"),
        dm.get("linearAccelRms"),
        dm.get("gyroRms"),
        dm.get("pitchDeg"),
    ]
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        return None
    return np.asarray([float(v) for v in vals], dtype=np.float64)


def analyze_touch_and_motion(
    records: Sequence[Dict],
    enroll_idx: Dict[Tuple[str, str, str], Dict],
) -> Dict[str, object]:
    touch_arena = [
        r
        for r in records
        if r.get("eventType") == "query"
        and r.get("domain") == "arena"
        and (
            (r.get("queryEnv") or {}).get("detectedInputMode") == "touch"
            or (r.get("queryEnv") or {}).get("keyboardType") in TOUCH_KB_TYPES
        )
    ]

    beta_sweep: Dict[str, Dict[str, object]] = {}
    best_beta = 0.20
    best_accept = -1.0

    for beta in BETA_SWEEP_GRID:
        enr_top1, enr_acc, enr_n = 0, 0, 0
        out_fa, out_n = 0, 0
        for q in touch_arena:
            rankings = q.get("rankings") or []
            if not rankings:
                continue
            verdict = q.get("verdict") or {}
            thr = float(verdict.get("openSetThreshold") or 0.42)
            margin = float(verdict.get("marginThreshold") or 0.03)
            rescored = []
            for r_cand in rankings:
                emb_s = float(r_cand.get("embeddingSimilarity") or r_cand.get("similarity") or 0.5)
                b_llr = float(r_cand.get("behaviorLlr") or 0.0)
                s = fuse(emb_s, b_llr, beta) if beta > 0 else emb_s
                rescored.append((s, str(r_cand.get("id") or "")))
            rescored.sort(key=lambda x: x[0], reverse=True)
            scores_desc = [s for s, _ in rescored]
            win_id = rescored[0][1]
            acc = _open_set_accept(scores_desc, thr, margin)

            gt = q.get("groundTruth") or {}
            if gt.get("role") == "enrolled":
                enr_n += 1
                if win_id == gt.get("profileId"):
                    enr_top1 += 1
                    if acc:
                        enr_acc += 1
            elif gt.get("role") == "outsider":
                out_n += 1
                if acc:
                    out_fa += 1

        acc_rate = (enr_acc / enr_n) if enr_n > 0 else 0.0
        far_rate = (out_fa / out_n) if out_n > 0 else 0.0
        if enr_n > 0 and far_rate <= 0.05 and acc_rate > best_accept:
            best_accept = acc_rate
            best_beta = beta

        beta_sweep[f"{beta:.2f}"] = {
            "beta": beta,
            "n_enrolled": enr_n,
            "n_outsider": out_n,
            "top1_acc": _safe_rate(enr_top1, enr_n),
            "genuine_accept_rate": _safe_rate(enr_acc, enr_n),
            "outsider_far": _safe_rate(out_fa, out_n),
        }

    # DeviceMotion audit across all records with deviceMotion
    motion_records = [r for r in records if (r.get("deviceMotion") or {}).get("available") is True]
    hz_vals = [r["deviceMotion"].get("effectiveHz") for r in motion_records]
    dt_vals = [r["deviceMotion"].get("medianIntervalMs") for r in motion_records]
    tap_hand, tap_desk = [], []
    posture_agreement = []

    for r in motion_records:
        env = r.get("env") or r.get("queryEnv") or {}
        declared = str(env.get("posture") or "")
        dm = r["deviceMotion"]
        inferred = dm.get("inferredPosture")
        if declared in ("hand_two_thumbs", "hand_one_finger"):
            if dm.get("tapImpulseMean") is not None:
                tap_hand.append(float(dm["tapImpulseMean"]))
            if inferred in ("in_hand", "on_desk_flat"):
                posture_agreement.append(inferred == "in_hand")
        elif declared == "desk_phone_flat":
            if dm.get("tapImpulseMean") is not None:
                tap_desk.append(float(dm["tapImpulseMean"]))
            if inferred in ("in_hand", "on_desk_flat"):
                posture_agreement.append(inferred == "on_desk_flat")

    # Same-query Keystroke-Only vs Keystroke + DeviceMotion comparison on touch Arena queries
    # where both candidate enrollments and the query have valid DeviceMotion vectors.
    enr_motion_vecs: List[Tuple[str, np.ndarray]] = []
    for rec in records:
        if rec.get("eventType") == "enroll":
            mv = _motion_feature_vec(rec.get("deviceMotion"))
            if mv is not None and rec.get("profileId"):
                enr_motion_vecs.append((str(rec["profileId"]), mv))

    # Estimate diagonal standardized Gaussian scale across available motion enrollments/queries
    all_mvecs = [mv for _, mv in enr_motion_vecs] + [
        _motion_feature_vec(q.get("deviceMotion")) for q in touch_arena
    ]
    all_mvecs = [v for v in all_mvecs if v is not None]
    m_scale = np.maximum(np.std(all_mvecs, axis=0), np.asarray([0.05, 0.05, 1.0, 4.0])) if len(all_mvecs) >= 3 else np.asarray([0.25, 0.25, 6.0, 12.0])

    def _eval_sensor_fusion(queries_sub: Sequence[Dict], gamma_motion: float) -> Dict[str, object]:
        enr_top1, enr_acc, enr_n = 0, 0, 0
        out_fa, out_n = 0, 0
        evaluated = 0
        for q in queries_sub:
            q_mv = _motion_feature_vec(q.get("deviceMotion"))
            if q_mv is None:
                continue
            room = str(q.get("roomCode") or "LOCAL")
            dom_key = f"arena_{q.get('lang') or 'en'}"
            rankings = q.get("rankings") or []
            cand_mvs = {}
            for r_cand in rankings:
                cid = str(r_cand.get("id") or "")
                enr = _lookup_enroll(enroll_idx, room, cid, dom_key)
                mv = _motion_feature_vec((enr or {}).get("deviceMotion"))
                if mv is not None:
                    cand_mvs[cid] = mv
            if not cand_mvs:
                continue

            evaluated += 1
            verdict = q.get("verdict") or {}
            thr = float(verdict.get("openSetThreshold") or 0.42)
            margin = float(verdict.get("marginThreshold") or 0.03)
            scored = []
            for r_cand in rankings:
                cid = str(r_cand.get("id") or "")
                base_s = float(r_cand.get("similarity") or 0.5)
                if gamma_motion > 0 and cid in cand_mvs:
                    z_diff = (q_mv - cand_mvs[cid]) / m_scale
                    # Zero-mean Gaussian LLR: positive when closer than population unit variance (d^2 < D*0.75)
                    m_llr = max(-3.0, min(3.0, 0.5 * (3.0 - float(np.sum(z_diff ** 2)))))
                    s = fuse(base_s, m_llr, gamma_motion)
                else:
                    s = base_s
                scored.append((s, cid))
            scored.sort(key=lambda x: x[0], reverse=True)
            scores_desc = [s for s, _ in scored]
            win_id = scored[0][1]
            acc = _open_set_accept(scores_desc, thr, margin)

            gt = q.get("groundTruth") or {}
            if gt.get("role") == "enrolled":
                enr_n += 1
                if win_id == gt.get("profileId"):
                    enr_top1 += 1
                    if acc:
                        enr_acc += 1
            elif gt.get("role") == "outsider":
                out_n += 1
                if acc:
                    out_fa += 1

        return {
            "n_sensor_paired_queries": evaluated,
            "n_enrolled": enr_n,
            "n_outsider": out_n,
            "top1_acc": _safe_rate(enr_top1, enr_n),
            "genuine_accept_rate": _safe_rate(enr_acc, enr_n),
            "outsider_far": _safe_rate(out_fa, out_n),
        }

    in_hand_q = [
        q for q in touch_arena if str((q.get("queryEnv") or {}).get("posture") or "") in ("hand_two_thumbs", "hand_one_finger")
    ]
    on_desk_q = [q for q in touch_arena if str((q.get("queryEnv") or {}).get("posture") or "") == "desk_phone_flat"]

    return {
        "n_touch_arena_queries": len(touch_arena),
        "selected_beta_star": best_beta,
        "beta_sweep": beta_sweep,
        "device_motion_audit": {
            "n_records_with_motion": len(motion_records),
            "median_effective_hz": _safe_median(hz_vals, 1),
            "median_interval_ms": _safe_median(dt_vals, 2),
            "mean_tap_impulse_in_hand": _safe_mean(tap_hand, 4),
            "mean_tap_impulse_on_desk": _safe_mean(tap_desk, 4),
            "posture_inference_agreement": _safe_rate(sum(1 for ok in posture_agreement if ok), len(posture_agreement)),
        },
        "sensor_fusion_ab": {
            "all_touch": {
                "keystroke_only": _eval_sensor_fusion(touch_arena, 0.0),
                "keystroke_plus_motion": _eval_sensor_fusion(touch_arena, 0.12),
            },
            "in_hand": {
                "keystroke_only": _eval_sensor_fusion(in_hand_q, 0.0),
                "keystroke_plus_motion": _eval_sensor_fusion(in_hand_q, 0.12),
            },
            "on_desk_flat": {
                "keystroke_only": _eval_sensor_fusion(on_desk_q, 0.0),
                "keystroke_plus_motion": _eval_sensor_fusion(on_desk_q, 0.12),
            },
        },
    }


# ---------------------------------------------------------------------------------------
# Section 4: User-Chosen Password String-Specific Cohort Calibration
# ---------------------------------------------------------------------------------------
def _compare_pw_calibration(queries: Sequence[Dict]) -> Dict[str, object]:
    dual_q = [
        q
        for q in queries
        if (q.get("verdict") or {}).get("lengthBaseline") and (q.get("verdict") or {}).get("stringCalibrated")
    ]
    labeled = [q for q in dual_q if (q.get("groundTruth") or {}).get("isGenuine") is not None]
    gen = [q for q in labeled if q["groundTruth"]["isGenuine"] is True]
    imp = [q for q in labeled if q["groundTruth"]["isGenuine"] is False]

    y = [1 if q["groundTruth"]["isGenuine"] else 0 for q in labeled]
    z_len = [float(q["verdict"]["lengthBaseline"]["z"]) for q in labeled]
    z_str = [float(q["verdict"]["stringCalibrated"]["z"]) for q in labeled]

    # Per-string FAR dispersion (keyed by publicExperimentId or passwordHash12)
    imp_by_str_len: Dict[str, List[int]] = defaultdict(list)
    imp_by_str_cal: Dict[str, List[int]] = defaultdict(list)
    for q in imp:
        pm = q.get("passwordMeta") or {}
        s_key = str(pm.get("publicExperimentId") or pm.get("passwordHash12") or "unknown")
        imp_by_str_len[s_key].append(1 if q["verdict"]["lengthBaseline"].get("accepted") else 0)
        imp_by_str_cal[s_key].append(1 if q["verdict"]["stringCalibrated"].get("accepted") else 0)

    fars_len = [float(np.mean(v)) for v in imp_by_str_len.values() if len(v) >= 2]
    fars_cal = [float(np.mean(v)) for v in imp_by_str_cal.values() if len(v) >= 2]
    latencies = [
        float(q["verdict"]["stringCalibrated"]["latencyMs"])
        for q in dual_q
        if q["verdict"]["stringCalibrated"].get("latencyMs") is not None
    ]

    return {
        "n_dual_queries": len(dual_q),
        "n_genuine": len(gen),
        "n_impostor": len(imp),
        "length_baseline": {
            "genuine_accept_rate": _safe_rate(sum(1 for q in gen if q["verdict"]["lengthBaseline"].get("accepted")), len(gen)),
            "genuine_frr": _safe_rate(sum(1 for q in gen if not q["verdict"]["lengthBaseline"].get("accepted")), len(gen)),
            "impostor_far": _safe_rate(sum(1 for q in imp if q["verdict"]["lengthBaseline"].get("accepted")), len(imp)),
            "roc_auc": _roc_auc_safe(y, z_len),
            "frr_at_far5pct": _frr_at_far(y, z_len, 0.05),
            "frr_at_far10pct": _frr_at_far(y, z_len, 0.10),
            "per_string_far_std": round(float(np.std(fars_len)), 4) if len(fars_len) >= 2 else None,
        },
        "string_calibrated": {
            "genuine_accept_rate": _safe_rate(sum(1 for q in gen if q["verdict"]["stringCalibrated"].get("accepted")), len(gen)),
            "genuine_frr": _safe_rate(sum(1 for q in gen if not q["verdict"]["stringCalibrated"].get("accepted")), len(gen)),
            "impostor_far": _safe_rate(sum(1 for q in imp if q["verdict"]["stringCalibrated"].get("accepted")), len(imp)),
            "roc_auc": _roc_auc_safe(y, z_str),
            "frr_at_far5pct": _frr_at_far(y, z_str, 0.05),
            "frr_at_far10pct": _frr_at_far(y, z_str, 0.10),
            "per_string_far_std": round(float(np.std(fars_cal)), 4) if len(fars_cal) >= 2 else None,
        },
        "browser_calib_latency_ms": {
            "median": _safe_median(latencies, 4),
            "p95": round(float(np.quantile(latencies, 0.95)), 4) if latencies else None,
        },
    }


def analyze_password_string_calibration(records: Sequence[Dict]) -> Dict[str, object]:
    pw_q = [r for r in records if r.get("eventType") == "query" and r.get("domain") == "password"]
    touch_q = [
        q
        for q in pw_q
        if (q.get("queryEnv") or {}).get("detectedInputMode") == "touch"
        or (q.get("queryEnv") or {}).get("keyboardType") in TOUCH_KB_TYPES
    ]
    kb_q = [q for q in pw_q if q not in touch_q]
    len12_q = [q for q in pw_q if (q.get("passwordMeta") or {}).get("length") == 12]
    rep_q = [q for q in pw_q if (q.get("passwordMeta") or {}).get("publicCategory") == "repeat_heavy"]
    alt_q = [q for q in pw_q if (q.get("passwordMeta") or {}).get("publicCategory") == "alt_hand"]

    return {
        "overall": _compare_pw_calibration(pw_q),
        "keyboard": _compare_pw_calibration(kb_q),
        "touch": _compare_pw_calibration(touch_q),
        "length_12": _compare_pw_calibration(len12_q),
        "repeat_heavy": _compare_pw_calibration(rep_q),
        "alt_hand": _compare_pw_calibration(alt_q),
    }


def analyze_jsonl_files(paths: Sequence[Path], repo_root: Optional[Path] = None) -> Dict[str, object]:
    root = repo_root or Path(__file__).resolve().parent.parent
    merged = load_and_merge_jsonl(paths)
    records = merged["records"]
    enroll_idx = _index_enrollments(records)

    return {
        "meta": {
            "files_loaded": merged["files_loaded"],
            "total_records": merged["total_records"],
            "duplicates_skipped": merged["duplicates_skipped"],
            "invalid_lines": merged["invalid_lines"],
        },
        "environment_and_errors": analyze_environment_and_errors(records, enroll_idx),
        "hold_calibration": analyze_hold_calibration(records, enroll_idx, root),
        "touch_and_motion": analyze_touch_and_motion(records, enroll_idx),
        "password_string_calibration": analyze_password_string_calibration(records),
    }


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val * 100:.1f}%" if val is not None else "N/A"


def format_korean_summary(report: Dict[str, object]) -> str:
    env = report["environment_and_errors"]
    cnt = env["counts"]
    integ = env["protocol_integrity"]
    lines = [
        "=== KeyPrint Guard 다기기 실측 · 코호트 보정 분석 요약 ===",
        f"• 병합 레코드: 총 {cnt['total_records']}건 (등록 {cnt['enrollments']} / 문장 판별 {cnt['arena_queries']} / 암호 검증 {cnt['password_queries']})",
        f"• 참가자: {', '.join(cnt['participants']) or '없음'} | 관측 기기: {', '.join(cnt['devices']) or '없음'}",
        f"• 실험 통제 점검: 비공개 암호 마스킹={'정상' if integ['custom_password_redaction_ok'] else '위반!'} "
        f"| 평가 중 템플릿 자동적응 차단={'정상' if integ['template_lock_ok'] else '주의'}",
    ]

    a_all = env["arena"]["overall"]
    a_same = env["arena"]["same_device"]
    a_cross = env["arena"]["cross_device"]
    lines.append(
        f"• [문장 판별] 전체 1위={_fmt_pct(a_all['top1_acc'])}, 수락={_fmt_pct(a_all['genuine_accept_rate'])}, 외부인 오수락={_fmt_pct(a_all['outsider_far'])} "
        f"| 동일기기 수락={_fmt_pct(a_same['genuine_accept_rate'])} vs 교차기기 수락={_fmt_pct(a_cross['genuine_accept_rate'])}"
    )

    hc = report["hold_calibration"]
    if hc["within_subject_switch_deltas_ms"]:
        deltas_txt = ", ".join(
            f"{k}: {v['mean_delta_ms']:+.1f}ms (n={v['n_participants']})"
            for k, v in hc["within_subject_switch_deltas_ms"].items()
        )
        lines.append(f"• [물리 스위치 홀드 편차 Δh] {deltas_txt}")
    if hc["rescoring"].get("available"):
        ar = hc["rescoring"]["arena_cross_switch"]
        if ar["uncalibrated"]["n_cross_switch_queries"] > 0:
            lines.append(
                "• [교차 스위치 홀드 보정 재채점 (문장)] "
                f"보정전 수락={_fmt_pct(ar['uncalibrated']['genuine_accept_rate'])} → "
                f"단순 H보정={_fmt_pct(ar['naive_h_only']['genuine_accept_rate'])} → "
                f"물리일관(H+UD)={_fmt_pct(ar['physics_switch_plus_shrink']['genuine_accept_rate'])}"
            )

    tm = report["touch_and_motion"]
    if tm["n_touch_arena_queries"] > 0:
        b0 = tm["beta_sweep"]["0.00"]
        b02 = tm["beta_sweep"]["0.20"]
        lines.append(
            f"• [스마트폰 터치 β 비교 (n={tm['n_touch_arena_queries']})] "
            f"β=0 수락={_fmt_pct(b0['genuine_accept_rate'])} (FAR {_fmt_pct(b0['outsider_far'])}) | "
            f"β=0.20 수락={_fmt_pct(b02['genuine_accept_rate'])} (FAR {_fmt_pct(b02['outsider_far'])}) | "
            f"최적 β*={tm['selected_beta_star']:.2f}"
        )
    dm = tm["device_motion_audit"]
    if dm["n_records_with_motion"] > 0:
        lines.append(
            f"• [DeviceMotion 센서] 수집 {dm['n_records_with_motion']}건, 중앙값 {dm['median_effective_hz']}Hz "
            f"({dm['median_interval_ms']}ms), 자세 일치율={_fmt_pct(dm['posture_inference_agreement'])}"
        )

    pw = report["password_string_calibration"]["overall"]
    if pw["n_dual_queries"] > 0:
        lb = pw["length_baseline"]
        sc = pw["string_calibrated"]
        lines.append(
            f"• [암호 문자열별 코호트 보정 (n={pw['n_dual_queries']})] "
            f"길이보정 FRR@FAR10%={_fmt_pct(lb['frr_at_far10pct'])} (AUC {lb['roc_auc']}) vs "
            f"문자열보정 FRR@FAR10%={_fmt_pct(sc['frr_at_far10pct'])} (AUC {sc['roc_auc']}) | "
            f"브라우저 지연 중앙값={pw['browser_calib_latency_ms']['median']}ms"
        )

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze KeyPrint Guard multi-device .jsonl evaluation logs.")
    parser.add_argument("jsonl_files", nargs="+", type=Path, help="One or more exported .jsonl evaluation log files")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write full JSON report")
    args = parser.parse_args(argv)

    report = analyze_jsonl_files(args.jsonl_files)
    print(format_korean_summary(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n상세 리포트 저장 완료: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
