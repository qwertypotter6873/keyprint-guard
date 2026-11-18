"""
Lightweight string-specific cohort calibration for user-chosen passwords, plus physics-consistent
cross-device hold calibration helpers (mirrored in web/src/features.js).

Design invariants:
  1) Frozen encoders (`keyprint_net_9ch.pt` and `keyprint_net.pt`) are NEVER modified or retrained.
  2) Digraph summary statistics and string-residual Ridge weights are fitted strictly on `Train`
     participants (`kr["train"]`, 68 subjects) and `Train`-only strings.
  3) Shrinkage hyperparameters (`ridge_alpha`, `support_kappa`, `shrink_scale`) and the updated
     decision thresholds `tau(L)` are selected strictly on `Validation` (`kr["val"]`, 10 subjects)
     using disjoint `Validation` strings.
  4) Final evaluation runs on completely unseen `Test` participants (`kr["test"]`, 18 subjects) and
     disjoint `Test` strings, as well as non-synthesized real human recordings (`KeyRecs-fixed-text`,
     natural repeated free-text words, and `CMU` `.tie5Roanl`).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from keyboard_layout import (
    CHAR_TO_CODE,
    KEY_GEOMETRY,
    UNKNOWN_GEOMETRY,
    code_for_label,
    key_pair_features,
)
from password_synth import CALIB_LENGTHS, LETTERS, covered_disjoint_s1, params_for_length

# ---------------------------------------------------------------------------------------
# Hangul 2-beolsik decomposition (exact Python mirror of web/src/features.js::textToKeystrokes)
# ---------------------------------------------------------------------------------------
_CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
_JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_COMPOUND = {
    "ㅘ": ["ㅗ", "ㅏ"], "ㅙ": ["ㅗ", "ㅐ"], "ㅚ": ["ㅗ", "ㅣ"],
    "ㅝ": ["ㅜ", "ㅓ"], "ㅞ": ["ㅜ", "ㅔ"], "ㅟ": ["ㅜ", "ㅣ"],
    "ㅢ": ["ㅡ", "ㅣ"], "ㄳ": ["ㄱ", "ㅅ"], "ㄵ": ["ㄴ", "ㅈ"],
    "ㄶ": ["ㄴ", "ㅎ"], "ㄺ": ["ㄹ", "ㄱ"], "ㄻ": ["ㄹ", "ㅁ"],
    "ㄼ": ["ㄹ", "ㅂ"], "ㄽ": ["ㄹ", "ㅅ"], "ㄾ": ["ㄹ", "ㅌ"],
    "ㄿ": ["ㄹ", "ㅍ"], "ㅀ": ["ㄹ", "ㅎ"], "ㅄ": ["ㅂ", "ㅅ"],
}

STRING_FEATURE_NAMES: Tuple[str, ...] = (
    "repeat_frac",
    "same_finger_frac",
    "alt_hand_frac",
    "mean_dist",
    "dup_pair_frac",
    "mean_log_count",
    "mean_between_std",
)

# 24 public experiment strings stratified across lengths {12, 16, 20} and motor structures.
# `tier="core"` marks the 6 primary strings (2 per length) for fast cross-device human sessions;
# `tier="extended"` provides 18 additional strings for multi-participant string-calibration studies.
PUBLIC_EXPERIMENT_STRINGS: List[Dict[str, object]] = [
    # --- 12 chars (8 strings) ---
    {"id": "EXP12_ALT_A", "length": 12, "category": "alt_hand", "tier": "core", "text": "Gachon2026pw", "label": "12자 · 양손교차·혼합 (핵심)"},
    {"id": "EXP12_REP_A", "length": 12, "category": "repeat_heavy", "tier": "core", "text": "CoffeeTree88", "label": "12자 · 연속반복 다수 (핵심)"},
    {"id": "EXP12_ALT_B", "length": 12, "category": "alt_hand", "tier": "extended", "text": "planetworkbd", "label": "12자 · 양손교차 영문"},
    {"id": "EXP12_REP_B", "length": 12, "category": "repeat_heavy", "tier": "extended", "text": "summerbreeze", "label": "12자 · 복수 자모반복"},
    {"id": "EXP12_LEFT", "length": 12, "category": "one_hand", "tier": "extended", "text": "stewardesses", "label": "12자 · 왼손 편중형"},
    {"id": "EXP12_RIGHT", "length": 12, "category": "one_hand", "tier": "extended", "text": "hilopolimion", "label": "12자 · 오른손 편중형"},
    {"id": "EXP12_MIX_A", "length": 12, "category": "mixed_alnum", "tier": "extended", "text": "SilverKey402", "label": "12자 · 대소문·숫자 A"},
    {"id": "EXP12_MIX_B", "length": 12, "category": "mixed_alnum", "tier": "extended", "text": "Harbor97Gate", "label": "12자 · 대소문·숫자 B"},
    # --- 16 chars (8 strings) ---
    {"id": "EXP16_ALT_A", "length": 16, "category": "alt_hand", "tier": "core", "text": "KeyPrint2026Safe", "label": "16자 · 양손교차·혼합 (핵심)"},
    {"id": "EXP16_REP_A", "length": 16, "category": "repeat_heavy", "tier": "core", "text": "BookkeeperRoom77", "label": "16자 · 연속반복 다수 (핵심)"},
    {"id": "EXP16_ALT_B", "length": 16, "category": "alt_hand", "tier": "extended", "text": "formalsignalwork", "label": "16자 · 양손교차 영문"},
    {"id": "EXP16_REP_B", "length": 16, "category": "repeat_heavy", "tier": "extended", "text": "greenapplecoffee", "label": "16자 · 복수 자모반복"},
    {"id": "EXP16_LEFT", "length": 16, "category": "one_hand", "tier": "extended", "text": "sweaterdressedat", "label": "16자 · 왼손 편중형"},
    {"id": "EXP16_RIGHT", "length": 16, "category": "one_hand", "tier": "extended", "text": "minimumopinionlk", "label": "16자 · 오른손 편중형"},
    {"id": "EXP16_MIX_A", "length": 16, "category": "mixed_alnum", "tier": "extended", "text": "AutumnBridge2048", "label": "16자 · 대소문·숫자 A"},
    {"id": "EXP16_MIX_B", "length": 16, "category": "mixed_alnum", "tier": "extended", "text": "QuantumShield519", "label": "16자 · 대소문·숫자 B"},
    # --- 20 chars (8 strings) ---
    {"id": "EXP20_ALT_A", "length": 20, "category": "alt_hand", "tier": "core", "text": "Gachon Rhythm2026Key", "label": "20자 · 양손교차·혼합 (핵심)"},
    {"id": "EXP20_REP_A", "length": 20, "category": "repeat_heavy", "tier": "core", "text": "SweetCoffeeBalloon99", "label": "20자 · 연속반복 다수 (핵심)"},
    {"id": "EXP20_ALT_B", "length": 20, "category": "alt_hand", "tier": "extended", "text": "modernvisualworkland", "label": "20자 · 양손교차 영문"},
    {"id": "EXP20_REP_B", "length": 20, "category": "repeat_heavy", "tier": "extended", "text": "sleeplesswoodlandsee", "label": "20자 · 복수 자모반복"},
    {"id": "EXP20_LEFT", "length": 20, "category": "one_hand", "tier": "extended", "text": "afterwardsdressedvet", "label": "20자 · 왼손 편중형"},
    {"id": "EXP20_RIGHT", "length": 20, "category": "one_hand", "tier": "extended", "text": "millionopinionhookup", "label": "20자 · 오른손 편중형"},
    {"id": "EXP20_MIX_A", "length": 20, "category": "mixed_alnum", "tier": "extended", "text": "Morninglantern735Box", "label": "20자 · 대소문·숫자 A"},
    {"id": "EXP20_MIX_B", "length": 20, "category": "mixed_alnum", "tier": "extended", "text": "CrystalHorizon842Map", "label": "20자 · 대소문·숫자 B"},
]
assert all(len(str(item["text"])) == int(item["length"]) for item in PUBLIC_EXPERIMENT_STRINGS)


def text_to_keystrokes(text: str) -> List[str]:
    """Expand text into produced keystroke characters (exact mirror of features.js::textToKeystrokes)."""
    out: List[str] = []
    for ch in str(text or ""):
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3:
            idx = cp - 0xAC00
            cho = idx // 588
            jung = (idx % 588) // 28
            jong = idx % 28
            out.append(_CHO[cho])
            out.extend(_COMPOUND.get(_JUNG[jung], [_JUNG[jung]]))
            if jong:
                out.extend(_COMPOUND.get(_JONG[jong], [_JONG[jong]]))
        elif ch in _COMPOUND:
            out.extend(_COMPOUND[ch])
        else:
            out.append(ch)
    return out


def string_to_codes(text: str, char_to_code: Optional[Dict[str, str]] = None) -> List[Optional[str]]:
    mapping = CHAR_TO_CODE if char_to_code is None else char_to_code
    codes: List[Optional[str]] = []
    for ch in text_to_keystrokes(text):
        if ch == " ":
            codes.append("Space")
        else:
            codes.append(mapping.get(ch) or mapping.get(ch.lower()))
    return codes


# ---------------------------------------------------------------------------------------
# Physics-consistent Hold calibration (Python mirror of features.js::applyHoldCalibration)
# ---------------------------------------------------------------------------------------
def apply_hold_calibration(
    stroke: Dict,
    *,
    delta_sec: float = 0.0,
    scale_lambda: float = 0.0,
    target_mean_hold_sec: Optional[float] = None,
    recompute_ud: bool = True,
    min_hold_sec: float = 0.008,
    max_hold_sec: float = 1.2,
) -> Dict:
    """Adjust observed physical holds for cross-switch bias while keeping UD = DD - H_curr consistent.

    `delta_sec` represents `(query_switch_hold_bias - enroll_switch_hold_bias)` in seconds:
    subtracting `delta_sec` aligns the query hold back to the enrollment switch baseline.
    `scale_lambda` in [0, 1] optionally shrinks remaining query mean-hold gap toward `target_mean_hold_sec`.
    Touch/missing holds (`hold_valid == False`) are left untouched as `None`.
    """
    snake = "h_curr" in stroke
    u_holds = list(stroke["holds"])
    u_hv = list(stroke["hold_valid"] if "hold_valid" in stroke else stroke["holdValid"])
    h_curr = list(stroke["h_curr"] if snake else stroke["hCurr"])
    h_next = list(stroke["h_next"] if snake else stroke["hNext"])
    hv_curr = list(stroke["hv_curr"] if snake else stroke["hvCurr"])
    hv_next = list(stroke["hv_next"] if snake else stroke["hvNext"])
    dds = list(stroke["dds"])
    uds = list(stroke["uds"])

    valid_vals = [float(h) for h, ok in zip(u_holds, u_hv) if ok and h is not None]
    if not valid_vals:
        return dict(stroke)

    q_mean = sum(valid_vals) / len(valid_vals)
    shift = -float(delta_sec)
    if scale_lambda and target_mean_hold_sec is not None and target_mean_hold_sec > 0:
        shift += float(scale_lambda) * (float(target_mean_hold_sec) - (q_mean + shift))

    def _adj(val: Optional[float], ok: bool) -> Optional[float]:
        if not ok or val is None:
            return None
        return min(max_hold_sec, max(min_hold_sec, float(val) + shift))

    new_u_holds = [_adj(h, bool(ok)) for h, ok in zip(u_holds, u_hv)]
    new_h_curr = [_adj(h, bool(ok)) for h, ok in zip(h_curr, hv_curr)]
    new_h_next = [_adj(h, bool(ok)) for h, ok in zip(h_next, hv_next)]
    if recompute_ud:
        new_uds = [
            (float(dds[t]) - float(new_h_curr[t])) if (hv_curr[t] and new_h_curr[t] is not None and uds[t] is not None) else uds[t]
            for t in range(len(dds))
        ]
    else:
        new_uds = uds

    out = dict(stroke)
    out["holds"] = new_u_holds
    out["uds"] = new_uds
    if snake:
        out["h_curr"] = new_h_curr
        out["h_next"] = new_h_next
    else:
        out["hCurr"] = new_h_curr
        out["hNext"] = new_h_next
    return out


# ---------------------------------------------------------------------------------------
# Train-only digraph summary table & string feature extraction
# ---------------------------------------------------------------------------------------
def build_digraph_summary(
    tabs_train: Dict,
    train_people: Iterable[str],
    min_count: int = 5,
) -> Dict[str, List[float]]:
    """Build compact `{ 'CodeA>CodeB': [count, n_people, med_rel_dd, between_std] }` strictly from Train."""
    plist = sorted(set(train_people))
    person_med_dd: Dict[str, float] = {}
    for p in plist:
        all_dds = [
            occ.dd
            for s in (1, 2)
            if (p, s) in tabs_train
            for occs in tabs_train[(p, s)].values()
            for occ in occs
        ]
        if all_dds:
            person_med_dd[p] = float(np.median(all_dds))

    raw_rel: Dict[str, List[float]] = defaultdict(list)
    per_person_rel: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for p in plist:
        base = person_med_dd.get(p)
        if not base or base <= 0:
            continue
        for s in (1, 2):
            tab = tabs_train.get((p, s), {})
            for (ca, cb), occs in tab.items():
                ka, kb = code_for_label(ca), code_for_label(cb)
                if not ka or not kb:
                    continue
                pair_id = f"{ka}>{kb}"
                for occ in occs:
                    rel = float(occ.dd) / base
                    raw_rel[pair_id].append(rel)
                    per_person_rel[pair_id][p].append(rel)

    summary: Dict[str, List[float]] = {}
    for pair_id, vals in sorted(raw_rel.items()):
        cnt = len(vals)
        if cnt < min_count:
            continue
        p_medians = [float(np.median(v)) for v in per_person_rel[pair_id].values() if len(v) >= 2]
        n_ppl = len(per_person_rel[pair_id])
        med_rel = float(np.median(vals))
        b_std = float(np.std(p_medians)) if len(p_medians) >= 3 else 0.15
        summary[pair_id] = [
            int(cnt),
            int(n_ppl),
            round(med_rel, 5),
            round(b_std, 5),
        ]
    return summary


def extract_string_features(
    text: str,
    digraph_table: Dict[str, Sequence[float]],
    support_kappa: float = 40.0,
) -> Dict[str, object]:
    """Compute deterministic 7-D length-orthogonal string features and transition support weight."""
    codes = string_to_codes(text)
    k = len(codes)
    m = max(1, k - 1)
    if k < 2:
        return {
            "features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
            "support": 0.0,
            "known_pair_frac": 0.0,
            "num_transitions": 0,
        }

    repeats = 0.0
    same_finger = 0.0
    alt_hand = 0.0
    dists: List[float] = []
    seen_pairs = set()
    log_counts: List[float] = []
    b_stds: List[float] = []
    b_weights: List[float] = []
    supports: List[float] = []
    known_pairs = 0

    for t in range(m):
        ca, cb = codes[t], codes[t + 1]
        is_known = ca is not None and cb is not None
        if is_known:
            known_pairs += 1
        is_rep = 1.0 if (is_known and ca == cb) else 0.0
        repeats += is_rep
        kp = key_pair_features(ca, cb)
        dist, sh, sf = float(kp[4]), float(kp[5]), float(kp[6])
        same_finger += sf
        alt_hand += 1.0 if sh == 0.0 else 0.0
        dists.append(dist)
        pair_key = f"{ca or '?'}>{cb or '?'}"
        seen_pairs.add(pair_key)

        entry = digraph_table.get(pair_key) if is_known else None
        cnt = float(entry[0]) if entry is not None else 0.0
        b_std = float(entry[3]) if entry is not None else 0.15
        log_counts.append(math.log1p(cnt) / 8.0)
        sup = (cnt / (cnt + support_kappa)) if (is_known and cnt > 0) else 0.0
        supports.append(sup)
        b_stds.append(b_std)
        b_weights.append(0.2 + sup)

    w_sum = sum(b_weights) or 1.0
    mean_b_std = sum(v * w for v, w in zip(b_stds, b_weights)) / w_sum

    feats = [
        repeats / m,
        same_finger / m,
        alt_hand / m,
        sum(dists) / m,
        1.0 - (len(seen_pairs) / m),
        sum(log_counts) / m,
        mean_b_std,
    ]
    return {
        "features": feats,
        "support": sum(supports) / m,
        "known_pair_frac": known_pairs / m,
        "num_transitions": m,
    }


# ---------------------------------------------------------------------------------------
# Stratified disjoint string generator (including repeated letters & hand-bias strata!)
# ---------------------------------------------------------------------------------------
def _hand_of_letter(ch: str) -> Optional[int]:
    code = code_for_label(ch)
    geom = KEY_GEOMETRY.get(code or "", UNKNOWN_GEOMETRY)
    f = geom[2]
    if 0 <= f <= 3:
        return 0
    if 6 <= f <= 9:
        return 1
    return None


def generate_stratified_strings(
    tabs: Dict,
    people: Sequence[str],
    lengths: Sequence[int],
    per_length: int,
    rng: np.random.Generator,
    exclude: Iterable[str] = (),
    min_share: float = 0.65,
) -> Dict[int, List[Dict[str, str]]]:
    """Generate disjoint strings across 4 structural strata (repeat_heavy, one_hand, alt_hand, general)."""
    used = set(exclude)
    min_people = max(5, int(min_share * len(people)))

    pair_cov = {
        a + b: sum(covered_disjoint_s1(tabs, p, a + b) for p in people)
        for a in LETTERS
        for b in LETTERS
    }
    ok_general = {pr for pr, c in pair_cov.items() if c >= int(0.72 * len(people))}
    ok_repeat = [a for a in LETTERS if pair_cov[a + a] >= max(4, int(0.55 * len(people)))]

    def _eligible(w: str) -> bool:
        if w in used:
            return False
        return sum(covered_disjoint_s1(tabs, p, w) for p in people) >= min_people

    out: Dict[int, List[Dict[str, str]]] = {}
    strata = ("repeat_heavy", "one_hand", "alt_hand", "general")

    for L in lengths:
        bucket: List[Dict[str, str]] = []
        quotas = {
            "repeat_heavy": per_length // 4,
            "one_hand": per_length // 4,
            "alt_hand": per_length // 4,
            "general": per_length - 3 * (per_length // 4),
        }
        for stratum in strata:
            need = quotas[stratum]
            tries = 0
            got = 0
            while got < need and tries < 120000:
                tries += 1
                w = LETTERS[int(rng.integers(26))]
                while len(w) < L:
                    prev = w[-1]
                    if stratum == "repeat_heavy" and prev in ok_repeat and len(w) < L and rng.random() < 0.34:
                        # Avoid 3x identical in a row unless rare
                        if len(w) < 2 or w[-2] != prev:
                            w += prev
                            continue
                    cands = []
                    h_prev = _hand_of_letter(prev)
                    for b in LETTERS:
                        if (prev + b) not in ok_general:
                            continue
                        if stratum == "general" and b == prev and rng.random() < 0.7:
                            continue
                        h_b = _hand_of_letter(b)
                        if stratum == "one_hand" and h_prev is not None and h_b != h_prev and rng.random() < 0.85:
                            continue
                        if stratum == "alt_hand" and (b == prev or (h_prev is not None and h_b == h_prev and rng.random() < 0.88)):
                            continue
                        cands.append(b)
                    if not cands:
                        break
                    w += cands[int(rng.integers(len(cands)))]
                if len(w) != L:
                    continue
                if stratum == "repeat_heavy" and sum(a == b for a, b in zip(w, w[1:])) < max(2, L // 6):
                    continue
                if _eligible(w):
                    used.add(w)
                    bucket.append({"word": w, "stratum": stratum})
                    got += 1
        out[int(L)] = bucket
    return out


# ---------------------------------------------------------------------------------------
# Ridge residual regressor + transition-support shrinkage + Validation tau re-calibration
# ---------------------------------------------------------------------------------------
def per_string_impostor_stats(rows: Sequence[Dict], mode: str) -> Dict[str, Dict[str, float]]:
    """Aggregate empirical impostor (mu9, sd9, mu17, sd17) per string for `mode`."""
    by_word: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r["mode"] == mode and r["label"] == 0:
            by_word[r["word"]].append((float(r["raw9"]), float(r["raw17"])))
    stats: Dict[str, Dict[str, float]] = {}
    for w, pairs in by_word.items():
        if len(pairs) < 8:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        stats[w] = {
            "length": len(w),
            "n_imp": len(pairs),
            "mu9": float(arr[:, 0].mean()),
            "sd9": max(0.04, float(arr[:, 0].std())),
            "mu17": float(arr[:, 1].mean()),
            "sd17": max(0.04, float(arr[:, 1].std())),
        }
    return stats


def fit_ridge_residual_model(
    train_rows: Sequence[Dict],
    by_length_table: List[Dict],
    digraph_table: Dict[str, Sequence[float]],
    ridge_alpha: float = 8.0,
    support_kappa: float = 40.0,
) -> Dict[str, Dict]:
    """Fit linear Ridge models for residuals (d_mu9, d_log_sd9, d_mu17, d_log_sd17) on Train strings."""
    models: Dict[str, Dict] = {}
    for mode in ("kb", "touch"):
        mode_key = "touch" if mode == "touch" else "keyboard"
        emp = per_string_impostor_stats(train_rows, mode)
        words = sorted(emp)
        X_list, Y_list, W_sample = [], [], []
        for w in words:
            st = emp[w]
            base = params_for_length(by_length_table, st["length"], mode)
            fx = extract_string_features(w, digraph_table, support_kappa=support_kappa)
            X_list.append(fx["features"])
            Y_list.append([
                st["mu9"] - base["mu9"],
                math.log(max(0.04, st["sd9"]) / base["sd9"]),
                st["mu17"] - base["mu17"],
                math.log(max(0.04, st["sd17"]) / base["sd17"]),
            ])
            W_sample.append(math.sqrt(st["n_imp"]))

        X = np.asarray(X_list, dtype=np.float64)
        Y = np.asarray(Y_list, dtype=np.float64)
        sw = np.asarray(W_sample, dtype=np.float64)
        sw = sw / (sw.mean() or 1.0)

        x_mean = X.mean(axis=0)
        x_std = np.maximum(X.std(axis=0), 1e-6)
        Xz = (X - x_mean) / x_std
        # Weighted Ridge with unpenalized intercept
        Xa = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
        W_diag = np.diag(sw)
        reg = np.eye(Xa.shape[1]) * float(ridge_alpha)
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(Xa.T @ W_diag @ Xa + reg, Xa.T @ W_diag @ Y)  # (D+1, 4)

        models[mode_key] = {
            "feature_mean": [round(float(v), 6) for v in x_mean],
            "feature_std": [round(float(v), 6) for v in x_std],
            "weights": [[round(float(v), 6) for v in row] for row in coef[:-1].T],  # (4, D)
            "bias": [round(float(v), 6) for v in coef[-1]],  # (4,)
            "num_train_strings": len(words),
        }
    return models


def predict_string_dual_params(
    text: str,
    mode: str,
    by_length_table: List[Dict],
    calib_model: Dict,
) -> Dict[str, float]:
    """Predict `(mu9, sd9, mu17, sd17, tau)` for `text` with support-based shrinkage toward length baseline.
    Exact Python counterpart of web/src/features.js::stringCalibratedDualParams.
    """
    mode_key = "touch" if mode == "touch" else "keyboard"
    codes = string_to_codes(text)
    length = len(codes) if codes else len(text or "")
    base = params_for_length(by_length_table, length, mode)
    tau_offset = float(calib_model.get("tau_offset", {}).get(mode_key, 0.0))
    tau_base = base["tau"] + tau_offset

    m_cfg = calib_model.get("models", {}).get(mode_key)
    if not m_cfg:
        return {**base, "tau": tau_base, "shrink_weight": 0.0, "support": 0.0}

    kappa = float(calib_model.get("support_kappa", 40.0))
    shrink_scale = float(calib_model.get("shrink_scale", {}).get(mode_key, 1.0))
    fx = extract_string_features(text, calib_model["digraph_table"], support_kappa=kappa)
    feats = fx["features"]
    f_mean = m_cfg["feature_mean"]
    f_std = m_cfg["feature_std"]
    z_feat = [(feats[i] - f_mean[i]) / f_std[i] for i in range(len(feats))]

    W = m_cfg["weights"]  # 4 x D
    b = m_cfg["bias"]     # 4
    deltas = [
        sum(W[o][i] * z_feat[i] for i in range(len(z_feat))) + b[o]
        for o in range(4)
    ]

    # Clamp raw deltas so extreme out-of-distribution strings can never blow up variance
    d_mu9 = max(-0.22, min(0.22, deltas[0]))
    d_lsd9 = max(-0.45, min(0.45, deltas[1]))
    d_mu17 = max(-0.25, min(0.25, deltas[2]))
    d_lsd17 = max(-0.45, min(0.45, deltas[3]))

    w = max(0.0, min(1.0, shrink_scale * float(fx["support"])))
    mu9 = base["mu9"] + w * d_mu9
    sd9 = max(0.05, base["sd9"] * math.exp(w * d_lsd9))
    mu17 = base["mu17"] + w * d_mu17
    sd17 = max(0.05, base["sd17"] * math.exp(w * d_lsd17))

    return {
        "mu9": mu9,
        "sd9": sd9,
        "mu17": mu17,
        "sd17": sd17,
        "tau": tau_base,
        "shrink_weight": w,
        "support": float(fx["support"]),
    }


def build_calibration_parity_cases(model_config: Dict, calib_bundle: Dict) -> Dict[str, object]:
    """Build deterministic Python expectations for web/scripts/parity_check.mjs."""
    from capture_sim import build_input_tensor

    by_length = model_config["password_dual"]["by_length"]
    digraph_table = calib_bundle["digraph_table"]
    kappa = float(calib_bundle.get("support_kappa", 40.0))

    probe_strings = [str(item["text"]) for item in PUBLIC_EXPERIMENT_STRINGS] + [
        "가천값2026!Key",
        "안녕 세상아 1234",
        "aaaaBBBBcccc12",
        "z",
    ]
    string_cases = []
    for text in probe_strings:
        fx = extract_string_features(text, digraph_table, support_kappa=kappa)
        prm_kb = predict_string_dual_params(text, "kb", by_length, calib_bundle)
        prm_tc = predict_string_dual_params(text, "touch", by_length, calib_bundle)
        string_cases.append(
            {
                "text": text,
                "codes": string_to_codes(text),
                "features": [round(float(v), 8) for v in fx["features"]],
                "support": round(float(fx["support"]), 8),
                "known_pair_frac": round(float(fx["known_pair_frac"]), 8),
                "num_transitions": int(fx["num_transitions"]),
                "dual_kb": {k: round(float(v), 8) for k, v in prm_kb.items()},
                "dual_touch": {k: round(float(v), 8) for k, v in prm_tc.items()},
            }
        )

    base_stroke = {
        "holds": [0.115, 0.095, None, 0.130],
        "hold_valid": [True, True, False, True],
        "h_curr": [0.115, 0.095, None],
        "h_next": [0.095, None, 0.130],
        "hv_curr": [True, True, False],
        "hv_next": [True, False, True],
        "dds": [0.180, 0.210, 0.240],
        "uds": [0.065, 0.115, None],
        "stitch_flags": [False, False, True],
        "pair_keys": [["KeyG", "KeyA"], ["KeyA", "KeyC"], ["KeyC", "KeyH"]],
    }
    bounds = {k: tuple(v) for k, v in model_config["outlier_bounds"].items()}
    mean = np.asarray(model_config["channel_mean"], dtype=np.float64)
    std = np.asarray(model_config["channel_std"], dtype=np.float64)
    impute_hold = float(model_config.get("touch_hold_impute_sec", 0.09))

    hold_specs = [
        {"name": "physics_plus25ms", "delta_sec": 0.025, "scale_lambda": 0.0, "target_mean_hold_sec": None, "recompute_ud": True},
        {"name": "naive_plus25ms_no_ud", "delta_sec": 0.025, "scale_lambda": 0.0, "target_mean_hold_sec": None, "recompute_ud": False},
        {"name": "physics_shrink_to_85ms", "delta_sec": 0.020, "scale_lambda": 0.5, "target_mean_hold_sec": 0.085, "recompute_ud": True},
    ]
    hold_cases = []
    for sp in hold_specs:
        cal = apply_hold_calibration(
            base_stroke,
            delta_sec=sp["delta_sec"],
            scale_lambda=sp["scale_lambda"],
            target_mean_hold_sec=sp["target_mean_hold_sec"],
            recompute_ud=sp["recompute_ud"],
        )
        ten = build_input_tensor(cal, bounds, impute_hold, mean, std)
        hold_cases.append(
            {
                "spec": sp,
                "input_stroke": base_stroke,
                "calibrated_stroke": cal,
                "tensor": np.round(ten, 6).tolist(),
            }
        )

    return {
        "schema": "kp-calibration-parity-v1",
        "string_cases": string_cases,
        "hold_cases": hold_cases,
    }
