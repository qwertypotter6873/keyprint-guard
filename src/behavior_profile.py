"""
Behavioural typing profile: typo-correction habits, modifier strategy and pause rhythm.

Each feature is defined as the mean of per-event values, so a profile is just
(sum, count) per feature: enrollment profiles merge by adding sums/counts, and the amount of
evidence behind every feature is explicit.

Scoring uses a per-feature two-covariance model (PLDA-style, features independent):
    x_query - x_enroll = d
    same person      d ~ N(0, v0),  v0 = sigma2 * (1/n_q + 1/n_e) + 2 * delta2
    different person d ~ N(0, v1),  v1 = v0 + 2 * tau2
    LLR_j = 0.5 * ln(v1 / v0) - 0.5 * d^2 * (1/v0 - 1/v1)
sigma2 = per-event (within-capture) variance, delta2 = session drift, tau2 = between-person
variance, all fitted on Train participants only. With few observed events v0 is large and the
LLR collapses to ~0, so short inputs (passwords) automatically get almost no behavioural weight,
while long inputs with many corrections get more — no per-mode rules.

The browser counterpart lives in web/src/features.js and must stay numerically identical
(verified by web/scripts/parity_check.mjs against a fixture exported by the pipeline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# (name, group, scored). Unscored features are collected/displayed but never enter the score
# because no training data exists to fit their parameters (KeyRecs does not log Shift side).
FEATURES: List[Tuple[str, str, bool]] = [
    ("bs_rate", "correction", True),
    ("bs_burst_len", "correction", True),
    ("bs_single_frac", "correction", True),
    ("bs_wipe_frac", "correction", True),
    ("bs_detect_lat", "correction", True),
    ("bs_resume_lat", "correction", True),
    ("bs_repeat_ivl", "correction", True),
    ("bs_hold", "correction", True),
    ("pre_err_var", "correction", True),
    ("caps_ratio", "modifier", True),
    ("shift_lead", "modifier", True),
    ("shift_hold", "modifier", True),
    ("shift_right_ratio", "modifier", False),
    ("pause_500", "rhythm", True),
    ("pause_1000", "rhythm", True),
    ("space_after_rel", "rhythm", True),
    ("space_before_rel", "rhythm", True),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
LLR_CLIP = 4.0


@dataclass
class Event:
    kind: str                 # 'char' | 'space' | 'bs' | 'shift' | 'caps' | 'other'
    t: float                  # keydown time (s)
    hold: Optional[float]     # hold duration (s) or None when unknown (touch input)
    side: Optional[str] = None  # 'L' / 'R' for Shift when known
    rep: bool = False         # OS auto-repeat


def kind_of(label: str) -> Tuple[str, Optional[str]]:
    """Canonical event kind for a KeyboardEvent.code or KeyRecs key label."""
    if label in ("Backspace", "Delete"):
        return "bs", None
    if label in ("ShiftLeft", "ShiftRight", "Shift"):
        side = "L" if label == "ShiftLeft" else ("R" if label == "ShiftRight" else None)
        return "shift", side
    if label == "CapsLock":
        return "caps", None
    if label == "Space" or label == " ":
        return "space", None
    if len(label) == 1 or label.startswith("Key") or label.startswith("Digit") or label in (
        "Period", "Comma", "Slash", "Semicolon", "Quote", "BracketLeft", "BracketRight",
        "Minus", "Equal", "Backquote", "Backslash",
    ):
        return "char", None
    return "other", None


def _typing(e: Event) -> bool:
    return e.kind in ("char", "space", "bs")


def _textual(e: Event) -> bool:
    return e.kind in ("char", "space")


def event_values(events: Sequence[Event]) -> Dict[str, List[float]]:
    """Per-event values of every behavioural feature (definitions mirrored in features.js)."""
    ev = sorted(events, key=lambda e: e.t)
    vals: Dict[str, List[float]] = {n: [] for n in FEATURE_NAMES}
    presses = [e for e in ev if _typing(e)]

    # Tempo = median char->char press interval (normaliser for latency features)
    cc = []
    for a, b in zip(presses, presses[1:]):
        if _textual(a) and _textual(b):
            d = b.t - a.t
            if 0.008 < d < 2.0:
                cc.append(d)
    tempo = float(np.median(cc)) if len(cc) >= 3 else None
    log_cc = np.log(np.asarray(cc)) if len(cc) >= 4 else None
    base_std = float(np.std(log_cc)) if log_cc is not None else 0.0

    for p in presses:
        vals["bs_rate"].append(1.0 if p.kind == "bs" else 0.0)

    i = 0
    n = len(presses)
    while i < n:
        if presses[i].kind != "bs":
            i += 1
            continue
        j = i
        while j + 1 < n and presses[j + 1].kind == "bs":
            j += 1
        length = j - i + 1
        vals["bs_burst_len"].append(math.log(length))
        vals["bs_single_frac"].append(1.0 if length == 1 else 0.0)
        vals["bs_wipe_frac"].append(1.0 if length >= 4 else 0.0)
        if tempo is not None and i > 0 and _textual(presses[i - 1]):
            d = presses[i].t - presses[i - 1].t
            if 0.0 < d < 5.0:
                vals["bs_detect_lat"].append(math.log(d / tempo))
        if tempo is not None and j + 1 < n and _textual(presses[j + 1]):
            d = presses[j + 1].t - presses[j].t
            if 0.0 < d < 5.0:
                vals["bs_resume_lat"].append(math.log(d / tempo))
        for q in range(i, j):
            a, b = presses[q], presses[q + 1]
            if not a.rep and not b.rep:
                d = b.t - a.t
                if 0.02 < d < 2.0:
                    vals["bs_repeat_ivl"].append(math.log(d))
        if base_std > 1e-6 and i >= 4 and all(_textual(presses[k]) for k in range(i - 4, i)):
            ivl = np.diff([presses[k].t for k in range(i - 4, i)])
            if np.all(ivl > 0.008):
                vals["pre_err_var"].append(float(np.std(np.log(ivl))) / base_std)
        i = j + 1

    for e in ev:
        if e.kind == "bs" and not e.rep and e.hold is not None and 0.008 < e.hold < 1.5:
            vals["bs_hold"].append(math.log(e.hold))

    for idx, e in enumerate(ev):
        if e.rep or e.kind not in ("shift", "caps"):
            continue
        vals["caps_ratio"].append(1.0 if e.kind == "caps" else 0.0)
        if e.kind != "shift":
            continue
        if e.hold is not None and 0.008 < e.hold < 2.0:
            vals["shift_hold"].append(math.log(e.hold))
        if e.side is not None:
            vals["shift_right_ratio"].append(1.0 if e.side == "R" else 0.0)
        nxt = next((x for x in ev[idx + 1 :] if _typing(x)), None)
        if nxt is not None and nxt.kind == "char":
            d = nxt.t - e.t
            if 0.0 < d < 2.0:
                vals["shift_lead"].append(math.log(d))

    for a, b in zip(presses, presses[1:]):
        d = b.t - a.t
        if d > 0:
            vals["pause_500"].append(1.0 if d > 0.5 else 0.0)
            vals["pause_1000"].append(1.0 if d > 1.0 else 0.0)
        if tempo is not None and 0.008 < d < 2.0:
            if a.kind == "char" and b.kind == "space":
                vals["space_after_rel"].append(math.log(d / tempo))
            elif a.kind == "space" and b.kind == "char":
                vals["space_before_rel"].append(math.log(d / tempo))
    return vals


def summarize(events: Sequence[Event]) -> Dict[str, List[float]]:
    """Profile statistics {feature: [sum, count]}."""
    return {k: [float(sum(v)), float(len(v))] for k, v in event_values(events).items()}


def merge(a: Optional[Dict[str, List[float]]], b: Dict[str, List[float]]) -> Dict[str, List[float]]:
    if not a:
        return {k: list(v) for k, v in b.items()}
    out = {}
    for k in FEATURE_NAMES:
        sa, na = a.get(k, [0.0, 0.0])
        sb, nb = b.get(k, [0.0, 0.0])
        out[k] = [sa + sb, na + nb]
    return out


@dataclass
class BehaviorModel:
    params: Dict[str, Dict[str, float]]  # name -> {sigma2, tau2, delta2}

    def llr(self, query: Dict[str, List[float]], enroll: Dict[str, List[float]]) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        parts: Dict[str, float] = {}
        for name, _, scored in FEATURES:
            if not scored or name not in self.params:
                continue
            sq, nq = query.get(name, [0.0, 0.0])
            se, ne = enroll.get(name, [0.0, 0.0])
            if nq < 1 or ne < 1:
                continue
            p = self.params[name]
            v0 = p["sigma2"] * (1.0 / nq + 1.0 / ne) + 2.0 * p["delta2"]
            v1 = v0 + 2.0 * p["tau2"]
            d = sq / nq - se / ne
            val = 0.5 * math.log(v1 / v0) - 0.5 * d * d * (1.0 / v0 - 1.0 / v1)
            val = max(-LLR_CLIP, min(LLR_CLIP, val))
            parts[name] = val
            total += val
        return total, parts

    def to_json(self) -> Dict:
        return {
            "features": [
                {"name": n, "group": g, "scored": s, **({k: round(v, 8) for k, v in self.params[n].items()} if n in self.params else {})}
                for n, g, s in FEATURES
            ],
            "llr_clip": LLR_CLIP,
        }


def windows_of(events: Sequence[Event], size: int) -> List[List[Event]]:
    """Split a session into consecutive windows containing `size` typing presses each."""
    out, cur, count = [], [], 0
    for e in sorted(events, key=lambda x: x.t):
        cur.append(e)
        if _typing(e):
            count += 1
            if count == size:
                out.append(cur)
                cur, count = [], 0
    return out


def fit_behavior_model(
    sessions: Dict[Tuple[str, int], Sequence[Event]],
    participants: Iterable[str],
    window: int = 70,
) -> BehaviorModel:
    """Fit sigma2 / delta2 / tau2 per feature on Train participants (two sessions each)."""
    params: Dict[str, Dict[str, float]] = {}
    plist = [p for p in participants if (p, 1) in sessions and (p, 2) in sessions]
    sess_stats = {key: summarize(sessions[key]) for p in plist for key in ((p, 1), (p, 2))}
    win_stats = {
        key: [summarize(w) for w in windows_of(sessions[key], window)]
        for p in plist
        for key in ((p, 1), (p, 2))
    }

    for name, _, scored in FEATURES:
        if not scored:
            continue
        # sigma2: within-session spread of window means at the capture scale
        sig_terms = []
        for key, wins in win_stats.items():
            s_sum, s_n = sess_stats[key][name]
            if s_n < 2:
                continue
            m_s = s_sum / s_n
            for w in wins:
                w_sum, w_n = w[name]
                if w_n < 1 or w_n >= s_n:
                    continue
                factor = 1.0 / w_n - 1.0 / s_n
                sig_terms.append((w_sum / w_n - m_s) ** 2 / factor)
        if len(sig_terms) < 10:
            continue
        sigma2 = float(np.mean(sig_terms))

        d_terms, means, noise_terms = [], [], []
        for p in plist:
            s1, n1 = sess_stats[(p, 1)][name]
            s2, n2 = sess_stats[(p, 2)][name]
            if n1 < 1 or n2 < 1:
                continue
            m1, m2 = s1 / n1, s2 / n2
            d_terms.append((m1 - m2) ** 2 - sigma2 * (1.0 / n1 + 1.0 / n2))
            means.append(0.5 * (m1 + m2))
            noise_terms.append(sigma2 / 4.0 * (1.0 / n1 + 1.0 / n2))
        if len(means) < 10:
            continue
        delta2 = max(0.0, float(np.mean(d_terms)) / 2.0)
        tau2 = float(np.var(means, ddof=1)) - float(np.mean(noise_terms)) - delta2 / 2.0
        tau2 = max(1e-3 * sigma2, tau2)
        params[name] = {"sigma2": sigma2, "tau2": tau2, "delta2": max(delta2, 1e-4 * sigma2)}
    return BehaviorModel(params=params)


# ---------------------------------------------------------------------------------------
# KeyRecs raw-stream loading (lossless line-by-line parser + press-ordered events)
# ---------------------------------------------------------------------------------------
SESSION_BREAK_GAP_SEC = 100.0


def parse_keyrecs_free_csv(
    free_csv_path,
) -> Tuple[Dict[Tuple[str, int], List[Tuple[str, str, float, float]]], Dict[str, int]]:
    """
    Lossless line-by-line parser for KeyRecs-free-text.csv.
    - Protects quoted comma tokens (\",\") while preserving unescaped literal '\"' keystrokes (26 rows).
    - Recovers all 562,398 transition rows AND all 198 session-end trailing strokes (k2 == \"\" and dd == \"\").
    - Emits an explicit (\"break\", \"\", 0.0, 0.0) discontinuity marker whenever any corrupt/unparseable
      row occurs mid-session so adjacent valid rows are never bridged across a dropped row.
    """
    from collections import defaultdict

    rows_by_ps: Dict[Tuple[str, int], List[Tuple[str, str, float, float]]] = defaultdict(list)
    active_ps: Optional[Tuple[str, int]] = None
    prev_k2_by_ps: Dict[Tuple[str, int], Optional[str]] = {}
    stats = {
        "total_lines": 0,
        "valid_transitions": 0,
        "session_end_strokes": 0,
        "recovered_quote_rows": 0,
        "corrupt_rows": 0,
    }

    with open(free_csv_path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            stats["total_lines"] += 1

            head_parts = line.split(",", 2)
            if len(head_parts) < 3:
                stats["corrupt_rows"] += 1
                if active_ps is not None:
                    rows_by_ps[active_ps].append(("break", "", 0.0, 0.0))
                    prev_k2_by_ps[active_ps] = None
                continue

            p_raw, s_raw, rem = head_parts
            p_clean = p_raw.strip()
            try:
                sess_int = int(s_raw.strip())
                if not p_clean:
                    raise ValueError("empty participant")
                ps = (p_clean, sess_int)
                active_ps = ps
            except ValueError:
                stats["corrupt_rows"] += 1
                if active_ps is not None:
                    rows_by_ps[active_ps].append(("break", "", 0.0, 0.0))
                    prev_k2_by_ps[active_ps] = None
                continue

            # Strip optional trailing empty 10th column if present
            if rem.endswith(","):
                cand = rem[:-1]
                if cand.count(",") >= 6:
                    rem = cand

            tail_parts = rem.rsplit(",", 5)
            if len(tail_parts) != 6:
                stats["corrupt_rows"] += 1
                rows_by_ps[ps].append(("break", "", 0.0, 0.0))
                prev_k2_by_ps[ps] = None
                continue

            keys_part, du1_raw, dd_raw, _du12_raw, _ud_raw, _uu_raw = tail_parts
            if keys_part == '","","':
                k1, k2 = ",", ","
            elif keys_part == '",","':
                # Unescaped `"` adjacent to quoted `","` produces identical 5-char `",","`;
                # disambiguate losslessly using the preceding transition's key2 in this session.
                if prev_k2_by_ps.get(ps) == '"':
                    k1, k2 = '"', ","
                else:
                    k1, k2 = ",", '"'
            elif keys_part.startswith('",",'):
                k1, k2 = ",", keys_part[4:]
            elif keys_part.endswith(',","'):
                k1, k2 = keys_part[:-4], ","
            else:
                kp = keys_part.split(",")
                if len(kp) != 2:
                    stats["corrupt_rows"] += 1
                    rows_by_ps[ps].append(("break", "", 0.0, 0.0))
                    prev_k2_by_ps[ps] = None
                    continue
                k1, k2 = kp

            try:
                h1 = float(du1_raw)
                if not (math.isfinite(h1) and h1 >= 0.0) or not k1:
                    raise ValueError("invalid hold or empty key1")
            except ValueError:
                stats["corrupt_rows"] += 1
                rows_by_ps[ps].append(("break", "", 0.0, 0.0))
                prev_k2_by_ps[ps] = None
                continue

            if k2 == "" and dd_raw == "":
                if not (h1 > 0.0):
                    stats["corrupt_rows"] += 1
                    rows_by_ps[ps].append(("break", "", 0.0, 0.0))
                    prev_k2_by_ps[ps] = None
                    continue
                stats["session_end_strokes"] += 1
                if k1 == '"':
                    stats["recovered_quote_rows"] += 1
                rows_by_ps[ps].append(("end", k1, h1, 0.0))
                prev_k2_by_ps[ps] = None
            else:
                if not k2 or dd_raw == "":
                    stats["corrupt_rows"] += 1
                    rows_by_ps[ps].append(("break", "", 0.0, 0.0))
                    prev_k2_by_ps[ps] = None
                    continue
                try:
                    dd = float(dd_raw)
                    if not math.isfinite(dd):
                        raise ValueError("non-finite dd")
                except ValueError:
                    stats["corrupt_rows"] += 1
                    rows_by_ps[ps].append(("break", "", 0.0, 0.0))
                    prev_k2_by_ps[ps] = None
                    continue
                stats["valid_transitions"] += 1
                if k1 == '"' or k2 == '"':
                    stats["recovered_quote_rows"] += 1
                rows_by_ps[ps].append(("trans", k1, h1, dd))
                prev_k2_by_ps[ps] = k2

    stats["raw_participants"] = len({p for p, _ in rows_by_ps})
    stats["raw_sessions"] = len(rows_by_ps)
    return dict(rows_by_ps), stats


def load_keyrecs_sessions(free_csv_path) -> Dict[Tuple[str, int], List[Tuple[str, float, float]]]:
    """Return {(participant, session): [(key_label, t_down, hold), ...]} sorted by keydown.
    Includes session-end trailing strokes at t[N] = sum(dds[:N]) and inserts a >=100s gap across
    any mid-session corrupt-row discontinuity so RecorderSim never bridges across a dropped row."""
    rows_by_ps, _ = parse_keyrecs_free_csv(free_csv_path)
    out: Dict[Tuple[str, int], List[Tuple[str, float, float]]] = {}

    for ps, items in rows_by_ps.items():
        segments: List[List[Tuple[str, str, float, float]]] = []
        cur_seg: List[Tuple[str, str, float, float]] = []
        for item in items:
            if item[0] == "break":
                if cur_seg:
                    segments.append(cur_seg)
                    cur_seg = []
                elif segments:
                    # Keep break awareness if consecutive corrupt rows occur
                    pass
                continue
            cur_seg.append(item)
            if item[0] == "end":
                segments.append(cur_seg)
                cur_seg = []
        if cur_seg:
            segments.append(cur_seg)

        reconstructed: List[Tuple[str, float, float]] = []
        prev_max_t: Optional[float] = None
        had_leading_break = bool(items and items[0][0] == "break")

        for seg_idx, seg in enumerate(segments):
            keys: List[str] = []
            holds: List[float] = []
            ts: List[float] = []
            t_acc = 0.0
            for kind, k1, h1, dd in seg:
                keys.append(k1)
                holds.append(h1)
                ts.append(t_acc)
                if kind == "trans":
                    t_acc += dd
            t_arr = np.asarray(ts, dtype=np.float64)
            if prev_max_t is not None:
                offset = prev_max_t + SESSION_BREAK_GAP_SEC - float(np.min(t_arr))
                t_arr = t_arr + offset
            elif had_leading_break and seg_idx == 0:
                t_arr = t_arr + SESSION_BREAK_GAP_SEC
            order = np.argsort(t_arr, kind="stable")
            for i in order:
                reconstructed.append((keys[i], float(t_arr[i]), float(holds[i])))
            prev_max_t = float(np.max(t_arr))

        if reconstructed:
            out[ps] = reconstructed
    return out


def to_events(raw: Sequence[Tuple[str, float, Optional[float]]], touch: bool = False) -> List[Event]:
    """Convert (label, t_down, hold) tuples to Events. `touch=True` simulates a phone keyboard:
    no hold durations and no modifier keystrokes (the soft keyboard handles Shift itself)."""
    evs = []
    for label, t, h in raw:
        kind, side = kind_of(label)
        if kind == "other":
            continue
        if touch and kind in ("shift", "caps"):
            continue
        hold = None if (touch or h is None or not (0.0 < h < 5.0)) else float(h)
        evs.append(Event(kind=kind, t=float(t), hold=hold, side=side))
    return evs
