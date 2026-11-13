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
# KeyRecs raw-stream loading (press-ordered events with hold durations)
# ---------------------------------------------------------------------------------------
def load_keyrecs_sessions(free_csv_path) -> Dict[Tuple[str, int], List[Tuple[str, float, float]]]:
    """Return {(participant, session): [(key_label, t_down, hold), ...]} sorted by keydown."""
    import pandas as pd

    df = pd.read_csv(free_csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ["DU.key1.key1", "DD.key1.key2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["DU.key1.key1", "DD.key1.key2"])
    out = {}
    for (participant, session), grp in df.groupby(["participant", "session"]):
        keys = grp["key1"].astype(str).tolist()
        holds = grp["DU.key1.key1"].to_numpy(dtype=np.float64)
        dds = grp["DD.key1.key2"].to_numpy(dtype=np.float64)
        t = np.concatenate([[0.0], np.cumsum(dds[:-1])])
        order = np.argsort(t, kind="stable")
        out[(str(participant), int(session))] = [(keys[i], float(t[i]), float(holds[i])) for i in order]
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
