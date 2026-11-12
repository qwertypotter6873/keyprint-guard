"""
Python mirror of the browser capture path (web/src/biometrics.js + web/src/features.js):
recorder stack with Backspace stitching -> clean stroke -> 17-channel tensor, the behavioural
event log, and the web's template scoring (centroid/exemplar mix + AS-Norm + behaviour fusion).

It is used to
  1) evaluate models exactly the way the web app consumes keystrokes ("web-faithful" captures
     replayed from KeyRecs raw streams, including real typos and Backspace repairs), and
  2) export a parity fixture so web/scripts/parity_check.mjs can assert that the JavaScript
     implementation produces the same strokes, tensors, behaviour statistics and scores.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from behavior_profile import BehaviorModel, Event, kind_of, summarize
from keyboard_layout import CHAR_TO_CODE, code_for_label, key_pair_features

LONG_PAUSE_SEC = 2.2
MIN_VALID_HOLD = 0.008
MAX_HOLD = 1.2
MIN_DD, MAX_DD = 0.006, 2.2


@dataclass
class Stroke:
    key: str                 # physical code, or 'char:<c>' when only the produced character is known
    down: float
    hold: Optional[float]    # None when the hold is unknown (touch) or implausibly short
    clean_from_prev: bool
    synthetic: bool = False


@dataclass
class RecorderSim:
    """Mirror of SmartKeystrokeRecorder. `touch=True` replays the input-event (soft keyboard)
    path: no physical codes, no keyup timing, no modifier keystrokes."""

    touch: bool = False
    stack: List[Stroke] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    just_edited: bool = False
    backspace_count: int = 0

    def reset(self) -> None:
        self.stack, self.events = [], []
        self.just_edited, self.backspace_count = False, 0

    def _clear_text_state(self) -> None:
        # Field erased back to empty: clean slate for rhythm, behaviour log is kept
        self.stack, self.just_edited, self.backspace_count = [], False, 0

    def feed(self, code: Optional[str], char: str, t: float, hold: Optional[float]) -> None:
        kind, side = kind_of(code or char)
        if kind == "other":
            return
        if kind in ("shift", "caps"):
            if not self.touch:
                self.events.append(Event(kind=kind, t=t, hold=_valid_hold(hold), side=side))
            return
        self.events.append(Event(kind=kind, t=t, hold=None if self.touch else _valid_hold(hold, keep_long=True)))
        if kind == "bs":
            self.backspace_count += 1
            self.just_edited = True
            if self.stack:
                self.stack.pop()
            if not self.stack and self.backspace_count > 0:
                self._clear_text_state()
            return
        paused = bool(self.stack) and (t - self.stack[-1].down) > LONG_PAUSE_SEC
        clean = (not self.stack) or (not self.just_edited and not paused)
        self.just_edited = False
        soft = self.touch or not code  # no physical code -> input-event stroke without keyup timing
        key = f"char:{char}" if soft else code
        self.stack.append(Stroke(key=key, down=t, hold=None if soft else _valid_hold(hold, keep_long=True), clean_from_prev=clean))

    def extract(self) -> Dict:
        return extract_clean_stroke(self.stack, self.events, self.backspace_count)


def _valid_hold(h: Optional[float], keep_long: bool = False) -> Optional[float]:
    if h is None or h < MIN_VALID_HOLD:
        return None
    if not keep_long and h > 5.0:
        return None
    return float(h)


def extract_clean_stroke(stack: Sequence[Stroke], events: Sequence[Event], backspace_count: int = 0) -> Dict:
    holds: List[Optional[float]] = []
    hold_valid: List[bool] = []
    dds: List[float] = []
    uds: List[Optional[float]] = []
    stitch: List[bool] = []
    pair_keys: List[List[str]] = []
    for i in range(len(stack) - 1):
        c0, c1 = stack[i], stack[i + 1]
        if not c1.clean_from_prev or c0.synthetic or c1.synthetic:
            continue
        h0, h1 = c0.hold, c1.hold
        if (h0 is not None and h0 > MAX_HOLD) or (h1 is not None and h1 > MAX_HOLD):
            continue
        dd = c1.down - c0.down
        if dd < MIN_DD or dd > MAX_DD:
            continue
        ud = (c1.down - (c0.down + h0)) if h0 is not None else None
        if not holds:
            holds.append(h0)
            hold_valid.append(h0 is not None)
        holds.append(h1)
        hold_valid.append(h1 is not None)
        dds.append(dd)
        uds.append(ud)
        stitch.append(i > 0 and not c0.clean_from_prev)
        pair_keys.append([c0.key, c1.key])
    return {
        "holds": holds,
        "hold_valid": hold_valid,
        "dds": dds,
        "uds": uds,
        "stitch_flags": stitch,
        "pair_keys": pair_keys,
        "clean_digraphs": len(dds),
        "backspace_count": backspace_count,
        "behavior": summarize(events),
        "input_mode": input_mode_of(hold_valid),
    }


def input_mode_of(hold_valid: Sequence[bool]) -> str:
    if not hold_valid:
        return "keyboard"
    frac = float(np.mean(hold_valid))
    if frac >= 0.8:
        return "keyboard"
    if frac <= 0.2:
        return "touch"
    return "mixed"


def resolve_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    if key.startswith("char:"):
        return CHAR_TO_CODE.get(key[5:])
    return key


def build_input_tensor(
    stroke: Dict,
    bounds: Dict[str, Tuple[float, float]],
    impute_hold: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Exact Python counterpart of features.js::buildInputTensor -> (17, M) scaled tensor."""
    eps = 1e-4
    clip = lambda v, b: min(b[1], max(b[0], v))  # noqa: E731
    h_imp = clip(impute_hold, bounds["H"])
    holds = [clip(h if h is not None else h_imp, bounds["H"]) for h in stroke["holds"]]
    dds = [clip(d, bounds["DD"]) for d in stroke["dds"]]
    uds = [clip(u if u is not None else dds[t] - h_imp, bounds["UD"]) for t, u in enumerate(stroke["uds"])]
    hv = stroke["hold_valid"]
    stitch = stroke.get("stitch_flags") or [False] * len(dds)
    m = len(dds)
    mean_h = max(eps, sum(holds) / len(holds))
    mean_dd = max(eps, sum(dds) / len(dds))
    out = np.zeros((len(mean), m), dtype=np.float64)
    for t in range(m):
        h_curr, h_next, dd, ud = holds[t], holds[t + 1], dds[t], uds[t]
        local_accel = 0.0 if (t == 0 or stitch[t]) else dd - dds[t - 1]
        a, b = stroke["pair_keys"][t]
        raw = [
            h_curr, h_next, dd, ud,
            h_curr / mean_h, dd / mean_dd,
            min(5.0, max(0.0, h_curr / max(dd, 0.02))),
            max(0.0, -ud), local_accel,
            *key_pair_features(resolve_key(a), resolve_key(b)),
            1.0 if (hv[t] and hv[t + 1]) else 0.0,
        ]
        out[:, t] = raw
    return ((out - mean[:, None]) / std[:, None]).astype(np.float32)


def simulate_captures(
    raw: Sequence[Tuple[str, float, float]],
    target_chars: int = 68,
    touch: bool = False,
    min_digraphs: int = 8,
) -> List[Dict]:
    """Replay a KeyRecs session as consecutive web captures of `target_chars` surviving chars."""
    rec = RecorderSim(touch=touch)
    caps = []
    for label, t, h in raw:
        kind, _ = kind_of(label)
        if kind == "other":
            continue
        code = "Backspace" if kind == "bs" else ("ShiftLeft" if kind == "shift" else ("CapsLock" if kind == "caps" else code_for_label(label)))
        char = " " if kind == "space" else (label if kind == "char" else "")
        rec.feed(code, char, t, h)
        if len(rec.stack) >= target_chars:
            cap = rec.extract()
            if cap["clean_digraphs"] >= min_digraphs:
                caps.append(cap)
            rec.reset()
    return caps


# ---------------------------------------------------------------------------------------
# Web scoring mirror (KeyPrintEngine.scoreTemplate / identifyTypist)
# ---------------------------------------------------------------------------------------
def make_template(embs: np.ndarray, behavior: Optional[Dict] = None) -> Dict:
    ex = np.asarray(embs[-12:], dtype=np.float64)
    c = ex.sum(axis=0)
    c = c / (np.linalg.norm(c) + 1e-8)
    return {"centroid": c, "exemplars": ex, "behavior": behavior}


def as_norm_similarity(emb: np.ndarray, tmpl: Dict, cohort: np.ndarray, top_m: int) -> float:
    c_sim = float(emb @ tmpl["centroid"])
    ex_sim = float(np.max(tmpl["exemplars"] @ emb))
    raw = 0.65 * c_sim + 0.35 * ex_sim
    k = min(top_m, len(cohort))
    u = np.sort(cohort @ tmpl["centroid"])[-k:]
    q = np.sort(cohort @ emb)[-k:]
    mu_u, sd_u = float(u.mean()), float(u.std()) + 0.05
    mu_q, sd_q = float(q.mean()), float(q.std()) + 0.05
    z = (raw - 0.5 * (mu_u + mu_q)) / (0.5 * (sd_u + sd_q))
    return 1.0 / (1.0 + math.exp(-0.85 * (z - 0.15)))


def fuse(similarity: float, behavior_llr: float, beta: float) -> float:
    s = min(1.0 - 1e-6, max(1e-6, similarity))
    logit = math.log(s / (1.0 - s)) + beta * behavior_llr
    return 1.0 / (1.0 + math.exp(-logit))


def identify(
    q_emb: np.ndarray,
    q_behavior: Optional[Dict],
    gallery: Sequence[Dict],
    cohort: np.ndarray,
    top_m: int,
    behavior_model: Optional[BehaviorModel],
    beta: float,
) -> np.ndarray:
    scores = []
    for tmpl in gallery:
        s = as_norm_similarity(q_emb, tmpl, cohort, top_m)
        if behavior_model is not None and beta > 0 and q_behavior and tmpl.get("behavior"):
            b, _ = behavior_model.llr(q_behavior, tmpl["behavior"])
            s = fuse(s, b, beta)
        scores.append(s)
    return np.asarray(scores)


# ---------------------------------------------------------------------------------------
# Parity fixture (Python expectations replayed by web/scripts/parity_check.mjs)
# ---------------------------------------------------------------------------------------
def js_event_script(raw: Sequence[Tuple[str, float, float]], touch: bool) -> List[Dict]:
    """DOM-level event script equivalent to a raw KeyRecs stream (times in ms)."""
    script = []
    for label, t, h in raw:
        kind, _ = kind_of(label)
        if kind == "other":
            continue
        t_ms = round(t * 1000.0, 3)
        up_ms = round((t + h) * 1000.0, 3)
        if kind in ("shift", "caps"):
            if touch:
                continue
            code = "ShiftLeft" if kind == "shift" else "CapsLock"
            script.append({"type": "keydown", "code": code, "key": "Shift" if kind == "shift" else "CapsLock", "t": t_ms})
            script.append({"type": "keyup", "code": code, "key": "Shift" if kind == "shift" else "CapsLock", "t": up_ms})
            continue
        if kind == "bs":
            if touch:
                script.append({"type": "keydown", "code": "", "key": "Unidentified", "t": t_ms})
            else:
                script.append({"type": "keydown", "code": "Backspace", "key": "Backspace", "t": t_ms})
            script.append({"type": "input", "op": "delete", "t": t_ms})
            if not touch:
                script.append({"type": "keyup", "code": "Backspace", "key": "Backspace", "t": up_ms})
            continue
        char = " " if kind == "space" else label
        code = code_for_label(label) or ""
        if touch:
            script.append({"type": "keydown", "code": "", "key": "Unidentified", "t": t_ms})
        else:
            script.append({"type": "keydown", "code": code, "key": char, "t": t_ms})
        script.append({"type": "input", "op": "insert", "data": char, "t": t_ms})
        if not touch:
            script.append({"type": "keyup", "code": code, "key": char, "t": up_ms})
    script.sort(key=lambda e: (e["t"], 0 if e["type"] == "keydown" else (1 if e["type"] == "input" else 2)))
    return script


def _jsonable_stroke(cap: Dict) -> Dict:
    return {k: cap[k] for k in ("holds", "hold_valid", "dds", "uds", "stitch_flags", "pair_keys", "clean_digraphs", "input_mode")}


def export_parity_fixture(
    out_path: Path,
    raw_segments: Sequence[Sequence[Tuple[str, float, float]]],
    bounds: Dict[str, Tuple[float, float]],
    impute_hold: float,
    mean: np.ndarray,
    std: np.ndarray,
    behavior_model: BehaviorModel,
    beta: float,
    dual_params: Optional[Dict[str, Dict[str, float]]] = None,
    display_slope: float = 1.0,
) -> None:
    cases = []
    for seg_idx, raw in enumerate(raw_segments):
        for touch in (False, True):
            rec = RecorderSim(touch=touch)
            for label, t, h in raw:
                kind, _ = kind_of(label)
                if kind == "other":
                    continue
                code = "Backspace" if kind == "bs" else ("ShiftLeft" if kind == "shift" else ("CapsLock" if kind == "caps" else code_for_label(label)))
                char = " " if kind == "space" else (label if kind == "char" else "")
                rec.feed(code, char, t, h)
            cap = rec.extract()
            # The JS clock is in ms; Python expectations use the identical ms-rounded times
            tensor = build_input_tensor(cap, bounds, impute_hold, mean, std) if cap["clean_digraphs"] > 0 else np.zeros((len(mean), 0))
            cases.append(
                {
                    "name": f"segment{seg_idx}_{'touch' if touch else 'keyboard'}",
                    "touch": touch,
                    "script": js_event_script(raw, touch),
                    "expected": {
                        "stroke": _jsonable_stroke(cap),
                        "tensor": np.round(tensor, 6).tolist(),
                        "behavior": cap["behavior"],
                    },
                }
            )
    # Cross-case LLR + fusion expectations
    llr_cases = []
    for i in range(0, len(cases) - 2, 2):
        a, b = cases[i]["expected"]["behavior"], cases[i + 2]["expected"]["behavior"]
        total, parts = behavior_model.llr(a, b)
        llr_cases.append({"query": i, "enroll": i + 2, "llr": total, "parts": parts, "fused_at_0_6": fuse(0.6, total, beta)})
    # Score-rule expectations (dual password score, multi-sentence query, open-set margin)
    rng = np.random.default_rng(0)
    unit = lambda n: (lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True))(rng.standard_normal((n, 64)))  # noqa: E731
    score_cases = []
    for mode, prm in (dual_params or {}).items():
        for _ in range(3):
            e9, e17 = unit(4), unit(4)
            base9, base17 = unit(1)[0], unit(1)[0]
            e9 = (e9 + 1.5 * base9) / np.linalg.norm(e9 + 1.5 * base9, axis=1, keepdims=True)
            e17 = (e17 + 1.5 * base17) / np.linalg.norm(e17 + 1.5 * base17, axis=1, keepdims=True)
            q9 = (unit(1)[0] + 1.2 * base9); q9 /= np.linalg.norm(q9)
            q17 = (unit(1)[0] + 1.2 * base17); q17 /= np.linalg.norm(q17)
            t9, t17 = make_template(e9), make_template(e17)
            r9 = 0.65 * float(q9 @ t9["centroid"]) + 0.35 * float(np.max(t9["exemplars"] @ q9))
            r17 = 0.65 * float(q17 @ t17["centroid"]) + 0.35 * float(np.max(t17["exemplars"] @ q17))
            z = (r9 - prm["mu9"]) / prm["sd9"] + (r17 - prm["mu17"]) / prm["sd17"]
            score_cases.append({
                "mode": mode, "prm": prm, "slope": display_slope,
                "q9": q9.tolist(), "q17": q17.tolist(),
                "t9": {"centroid": t9["centroid"].tolist(), "exemplars": t9["exemplars"].tolist()},
                "t17": {"centroid": t17["centroid"].tolist(), "exemplars": t17["exemplars"].tolist()},
                "z": z, "similarity": 1.0 / (1.0 + math.exp(-display_slope * (z - prm["tau"]))),
            })
    pair = unit(2)
    combined = pair.sum(0) / np.linalg.norm(pair.sum(0))
    open_set_cases = [
        {"scores": [0.91, 0.80, 0.4], "threshold": 0.85, "margin": 0.05, "accept": True},
        {"scores": [0.91, 0.88, 0.4], "threshold": 0.85, "margin": 0.05, "accept": False},
        {"scores": [0.84, 0.30], "threshold": 0.85, "margin": 0.0, "accept": False},
        {"scores": [0.86], "threshold": 0.85, "margin": 0.08, "accept": True},
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"beta": beta, "cases": cases, "llr_cases": llr_cases, "score_cases": score_cases,
                   "combine_case": {"inputs": pair.tolist(), "expected": combined.tolist()},
                   "open_set_cases": open_set_cases}, f, ensure_ascii=False)


def ms_rounded(raw: Sequence[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """Round a raw stream onto the millisecond grid the JS replay uses (t and t+hold in ms)."""
    out = []
    for label, t, h in raw:
        t_ms = round(t * 1000.0, 3)
        up_ms = round((t + h) * 1000.0, 3)
        out.append((label, t_ms / 1000.0, (up_ms - t_ms) / 1000.0))
    return out
