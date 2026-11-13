"""
Physical keyboard geometry shared by training (Python) and the browser (exported to
web/public/models/model_config.json -> `keyboard_layout`), so both sides compute the exact
same key-identity channels.

Every key is addressed by its *physical* KeyboardEvent.code (e.g. `KeyA`, `Digit5`, `Space`).
Physical codes are layout/IME independent, so English QWERTY and Korean 2-beolsik typing on
the same keyboard map to identical coordinates.

Coordinates use standard key units on an ANSI board: row 0 = number row ... row 4 = space bar,
column = horizontal key-centre offset including the row stagger.
Fingers follow touch-typing assignment: 0..3 left pinky->index, 5 thumb (space), 6..9 right
index->pinky. Hand = 0 (left) / 1 (right) / None (thumb or unknown).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# code -> (row, col, finger)
KEY_GEOMETRY: Dict[str, Tuple[float, float, int]] = {}


def _row(codes: Sequence[str], row: int, start_col: float, fingers: Sequence[int]) -> None:
    for i, (code, finger) in enumerate(zip(codes, fingers)):
        KEY_GEOMETRY[code] = (float(row), start_col + i, int(finger))


_row(
    ["Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6",
     "Digit7", "Digit8", "Digit9", "Digit0", "Minus", "Equal"],
    0, 0.0, [0, 0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9, 9],
)
_row(
    ["KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU", "KeyI", "KeyO", "KeyP",
     "BracketLeft", "BracketRight", "Backslash"],
    1, 1.5, [0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9, 9, 9],
)
_row(
    ["KeyA", "KeyS", "KeyD", "KeyF", "KeyG", "KeyH", "KeyJ", "KeyK", "KeyL",
     "Semicolon", "Quote"],
    2, 1.75, [0, 1, 2, 3, 3, 6, 6, 7, 8, 9, 9],
)
_row(
    ["KeyZ", "KeyX", "KeyC", "KeyV", "KeyB", "KeyN", "KeyM", "Comma", "Period", "Slash"],
    3, 2.25, [0, 1, 2, 3, 3, 6, 6, 7, 8, 9],
)
KEY_GEOMETRY["Enter"] = (2.0, 13.0, 9)
KEY_GEOMETRY["Space"] = (4.0, 6.0, 5)

ROW_SCALE = 4.0
COL_SCALE = 13.5
UNKNOWN_GEOMETRY = (2.0, 6.5, -1)  # neutral board centre, no finger

KEY_CHANNEL_NAMES: List[str] = [
    "Row_curr",     # 9:  physical row of key t (0 = number row .. 1 = space bar)
    "Col_curr",     # 10: physical column of key t (stagger-aware, normalised)
    "Row_next",     # 11: physical row of key t+1
    "Col_next",     # 12: physical column of key t+1
    "Key_Dist",     # 13: Euclidean travel distance between the two keys (key units)
    "Same_Hand",    # 14: 1 same hand / 0 hand alternation / 0.5 thumb or unknown key involved
    "Same_Finger",  # 15: 1 when the same finger must move to a different key
]

# --- character -> physical code (used for CMU/KeyRecs labels and soft/touch strokes) ---
CHAR_TO_CODE: Dict[str, str] = {" ": "Space"}
for _c in "abcdefghijklmnopqrstuvwxyz":
    CHAR_TO_CODE[_c] = f"Key{_c.upper()}"
    CHAR_TO_CODE[_c.upper()] = f"Key{_c.upper()}"
for _d in "0123456789":
    CHAR_TO_CODE[_d] = f"Digit{_d}"
for _plain, _shifted, _code in [
    ("`", "~", "Backquote"), ("-", "_", "Minus"), ("=", "+", "Equal"),
    ("[", "{", "BracketLeft"), ("]", "}", "BracketRight"), ("\\", "|", "Backslash"),
    (";", ":", "Semicolon"), ("'", '"', "Quote"), (",", "<", "Comma"),
    (".", ">", "Period"), ("/", "?", "Slash"),
]:
    CHAR_TO_CODE[_plain] = _code
    CHAR_TO_CODE[_shifted] = _code
for _sym, _digit in zip("!@#$%^&*()", "1234567890"):
    CHAR_TO_CODE[_sym] = f"Digit{_digit}"

# Korean 2-beolsik (두벌식 표준) single-keystroke jamo -> physical code
HANGUL_JAMO_TO_CODE: Dict[str, str] = {
    "ㅂ": "KeyQ", "ㅈ": "KeyW", "ㄷ": "KeyE", "ㄱ": "KeyR", "ㅅ": "KeyT",
    "ㅛ": "KeyY", "ㅕ": "KeyU", "ㅑ": "KeyI", "ㅐ": "KeyO", "ㅔ": "KeyP",
    "ㅁ": "KeyA", "ㄴ": "KeyS", "ㅇ": "KeyD", "ㄹ": "KeyF", "ㅎ": "KeyG",
    "ㅗ": "KeyH", "ㅓ": "KeyJ", "ㅏ": "KeyK", "ㅣ": "KeyL",
    "ㅋ": "KeyZ", "ㅌ": "KeyX", "ㅊ": "KeyC", "ㅍ": "KeyV", "ㅠ": "KeyB",
    "ㅜ": "KeyN", "ㅡ": "KeyM",
    # Shifted single keystrokes
    "ㅃ": "KeyQ", "ㅉ": "KeyW", "ㄸ": "KeyE", "ㄲ": "KeyR", "ㅆ": "KeyT",
    "ㅒ": "KeyO", "ㅖ": "KeyP",
}
CHAR_TO_CODE.update(HANGUL_JAMO_TO_CODE)

# KeyRecs / CMU column labels -> physical code
LABEL_TO_CODE: Dict[str, str] = {
    "period": "Period", "five": "Digit5", "Shift.r": "KeyR", "Return": "Enter",
    "Space": "Space", "Enter": "Enter",
}

CMU_PASSWORD_CODES: List[str] = [
    "Period", "KeyT", "KeyI", "KeyE", "Digit5", "KeyR", "KeyO", "KeyA", "KeyN", "KeyL", "Enter",
]
KEYRECS_FIXED_CODES: List[str] = [
    "KeyV", "KeyP", "KeyW", "KeyJ", "KeyK", "KeyE", "KeyU", "KeyR", "KeyK", "KeyB",
]


def code_for_label(label: str) -> Optional[str]:
    """Map a dataset key label or produced character to a physical code (None if unknown)."""
    if label in LABEL_TO_CODE:
        return LABEL_TO_CODE[label]
    if label in KEY_GEOMETRY:
        return label
    return CHAR_TO_CODE.get(label)


def geometry(code: Optional[str]) -> Tuple[float, float, int]:
    return KEY_GEOMETRY.get(code or "", UNKNOWN_GEOMETRY)


def hand_of(finger: int) -> Optional[int]:
    if 0 <= finger <= 3:
        return 0
    if 6 <= finger <= 9:
        return 1
    return None


def key_pair_features(code_a: Optional[str], code_b: Optional[str]) -> List[float]:
    """7 key-identity channels for one digraph transition (code_a -> code_b)."""
    ra, ca, fa = geometry(code_a)
    rb, cb, fb = geometry(code_b)
    dist = float(np.hypot(ra - rb, ca - cb))
    ha, hb = hand_of(fa), hand_of(fb)
    if ha is None or hb is None:
        same_hand = 0.5
    else:
        same_hand = 1.0 if ha == hb else 0.0
    same_finger = 1.0 if (fa >= 0 and fa == fb and code_a != code_b) else 0.0
    return [ra / ROW_SCALE, ca / COL_SCALE, rb / ROW_SCALE, cb / COL_SCALE, dist, same_hand, same_finger]


def key_channel_matrix(pair_codes: Sequence[Tuple[Optional[str], Optional[str]]]) -> np.ndarray:
    """(7, M) key-identity channel block for a list of M digraph code pairs."""
    if len(pair_codes) == 0:
        return np.zeros((len(KEY_CHANNEL_NAMES), 0), dtype=np.float32)
    return np.asarray([key_pair_features(a, b) for a, b in pair_codes], dtype=np.float32).T


def pairs_from_codes(codes: Sequence[Optional[str]]) -> List[Tuple[Optional[str], Optional[str]]]:
    return [(codes[i], codes[i + 1]) for i in range(len(codes) - 1)]


def export_layout_for_web() -> Dict:
    return {
        "row_scale": ROW_SCALE,
        "col_scale": COL_SCALE,
        "unknown": list(UNKNOWN_GEOMETRY),
        "keys": {code: [r, c, f] for code, (r, c, f) in KEY_GEOMETRY.items()},
        "char_to_code": dict(CHAR_TO_CODE),
    }
