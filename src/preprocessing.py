"""
Data loading and 4-stage preprocessing pipeline for KeyPrint Guard & Arena.
Datasets:
1. CMU Keystroke Dynamics Benchmark Dataset (Killourhy & Maxion, DSN 2009)
   - 51 subjects x 8 sessions x 50 reps = 20,400 fixed-password strokes (".tie5Roanl", L=10)
2. KeyRecs Multi-Session Keystroke Dataset (Zenodo DOI: 10.5281/zenodo.7886742, CC-BY 4.0)
   - 98 two-session subjects typing fixed strings (19,772 strokes, L=10) and
     English typing-practice passages (562,583 raw keystrokes -> clean typo-free sentence windows, L=24)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from behavior_profile import parse_keyrecs_free_csv
from keyboard_layout import (
    CMU_PASSWORD_CODES,
    KEY_CHANNEL_NAMES,
    KEYRECS_FIXED_CODES,
    code_for_label,
    key_channel_matrix,
    pairs_from_codes,
)

SEED = 42

# 11 keys typed for ".tie5Roanl" + Return -> 10 consecutive digraph transitions
KEY_NAMES: List[str] = [
    "period",
    "t",
    "i",
    "e",
    "five",
    "Shift.r",
    "o",
    "a",
    "n",
    "l",
    "Return",
]

DIGRAPH_LABELS: List[str] = [
    ".→t",
    "t→i",
    "i→e",
    "e→5",
    "5→R",
    "R→o",
    "o→a",
    "a→n",
    "n→l",
    "l→Enter",
]

TIMING_CHANNEL_NAMES: List[str] = [
    "H_curr",        # 0: Hold duration of key t (s)
    "H_next",        # 1: Hold duration of key t+1 (s)
    "DD",            # 2: Press-to-Press latency keydown[t+1] - keydown[t] (s)
    "UD",            # 3: Flight latency keydown[t+1] - keyup[t] (s, <0 means rollover)
    "Rel_H",         # 4: Tempo-invariant relative hold H_t / mean(H)
    "Rel_DD",        # 5: Tempo-invariant relative digraph latency DD_t / mean(DD)
    "Duty_Ratio",    # 6: Articulation overlap ratio H_t / max(DD_t, 0.02)
    "Rollover_Mag",  # 7: Simultaneous key overlap max(0, -UD_t) (s)
    "Local_Accel",   # 8: Local rhythm acceleration DD_t - DD_{t-1} (s)
]

# 9 timing channels + 7 physical key-identity channels + 1 hold-validity flag = 17 channels.
# The first 9 channels are bit-identical to the legacy 9-channel model input.
CHANNEL_NAMES: List[str] = TIMING_CHANNEL_NAMES + KEY_CHANNEL_NAMES + [
    "Hold_Valid",    # 16: 1 = real keydown/keyup hold measured, 0 = hold unknown (touch keyboard)
]
N_TIMING_CHANNELS = len(TIMING_CHANNEL_NAMES)
N_CHANNELS = len(CHANNEL_NAMES)
HOLD_VALID_IDX = N_CHANNELS - 1
UNSCALED_CHANNELS = (HOLD_VALID_IDX,)  # binary flag fed to the network as-is
TOUCH_PROB = 0.18  # fraction of training samples converted to touch-keyboard (hold-less) inputs

CMU_KEY_BLOCK = key_channel_matrix(pairs_from_codes(CMU_PASSWORD_CODES))          # (7, 10)
KEYRECS_FIXED_KEY_BLOCK = key_channel_matrix(pairs_from_codes(KEYRECS_FIXED_CODES))  # (7, 9)

KEYRECS_HOLD_COLS: List[str] = [
    "DU.v.v", "DU.p.p", "DU.w.w", "DU.j.j", "DU.k.k",
    "DU.e.e", "DU.u.u", "DU.r.r", "DU.k.k.1", "DU.b.b",
]
KEYRECS_DD_COLS: List[str] = [
    "DD.v.p", "DD.p.w", "DD.w.j", "DD.j.k", "DD.k.e",
    "DD.e.u", "DD.u.r", "DD.r.k", "DD.k.b",
]


@dataclass
class SplitData:
    df: pd.DataFrame
    raw_flat: np.ndarray        # (N, F) clipped flat timing vector (for statistical baseline)
    seq_unscaled: np.ndarray    # (N, 17, M) unscaled 17-channel sequence
    seq_scaled: np.ndarray      # (N, 17, M) Z-score standardized 17-channel sequence
    subject_ids: np.ndarray     # (N,) integer label within split
    subjects: List[str]         # Unique subject codes in this split
    sessions: np.ndarray        # (N,) session index
    reps: np.ndarray            # (N,) repetition / chunk index
    key_block: np.ndarray = None  # (7, M) key-identity channels (identical for every sample)
    raw_holds: np.ndarray = None  # (N, M+1) unclipped raw hold durations in seconds
    raw_dds: np.ndarray = None    # (N, M) unclipped raw press-to-press latencies in seconds
    raw_uds: np.ndarray = None    # (N, M) unclipped raw flight latencies in seconds


@dataclass
class SentenceSplitData:
    holds: np.ndarray           # (N, M+1) clipped hold durations in seconds
    dds: np.ndarray             # (N, M) clipped press-to-press latencies in seconds
    uds: np.ndarray             # (N, M) clipped flight latencies in seconds
    raw_flat: np.ndarray        # (N, 3*M+1) flat vector for statistical baseline
    seq_unscaled: np.ndarray    # (N, 17, M)
    seq_scaled: np.ndarray      # (N, 17, M)
    subject_ids: np.ndarray     # (N,) local subject index 0..len(subjects)-1
    subjects: List[str]         # Unique participant IDs
    sessions: np.ndarray        # (N,) session 1 or 2
    chunk_ids: np.ndarray       # (N,) sequential chunk index within session
    texts: List[str]            # (N,) actual typed English substring
    key_blocks: np.ndarray = None  # (N, 7, M) key-identity channels per chunk
    codes: List[List[str]] = None  # (N, M+1) physical key codes per chunk
    raw_holds: np.ndarray = None   # (N, M+1) unclipped raw hold durations in seconds
    raw_dds: np.ndarray = None     # (N, M) unclipped raw press-to-press latencies in seconds
    raw_uds: np.ndarray = None     # (N, M) unclipped raw flight latencies in seconds
    t_ranges: np.ndarray = None    # (N, 2) [t_first_key, t_last_key] within session
    key_idx_ranges: np.ndarray = None  # (N, 2) [first_row_idx, last_row_idx] within session (strictly disjoint)


def get_feature_columns() -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return ordered CMU column names (all_31, hold_11, dd_10, ud_10)."""
    hold_cols = [f"H.{k}" for k in KEY_NAMES]
    dd_cols = [f"DD.{KEY_NAMES[i]}.{KEY_NAMES[i + 1]}" for i in range(len(KEY_NAMES) - 1)]
    ud_cols = [f"UD.{KEY_NAMES[i]}.{KEY_NAMES[i + 1]}" for i in range(len(KEY_NAMES) - 1)]

    all_cols: List[str] = []
    for i in range(len(KEY_NAMES) - 1):
        all_cols.extend([hold_cols[i], dd_cols[i], ud_cols[i]])
    all_cols.append(hold_cols[-1])
    return all_cols, hold_cols, dd_cols, ud_cols


def split_subjects_disjoint(
    subjects: List[str],
    n_train: int = 35,
    n_val: int = 6,
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """Subject-Disjoint Split (zero identity leakage across Train / Val / Test)."""
    rng = np.random.RandomState(seed)
    ordered = sorted(subjects)
    rng.shuffle(ordered)
    train_subjs = sorted(ordered[:n_train])
    val_subjs = sorted(ordered[n_train : n_train + n_val])
    test_subjs = sorted(ordered[n_train + n_val :])
    return train_subjs, val_subjs, test_subjs


def fit_outlier_bounds(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Preprocessing Step 1: Fit physiological outlier clipping bounds strictly on Train split.
    Eliminates multi-second distraction pauses (up to 25.99s in raw data) while preserving
    genuine negative rollover flight times (down to -0.25s).
    """
    _, hold_cols, dd_cols, ud_cols = get_feature_columns()

    h_vals = train_df[hold_cols].values.ravel()
    dd_vals = train_df[dd_cols].values.ravel()
    ud_vals = train_df[ud_cols].values.ravel()

    bounds = {
        "H": (
            float(max(0.012, np.percentile(h_vals, 0.05))),
            float(min(0.450, np.percentile(h_vals, 99.8))),
        ),
        "DD": (
            float(max(0.015, np.percentile(dd_vals, 0.05))),
            float(min(1.500, np.percentile(dd_vals, 99.5))),
        ),
        "UD": (
            float(max(-0.250, np.percentile(ud_vals, 0.05))),
            float(min(1.400, np.percentile(ud_vals, 99.5))),
        ),
    }
    return bounds


def apply_outlier_clipping(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """Apply fitted physiological clipping bounds to Hold, DD, and UD columns."""
    _, hold_cols, dd_cols, ud_cols = get_feature_columns()
    clipped = df.copy()
    clipped[hold_cols] = clipped[hold_cols].clip(*bounds["H"])
    clipped[dd_cols] = clipped[dd_cols].clip(*bounds["DD"])
    clipped[ud_cols] = clipped[ud_cols].clip(*bounds["UD"])
    return clipped


def build_flat_timings(holds: np.ndarray, dds: np.ndarray, uds: np.ndarray) -> np.ndarray:
    """Build interleaved flat timing vector [H_0, DD_0, UD_0, ..., H_M] of shape (N, 3*M + 1)."""
    parts = []
    m = dds.shape[1]
    for k in range(m):
        parts.extend([holds[:, k : k + 1], dds[:, k : k + 1], uds[:, k : k + 1]])
    parts.append(holds[:, m : m + 1])
    return np.concatenate(parts, axis=1).astype(np.float32)


def extract_9channel_sequences(
    holds: np.ndarray,
    dds: np.ndarray,
    uds: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Preprocessing Step 2: Convert variable-length keystroke timings into 9-channel
    biomechanical digraph tensors of shape (N, 9, M), where M = K - 1 transitions.
    """
    h_curr = holds[:, :-1]                          # (N, M)
    h_next = holds[:, 1:]                           # (N, M)

    mean_h = np.mean(holds, axis=1, keepdims=True)  # (N, 1)
    mean_dd = np.mean(dds, axis=1, keepdims=True)   # (N, 1)

    rel_h = h_curr / np.maximum(mean_h, eps)
    rel_dd = dds / np.maximum(mean_dd, eps)
    duty_ratio = np.clip(h_curr / np.maximum(dds, 0.020), 0.0, 5.0)
    rollover_mag = np.maximum(0.0, -uds)

    local_accel = np.zeros_like(dds)
    local_accel[:, 1:] = dds[:, 1:] - dds[:, :-1]

    seq = np.stack(
        [
            h_curr,
            h_next,
            dds,
            uds,
            rel_h,
            rel_dd,
            duty_ratio,
            rollover_mag,
            local_accel,
        ],
        axis=1,
    ).astype(np.float32)
    return seq


def extract_sequences(
    holds: np.ndarray,
    dds: np.ndarray,
    uds: np.ndarray,
    key_block: np.ndarray,
    hold_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full 17-channel input: 9 timing channels + 7 key-identity channels + Hold_Valid flag.
    `key_block` is (7, M) shared by all rows or (N, 7, M) per row; `hold_valid` is None (all
    measured), a per-row (N,) vector, or a per-transition (N, M) array.
    """
    timing = extract_9channel_sequences(holds, dds, uds)
    n, _, m = timing.shape
    kb = key_block if key_block.ndim == 3 else np.broadcast_to(key_block[None], (n,) + key_block.shape)
    if hold_valid is None:
        hv = np.ones((n, 1, m), dtype=np.float32)
    else:
        hv = np.asarray(hold_valid, dtype=np.float32)
        hv = np.broadcast_to(hv[:, None, None], (n, 1, m)) if hv.ndim == 1 else hv[:, None, :]
    return np.concatenate([timing, kb.astype(np.float32), hv.astype(np.float32)], axis=1).astype(np.float32)


def touchify_timings(
    holds: np.ndarray, dds: np.ndarray, touch_hold: float, bounds: Dict[str, Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a touch keyboard: hold durations are unknown, so every hold is replaced with
    the clipped population median (matching browser buildInputTensor) and UD is derived from
    clipped DD minus imputed hold."""
    c_dds = np.clip(dds, *bounds["DD"]).astype(np.float32)
    h_imp = float(np.clip(touch_hold, *bounds["H"]))
    h = np.full_like(holds, h_imp, dtype=np.float32)
    u = np.clip(c_dds - h_imp, *bounds["UD"]).astype(np.float32)
    return h, u


def dataframe_to_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract both (N, 31) flat vectors and (N, 17, 10) sequences for the CMU password."""
    all_cols, hold_cols, dd_cols, ud_cols = get_feature_columns()
    holds = df[hold_cols].values.astype(np.float32)
    dds = df[dd_cols].values.astype(np.float32)
    uds = df[ud_cols].values.astype(np.float32)
    flat = df[all_cols].values.astype(np.float32)
    seq = extract_sequences(holds, dds, uds, CMU_KEY_BLOCK)
    return flat, seq


def fit_channel_scaler(train_seq_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing Step 3b: Fit leakage-free channel-wise Z-score standardization
    across variable-length Train split sequences [(N_i, C, M_i), ...] -> mean (C,), std (C,).
    Binary flag channels in UNSCALED_CHANNELS pass through unchanged (mean 0, std 1).
    """
    c = train_seq_list[0].shape[1]
    flattened = np.concatenate([s.transpose(1, 0, 2).reshape(c, -1) for s in train_seq_list], axis=1)
    mean = flattened.mean(axis=1).astype(np.float32)
    std = (flattened.std(axis=1) + 1e-6).astype(np.float32)
    for idx in UNSCALED_CHANNELS:
        if idx < c:
            mean[idx], std[idx] = 0.0, 1.0
    return mean, std


def apply_channel_scaler(
    seq: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply channel-wise Z-score standardization to (N, C, M) sequences."""
    return ((seq - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def augment_keystroke_batch(
    holds: np.ndarray,
    dds: np.ndarray,
    uds: Optional[np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.RandomState,
    key_block: Optional[np.ndarray] = None,
    warp: bool = True,
    touch_prob: float = 0.0,
    touch_hold: float = 0.1,
) -> np.ndarray:
    """
    Preprocessing Step 4: Smooth Biomechanical Tempo-Warping & Micro-Jitter Augmentation.
    Operates strictly on UNCLIPPED raw holds/dds first:
      1) Apply smooth tempo-warping & micro-jitter to raw_holds and raw_dds (identity when warp=False).
      2) Compute aug_raw_uds = aug_raw_dds - aug_raw_holds[:, :-1] (or passed raw_uds when warp=False).
      3) Independently clip holds to bounds['H'], dds to bounds['DD'], and uds to bounds['UD'].
      4) For touch-converted rows, impute h_imp = clip(touch_hold, *bounds['H']) and recompute
         c_uds = clip(c_dds - h_imp, *bounds['UD']) with Hold_Valid = 0.0.
    """
    n, m = dds.shape
    if key_block is None:
        key_block = np.zeros((len(KEY_CHANNEL_NAMES), m), dtype=np.float32)

    raw_h = holds.astype(np.float32)
    raw_d = dds.astype(np.float32)
    if warp:
        tempo = rng.uniform(0.80, 1.25, size=(n, 1)).astype(np.float32)
        phase = rng.uniform(0, 2 * np.pi, size=(n, 1)).astype(np.float32)
        t_grid = np.linspace(0.0, np.pi, m, dtype=np.float32)[None, :]
        smooth_wave = 1.0 + 0.06 * np.sin(t_grid + phase)

        local_h = rng.normal(1.0, 0.035, size=raw_h.shape).astype(np.float32)
        local_d = rng.normal(1.0, 0.040, size=raw_d.shape).astype(np.float32)

        jitter_h = rng.normal(0.0, 0.0018, size=raw_h.shape).astype(np.float32)
        jitter_dd = rng.normal(0.0, 0.0020, size=raw_d.shape).astype(np.float32)

        aug_raw_holds = (raw_h * (0.65 + 0.35 * tempo) * local_h + jitter_h).astype(np.float32)
        aug_raw_dds = (raw_d * tempo * smooth_wave * local_d + jitter_dd).astype(np.float32)
        aug_raw_uds = (aug_raw_dds - aug_raw_holds[:, :-1]).astype(np.float32)
    else:
        aug_raw_holds = raw_h
        aug_raw_dds = raw_d
        if uds is not None:
            aug_raw_uds = uds.astype(np.float32)
        else:
            aug_raw_uds = (aug_raw_dds - aug_raw_holds[:, :-1]).astype(np.float32)

    c_holds = np.clip(aug_raw_holds, *bounds["H"]).astype(np.float32)
    c_dds = np.clip(aug_raw_dds, *bounds["DD"]).astype(np.float32)
    c_uds = np.clip(aug_raw_uds, *bounds["UD"]).astype(np.float32)

    if touch_prob > 0.0:
        touch = rng.rand(n) < touch_prob
    else:
        touch = np.zeros(n, dtype=bool)
    hv = (~touch).astype(np.float32)
    if touch.any():
        h_imp = float(np.clip(touch_hold, *bounds["H"]))
        c_holds[touch] = h_imp
        c_uds[touch] = np.clip(c_dds[touch] - h_imp, *bounds["UD"]).astype(np.float32)

    seq = extract_sequences(c_holds, c_dds, c_uds, key_block, hv)
    return apply_channel_scaler(seq, mean, std)


def parse_keyrecs_sentence_corpus(
    free_csv_path: Path,
    bounds: Dict[str, Tuple[float, float]],
    window_keys: int = 25,
    stride_keys: int = 25,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], List[List[Tuple[str, float, float]]]]]:
    """
    Reconstruct exact physical keydown/keyup timestamps from KeyRecs free-text transcription
    sessions (Proust & Murakami passages) via lossless line-by-line parsing (recovering all 26
    unescaped '\"' rows and 198 session-end strokes), break runs on any corrupt-line discontinuity
    or Backspace/edit, and slice non-overlapping sentence windows (`stride_keys = window_keys = 25`).
    """
    rows_by_ps, parser_stats = parse_keyrecs_free_csv(free_csv_path)
    modifiers = {"Shift", "CapsLock", "Control", "Alt", "Meta"}
    clean_runs_by_ps: Dict[Tuple[str, int], List[List[Tuple[str, float, float]]]] = {}
    run_rows_by_ps: Dict[Tuple[str, int], List[List[int]]] = {}

    for ps, items in rows_by_ps.items():
        runs: List[List[Tuple[str, float, float]]] = []
        r_rows: List[List[int]] = []
        curr: List[Tuple[str, float, float]] = []
        curr_rows: List[int] = []
        t_cur = 0.0
        for row_idx, (kind, k, hv, dd) in enumerate(items):
            if kind == "break":
                if len(curr) >= window_keys:
                    runs.append(curr)
                    r_rows.append(curr_rows)
                curr = []
                curr_rows = []
                t_cur += 100.0
                continue
            if k not in modifiers:
                ch = " " if k == "Space" else (k.lower() if len(k) == 1 else "")
                if not ch or hv < 0.008 or hv > 0.80:
                    if len(curr) >= window_keys:
                        runs.append(curr)
                        r_rows.append(curr_rows)
                    curr = []
                    curr_rows = []
                else:
                    if curr and (t_cur - curr[-1][1] < 0.008 or t_cur - curr[-1][1] > 1.80):
                        if len(curr) >= window_keys:
                            runs.append(curr)
                            r_rows.append(curr_rows)
                        curr = []
                        curr_rows = []
                    curr.append((ch, float(t_cur), float(hv)))
                    curr_rows.append(int(row_idx))
            if kind == "end":
                if len(curr) >= window_keys:
                    runs.append(curr)
                    r_rows.append(curr_rows)
                curr = []
                curr_rows = []
                t_cur += 100.0
            else:
                t_cur += dd
        if len(curr) >= window_keys:
            runs.append(curr)
            r_rows.append(curr_rows)

        if runs:
            clean_runs_by_ps[ps] = runs
            run_rows_by_ps[ps] = r_rows

    # Keep only participants present in BOTH Session 1 and Session 2 with >= 8 chunks each
    valid_participants = sorted(
        {
            p
            for (p, s) in clean_runs_by_ps
            if (p, 1) in clean_runs_by_ps and (p, 2) in clean_runs_by_ps
        }
    )

    records = []
    filtered_runs: Dict[Tuple[str, int], List[List[Tuple[str, float, float]]]] = {}
    for p in valid_participants:
        p_ok = True
        per_s_chunks = {1: [], 2: []}
        for s in (1, 2):
            chunk_idx = 0
            for run, run_ridx in zip(clean_runs_by_ps[(p, s)], run_rows_by_ps[(p, s)]):
                for start in range(0, len(run) - window_keys + 1, stride_keys):
                    seg = run[start : start + window_keys]
                    seg_rows = run_ridx[start : start + window_keys]
                    txt = "".join(x[0] for x in seg)
                    t_d = np.array([x[1] for x in seg], dtype=np.float64)
                    raw_holds = np.array([x[2] for x in seg], dtype=np.float32)
                    raw_dds = np.diff(t_d).astype(np.float32)
                    raw_uds = (raw_dds - raw_holds[:-1]).astype(np.float32)
                    holds = np.clip(raw_holds, *bounds["H"]).astype(np.float32)
                    dds = np.clip(raw_dds, *bounds["DD"]).astype(np.float32)
                    uds = np.clip(raw_uds, *bounds["UD"]).astype(np.float32)
                    per_s_chunks[s].append(
                        {
                            "participant": p,
                            "session": s,
                            "chunk_id": chunk_idx,
                            "text": txt,
                            "codes": [code_for_label(ch) for ch in txt],
                            "raw_holds": raw_holds,
                            "raw_dds": raw_dds,
                            "raw_uds": raw_uds,
                            "holds": holds,
                            "dds": dds,
                            "uds": uds,
                            "t_range": (float(t_d[0]), float(t_d[-1])),
                            "key_idx_range": (int(seg_rows[0]), int(seg_rows[-1])),
                        }
                    )
                    chunk_idx += 1
            if len(per_s_chunks[s]) < 8:
                p_ok = False
        if p_ok:
            records.extend(per_s_chunks[1])
            records.extend(per_s_chunks[2])
            filtered_runs[(p, 1)] = clean_runs_by_ps[(p, 1)]
            filtered_runs[(p, 2)] = clean_runs_by_ps[(p, 2)]

    out_df = pd.DataFrame(records)
    out_df.attrs["parser_stats"] = parser_stats
    return out_df, filtered_runs


def extract_shared_phrase_samples(
    filtered_runs: Dict[Tuple[str, int], List[List[Tuple[str, float, float]]]],
    participants: List[str],
    phrase: str,
    bounds: Dict[str, Tuple[float, float]],
) -> List[Dict]:
    """Extract exact typo-free occurrences of a shared English phrase across participants."""
    k_len = len(phrase)
    matched = []
    for p in participants:
        for s in (1, 2):
            for run in filtered_runs.get((p, s), []):
                txt = "".join(x[0] for x in run)
                pos = txt.find(phrase)
                if pos != -1:
                    seg = run[pos : pos + k_len]
                    t_d = np.array([x[1] for x in seg], dtype=np.float64)
                    raw_holds = np.array([x[2] for x in seg], dtype=np.float32)
                    raw_dds = np.diff(t_d).astype(np.float32)
                    raw_uds = (raw_dds - raw_holds[:-1]).astype(np.float32)
                    holds = np.clip(raw_holds, *bounds["H"]).astype(np.float32)
                    dds = np.clip(raw_dds, *bounds["DD"]).astype(np.float32)
                    uds = np.clip(raw_uds, *bounds["UD"]).astype(np.float32)
                    matched.append(
                        {
                            "participant": p,
                            "session": s,
                            "phrase": phrase,
                            "raw_holds": raw_holds,
                            "raw_dds": raw_dds,
                            "raw_uds": raw_uds,
                            "holds": holds,
                            "dds": dds,
                            "uds": uds,
                        }
                    )
                    break
    return matched


def build_sentence_split(
    records_df: pd.DataFrame,
    split_subjs: List[str],
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
) -> SentenceSplitData:
    sub = records_df[records_df["participant"].isin(split_subjs)].reset_index(drop=True)
    subj_map = {s: idx for idx, s in enumerate(split_subjs)}
    holds = np.stack(sub["holds"].values, axis=0).astype(np.float32)
    dds = np.stack(sub["dds"].values, axis=0).astype(np.float32)
    uds = np.stack(sub["uds"].values, axis=0).astype(np.float32)
    raw_holds = np.stack(sub["raw_holds"].values, axis=0).astype(np.float32)
    raw_dds = np.stack(sub["raw_dds"].values, axis=0).astype(np.float32)
    raw_uds = np.stack(sub["raw_uds"].values, axis=0).astype(np.float32)
    t_ranges = np.asarray(sub["t_range"].tolist(), dtype=np.float64)
    key_idx_ranges = np.asarray(sub["key_idx_range"].tolist(), dtype=np.int64)
    flat = build_flat_timings(holds, dds, uds)
    codes = sub["codes"].tolist()
    key_blocks = np.stack([key_channel_matrix(pairs_from_codes(c)) for c in codes], axis=0)
    seq_unscaled = extract_sequences(holds, dds, uds, key_blocks)
    seq_scaled = apply_channel_scaler(seq_unscaled, chan_mean, chan_std)
    return SentenceSplitData(
        holds=holds,
        dds=dds,
        uds=uds,
        raw_flat=flat,
        seq_unscaled=seq_unscaled,
        seq_scaled=seq_scaled,
        subject_ids=sub["participant"].map(subj_map).values.astype(np.int64),
        subjects=split_subjs,
        sessions=sub["session"].values.astype(np.int64),
        chunk_ids=sub["chunk_id"].values.astype(np.int64),
        texts=sub["text"].tolist(),
        key_blocks=key_blocks,
        codes=codes,
        raw_holds=raw_holds,
        raw_dds=raw_dds,
        raw_uds=raw_uds,
        t_ranges=t_ranges,
        key_idx_ranges=key_idx_ranges,
    )


def load_keyrecs_fixed_split(
    fixed_csv_path: Path,
    split_subjs: List[str],
    bounds: Dict[str, Tuple[float, float]],
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load KeyRecs fixed-text typing sequences (10 keys -> L=9 digraphs) for the given subjects.
    Returns (raw_holds, raw_dds, raw_uds, seq_scaled, local_subject_ids) so callers augmenting
    training batches receive unclipped raw timings while seq_scaled uses independently clipped timings.
    """
    if not fixed_csv_path.exists():
        return None
    df = pd.read_csv(fixed_csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["participant"].isin(split_subjs)].dropna(subset=KEYRECS_HOLD_COLS + KEYRECS_DD_COLS).reset_index(drop=True)
    if df.empty:
        return None
    subj_map = {s: idx for idx, s in enumerate(split_subjs)}
    raw_holds = df[KEYRECS_HOLD_COLS].values.astype(np.float32)
    raw_dds = df[KEYRECS_DD_COLS].values.astype(np.float32)
    raw_uds = (raw_dds - raw_holds[:, :-1]).astype(np.float32)
    holds = np.clip(raw_holds, *bounds["H"]).astype(np.float32)
    dds = np.clip(raw_dds, *bounds["DD"]).astype(np.float32)
    uds = np.clip(raw_uds, *bounds["UD"]).astype(np.float32)
    seq_unscaled = extract_sequences(holds, dds, uds, KEYRECS_FIXED_KEY_BLOCK)
    seq_scaled = apply_channel_scaler(seq_unscaled, chan_mean, chan_std)
    sids = df["participant"].map(subj_map).values.astype(np.int64)
    return raw_holds, raw_dds, raw_uds, seq_scaled, sids


def prepare_full_pipeline(
    csv_path: Path,
    processed_dir: Path,
    seed: int = SEED,
) -> Tuple[Dict[str, SplitData], Dict]:
    """
    Run complete 4-stage preprocessing on CMU StrongPassword Dataset AND (if present)
    KeyRecs Sentence Transcription Dataset with strict Subject-Disjoint Splits.
    """
    raw_df = pd.read_csv(csv_path)
    all_cols, hold_cols, dd_cols, ud_cols = get_feature_columns()

    subjects = sorted(raw_df["subject"].unique().tolist())
    train_subjs, val_subjs, test_subjs = split_subjects_disjoint(subjects, n_train=35, n_val=6, seed=seed)

    raw_train_df = raw_df[raw_df["subject"].isin(train_subjs)].reset_index(drop=True)
    bounds = fit_outlier_bounds(raw_train_df)

    clipped_df = apply_outlier_clipping(raw_df, bounds)
    train_clipped = clipped_df[clipped_df["subject"].isin(train_subjs)].reset_index(drop=True)

    # Compute Before vs After skewness / kurtosis strictly on the exact same Train population (n=14,000)
    before_skew = float(raw_train_df[all_cols].skew().mean())
    before_max_skew = float(raw_train_df[all_cols].skew().max())
    before_kurt = float(raw_train_df[all_cols].kurt().mean())
    after_skew = float(train_clipped[all_cols].skew().mean())
    after_max_skew = float(train_clipped[all_cols].skew().max())
    after_kurt = float(train_clipped[all_cols].kurt().mean())

    _, cmu_train_seq_unscaled = dataframe_to_sequences(train_clipped)

    # Check if KeyRecs sentence corpus is available alongside CMU
    free_csv_path = csv_path.parent / "KeyRecs-free-text.csv"
    fixed_csv_path = csv_path.parent / "KeyRecs-fixed-text.csv"

    kr_records_df = None
    kr_runs = None
    kr_splits_subjs = None
    kr_parser_stats = None
    scaler_inputs = [cmu_train_seq_unscaled]
    train_hold_pool = [raw_train_df[hold_cols].values.astype(np.float32).ravel()]

    if free_csv_path.exists():
        kr_records_df, kr_runs = parse_keyrecs_sentence_corpus(
            free_csv_path, bounds, window_keys=25, stride_keys=25
        )
        kr_parser_stats = dict(kr_records_df.attrs.get("parser_stats", {}))
        kr_all_subjs = sorted(kr_records_df["participant"].unique().tolist())
        kr_train_s, kr_val_s, kr_test_s = split_subjects_disjoint(
            kr_all_subjs, n_train=68, n_val=10, seed=seed
        )
        kr_splits_subjs = {"train": kr_train_s, "val": kr_val_s, "test": kr_test_s}
        kr_train_df = kr_records_df[kr_records_df["participant"].isin(kr_train_s)]
        kr_h = np.stack(kr_train_df["holds"].values, axis=0).astype(np.float32)
        kr_d = np.stack(kr_train_df["dds"].values, axis=0).astype(np.float32)
        kr_u = np.stack(kr_train_df["uds"].values, axis=0).astype(np.float32)
        kr_raw_h = np.stack(kr_train_df["raw_holds"].values, axis=0).astype(np.float32)
        kr_kb = np.stack(
            [key_channel_matrix(pairs_from_codes(c)) for c in kr_train_df["codes"].tolist()], axis=0
        )
        scaler_inputs.append(extract_sequences(kr_h, kr_d, kr_u, kr_kb))
        train_hold_pool.append(kr_raw_h.ravel())

    chan_mean, chan_std = fit_channel_scaler(scaler_inputs)
    # Median hold of Train keystrokes: imputed for touch keyboards that expose no keyup timing
    touch_hold = float(np.median(np.concatenate(train_hold_pool)))

    splits: Dict[str, SplitData] = {}
    for split_name, split_subjs in [
        ("train", train_subjs),
        ("val", val_subjs),
        ("test", test_subjs),
    ]:
        raw_sub_df = raw_df[raw_df["subject"].isin(split_subjs)].reset_index(drop=True)
        sub_df = clipped_df[clipped_df["subject"].isin(split_subjs)].reset_index(drop=True)
        subj_to_idx = {s: idx for idx, s in enumerate(split_subjs)}
        flat, seq_unscaled = dataframe_to_sequences(sub_df)
        seq_scaled = apply_channel_scaler(seq_unscaled, chan_mean, chan_std)
        splits[split_name] = SplitData(
            df=sub_df,
            raw_flat=flat,
            seq_unscaled=seq_unscaled,
            seq_scaled=seq_scaled,
            subject_ids=sub_df["subject"].map(subj_to_idx).values.astype(np.int64),
            subjects=split_subjs,
            sessions=sub_df["sessionIndex"].values.astype(np.int64),
            reps=sub_df["rep"].values.astype(np.int64),
            key_block=CMU_KEY_BLOCK,
            raw_holds=raw_sub_df[hold_cols].values.astype(np.float32),
            raw_dds=raw_sub_df[dd_cols].values.astype(np.float32),
            raw_uds=raw_sub_df[ud_cols].values.astype(np.float32),
        )

    sentence_splits: Dict[str, SentenceSplitData] = {}
    keyrecs_fixed_train = None
    keyrecs_meta = None
    if kr_records_df is not None and kr_splits_subjs is not None:
        for sname in ("train", "val", "test"):
            sentence_splits[sname] = build_sentence_split(
                kr_records_df, kr_splits_subjs[sname], chan_mean, chan_std
            )
        keyrecs_fixed_train = load_keyrecs_fixed_split(
            fixed_csv_path, kr_splits_subjs["train"], bounds, chan_mean, chan_std
        )
        n_kr_tot = sum(len(v) for v in kr_splits_subjs.values())
        n_chunks_tot = sum(int(len(sentence_splits[sn].subject_ids)) for sn in ("train", "val", "test"))
        keyrecs_meta = {
            **(kr_parser_stats or {}),
            "total_valid_subjects": n_kr_tot,
            "train_subjects": len(kr_splits_subjs["train"]),
            "train_participants": len(kr_splits_subjs["train"]),
            "train_subjects_pct": round(len(kr_splits_subjs["train"]) / max(1, n_kr_tot) * 100.0, 2),
            "train_sentence_chunks": int(len(sentence_splits["train"].subject_ids)),
            "train_chunks_pct": round(len(sentence_splits["train"].subject_ids) / max(1, n_chunks_tot) * 100.0, 2),
            "train_fixed_strokes": int(len(keyrecs_fixed_train[4])) if keyrecs_fixed_train is not None else 0,
            "val_subjects": len(kr_splits_subjs["val"]),
            "val_participants": len(kr_splits_subjs["val"]),
            "val_subjects_pct": round(len(kr_splits_subjs["val"]) / max(1, n_kr_tot) * 100.0, 2),
            "val_sentence_chunks": int(len(sentence_splits["val"].subject_ids)),
            "val_chunks_pct": round(len(sentence_splits["val"].subject_ids) / max(1, n_chunks_tot) * 100.0, 2),
            "test_subjects": len(kr_splits_subjs["test"]),
            "test_participants": len(kr_splits_subjs["test"]),
            "test_subjects_pct": round(len(kr_splits_subjs["test"]) / max(1, n_kr_tot) * 100.0, 2),
            "test_sentence_chunks": int(len(sentence_splits["test"].subject_ids)),
            "test_chunks_pct": round(len(sentence_splits["test"].subject_ids) / max(1, n_chunks_tot) * 100.0, 2),
            "sentence_window_keys": 25,
            "sentence_stride_keys": 25,
            "sentence_digraphs": 24,
        }

    n_cmu_tot = len(subjects)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "dataset_rows": int(len(raw_df)),
        "dataset_cols": int(raw_df.shape[1]),
        "num_subjects_total": n_cmu_tot,
        "split_counts": {
            "train_subjects": len(train_subjs),
            "train_subjects_pct": round(len(train_subjs) / max(1, n_cmu_tot) * 100.0, 2),
            "train_samples": int(len(splits["train"].df)),
            "val_subjects": len(val_subjs),
            "val_subjects_pct": round(len(val_subjs) / max(1, n_cmu_tot) * 100.0, 2),
            "val_samples": int(len(splits["val"].df)),
            "test_subjects": len(test_subjs),
            "test_subjects_pct": round(len(test_subjs) / max(1, n_cmu_tot) * 100.0, 2),
            "test_samples": int(len(splits["test"].df)),
        },
        "keyrecs_sentence_corpus": keyrecs_meta,
        "subjects_split": {
            "train": train_subjs,
            "val": val_subjs,
            "test": test_subjs,
        },
        "outlier_bounds": bounds,
        "preprocessing_before_after": {
            "population": "CMU Train split (35 subjects, n=14,000)",
            "population_n": int(len(raw_train_df)),
            "before_mean_skewness": round(before_skew, 4),
            "after_mean_skewness": round(after_skew, 4),
            "before_max_skewness": round(before_max_skew, 4),
            "after_max_skewness": round(after_max_skew, 4),
            "before_mean_kurtosis": round(before_kurt, 4),
            "after_mean_kurtosis": round(after_kurt, 4),
            "rollover_ratio_pct": round(float((raw_train_df[ud_cols] < 0).values.mean() * 100), 2),
        },
        "channel_names": CHANNEL_NAMES,
        "digraph_labels": DIGRAPH_LABELS,
        "channel_mean": [round(float(x), 6) for x in chan_mean],
        "channel_std": [round(float(x), 6) for x in chan_std],
        "touch_hold_impute_sec": round(touch_hold, 6),
    }

    # Attach auxiliary runtime bundles in memory via metadata["_runtime"] (not written to JSON)
    json_copy = dict(metadata)
    with open(processed_dir / "scaler_and_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_copy, f, indent=2, ensure_ascii=False)

    metadata["_sentence_splits"] = sentence_splits
    metadata["_keyrecs_fixed_train"] = keyrecs_fixed_train
    metadata["_keyrecs_runs"] = kr_runs
    metadata["_keyrecs_splits_subjs"] = kr_splits_subjs
    return splits, metadata
