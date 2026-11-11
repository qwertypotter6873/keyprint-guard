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

CHANNEL_NAMES: List[str] = [
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
    seq_unscaled: np.ndarray    # (N, 9, M) unscaled 9-channel sequence
    seq_scaled: np.ndarray      # (N, 9, M) Z-score standardized 9-channel sequence
    subject_ids: np.ndarray     # (N,) integer label within split
    subjects: List[str]         # Unique subject codes in this split
    sessions: np.ndarray        # (N,) session index
    reps: np.ndarray            # (N,) repetition / chunk index


@dataclass
class SentenceSplitData:
    holds: np.ndarray           # (N, M+1) clipped hold durations in seconds
    dds: np.ndarray             # (N, M) clipped press-to-press latencies in seconds
    uds: np.ndarray             # (N, M) clipped flight latencies in seconds
    raw_flat: np.ndarray        # (N, 3*M+1) flat vector for statistical baseline
    seq_unscaled: np.ndarray    # (N, 9, M)
    seq_scaled: np.ndarray      # (N, 9, M)
    subject_ids: np.ndarray     # (N,) local subject index 0..len(subjects)-1
    subjects: List[str]         # Unique participant IDs
    sessions: np.ndarray        # (N,) session 1 or 2
    chunk_ids: np.ndarray       # (N,) sequential chunk index within session
    texts: List[str]            # (N,) actual typed English substring


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


def dataframe_to_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract both (N, 31) flat vectors and (N, 9, 10) biomechanical sequences."""
    all_cols, hold_cols, dd_cols, ud_cols = get_feature_columns()
    holds = df[hold_cols].values.astype(np.float32)
    dds = df[dd_cols].values.astype(np.float32)
    uds = df[ud_cols].values.astype(np.float32)
    flat = df[all_cols].values.astype(np.float32)
    seq = extract_9channel_sequences(holds, dds, uds)
    return flat, seq


def fit_channel_scaler(train_seq_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing Step 3b: Fit leakage-free channel-wise Z-score standardization
    across variable-length Train split sequences [(N_i, C=9, M_i), ...] -> mean (9,), std (9,).
    """
    flattened = np.concatenate([s.transpose(1, 0, 2).reshape(9, -1) for s in train_seq_list], axis=1)
    mean = flattened.mean(axis=1).astype(np.float32)
    std = (flattened.std(axis=1) + 1e-6).astype(np.float32)
    return mean, std


def apply_channel_scaler(
    seq: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply channel-wise Z-score standardization to (N, C, M) sequences."""
    return ((seq - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def augment_keystroke_batch(
    holds: np.ndarray,
    dds: np.ndarray,
    uds: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Preprocessing Step 4: Smooth Biomechanical Tempo-Warping & Micro-Jitter Augmentation.
    Models natural session-to-session WPM speed drift (0.80x ~ 1.25x) plus smooth
    intra-word acceleration curves without destroying positional digraph alignment.
    """
    n, m = dds.shape
    tempo = rng.uniform(0.80, 1.25, size=(n, 1)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1)).astype(np.float32)
    t_grid = np.linspace(0.0, np.pi, m, dtype=np.float32)[None, :]
    smooth_wave = 1.0 + 0.06 * np.sin(t_grid + phase)

    local_h = rng.normal(1.0, 0.035, size=holds.shape).astype(np.float32)
    local_d = rng.normal(1.0, 0.040, size=dds.shape).astype(np.float32)

    jitter_h = rng.normal(0.0, 0.0018, size=holds.shape).astype(np.float32)
    jitter_dd = rng.normal(0.0, 0.0020, size=dds.shape).astype(np.float32)

    aug_holds = np.clip(holds * (0.65 + 0.35 * tempo) * local_h + jitter_h, *bounds["H"])
    aug_dds = np.clip(dds * tempo * smooth_wave * local_d + jitter_dd, *bounds["DD"])
    aug_uds = np.clip(aug_dds - aug_holds[:, :-1], *bounds["UD"])

    seq = extract_9channel_sequences(aug_holds, aug_dds, aug_uds)
    return apply_channel_scaler(seq, mean, std)


def parse_keyrecs_sentence_corpus(
    free_csv_path: Path,
    bounds: Dict[str, Tuple[float, float]],
    window_keys: int = 25,
    stride_keys: int = 20,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], List[Tuple[str, float, float]]]]:
    """
    Reconstruct exact physical keydown/keyup timestamps from KeyRecs free-text transcription
    sessions (Proust & Murakami passages), strip modifier keys identically to browser
    LiveKeystrokeRecorder, split on Backspace/edits into 100% typo-free contiguous spans,
    and slice sentence windows of `window_keys` keys (`L = window_keys - 1` digraphs).
    """
    df = pd.read_csv(free_csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ["DU.key1.key1", "DD.key1.key2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["DU.key1.key1", "DD.key1.key2"])

    modifiers = {"Shift", "CapsLock", "Control", "Alt", "Meta"}
    clean_runs_by_ps: Dict[Tuple[str, int], List[List[Tuple[str, float, float]]]] = {}

    for (participant, session), grp in df.groupby(["participant", "session"]):
        keys = grp["key1"].astype(str).tolist()
        h_arr = grp["DU.key1.key1"].to_numpy(dtype=np.float64)
        dd_arr = grp["DD.key1.key2"].to_numpy(dtype=np.float64)
        t_down = np.concatenate([[0.0], np.cumsum(dd_arr[:-1])])

        runs: List[List[Tuple[str, float, float]]] = []
        curr: List[Tuple[str, float, float]] = []
        for k, td, hv in zip(keys, t_down, h_arr):
            if k in modifiers:
                continue
            ch = " " if k == "Space" else (k.lower() if len(k) == 1 else "")
            if not ch or hv < 0.008 or hv > 0.80:
                if len(curr) >= window_keys:
                    runs.append(curr)
                curr = []
                continue
            if curr and (td - curr[-1][1] < 0.008 or td - curr[-1][1] > 1.80):
                if len(curr) >= window_keys:
                    runs.append(curr)
                curr = []
            curr.append((ch, float(td), float(hv)))
        if len(curr) >= window_keys:
            runs.append(curr)

        if runs:
            clean_runs_by_ps[(str(participant), int(session))] = runs

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
            for run in clean_runs_by_ps[(p, s)]:
                for start in range(0, len(run) - window_keys + 1, stride_keys):
                    seg = run[start : start + window_keys]
                    txt = "".join(x[0] for x in seg)
                    t_d = np.array([x[1] for x in seg], dtype=np.float32)
                    holds = np.clip(np.array([x[2] for x in seg], dtype=np.float32), *bounds["H"])
                    dds = np.clip(np.diff(t_d), *bounds["DD"])
                    uds = np.clip(dds - holds[:-1], *bounds["UD"])
                    per_s_chunks[s].append(
                        {
                            "participant": p,
                            "session": s,
                            "chunk_id": chunk_idx,
                            "text": txt,
                            "holds": holds,
                            "dds": dds,
                            "uds": uds,
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

    return pd.DataFrame(records), filtered_runs


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
                    t_d = np.array([x[1] for x in seg], dtype=np.float32)
                    holds = np.clip(np.array([x[2] for x in seg], dtype=np.float32), *bounds["H"])
                    dds = np.clip(np.diff(t_d), *bounds["DD"])
                    uds = np.clip(dds - holds[:-1], *bounds["UD"])
                    matched.append(
                        {
                            "participant": p,
                            "session": s,
                            "phrase": phrase,
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
    flat = build_flat_timings(holds, dds, uds)
    seq_unscaled = extract_9channel_sequences(holds, dds, uds)
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
    Returns (holds, dds, uds, seq_scaled, local_subject_ids).
    """
    if not fixed_csv_path.exists():
        return None
    df = pd.read_csv(fixed_csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["participant"].isin(split_subjs)].dropna(subset=KEYRECS_HOLD_COLS + KEYRECS_DD_COLS).reset_index(drop=True)
    if df.empty:
        return None
    subj_map = {s: idx for idx, s in enumerate(split_subjs)}
    holds = np.clip(df[KEYRECS_HOLD_COLS].values.astype(np.float32), *bounds["H"])
    dds = np.clip(df[KEYRECS_DD_COLS].values.astype(np.float32), *bounds["DD"])
    uds = np.clip(dds - holds[:, :-1], *bounds["UD"])
    seq_unscaled = extract_9channel_sequences(holds, dds, uds)
    seq_scaled = apply_channel_scaler(seq_unscaled, chan_mean, chan_std)
    sids = df["participant"].map(subj_map).values.astype(np.int64)
    return holds, dds, uds, seq_scaled, sids


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

    # Compute Before vs After skewness / kurtosis metrics for reporting
    before_skew = float(raw_df[all_cols].skew().mean())
    before_max_skew = float(raw_df[all_cols].skew().max())
    before_kurt = float(raw_df[all_cols].kurt().mean())

    clipped_df = apply_outlier_clipping(raw_df, bounds)
    after_skew = float(clipped_df[all_cols].skew().mean())
    after_max_skew = float(clipped_df[all_cols].skew().max())
    after_kurt = float(clipped_df[all_cols].kurt().mean())

    train_clipped = clipped_df[clipped_df["subject"].isin(train_subjs)].reset_index(drop=True)
    _, cmu_train_seq_unscaled = dataframe_to_sequences(train_clipped)

    # Check if KeyRecs sentence corpus is available alongside CMU
    free_csv_path = csv_path.parent / "KeyRecs-free-text.csv"
    fixed_csv_path = csv_path.parent / "KeyRecs-fixed-text.csv"

    kr_records_df = None
    kr_runs = None
    kr_splits_subjs = None
    scaler_inputs = [cmu_train_seq_unscaled]

    if free_csv_path.exists():
        kr_records_df, kr_runs = parse_keyrecs_sentence_corpus(
            free_csv_path, bounds, window_keys=25, stride_keys=20
        )
        kr_all_subjs = sorted(kr_records_df["participant"].unique().tolist())
        kr_train_s, kr_val_s, kr_test_s = split_subjects_disjoint(
            kr_all_subjs, n_train=68, n_val=10, seed=seed
        )
        kr_splits_subjs = {"train": kr_train_s, "val": kr_val_s, "test": kr_test_s}
        kr_train_df = kr_records_df[kr_records_df["participant"].isin(kr_train_s)]
        kr_h = np.stack(kr_train_df["holds"].values, axis=0).astype(np.float32)
        kr_d = np.stack(kr_train_df["dds"].values, axis=0).astype(np.float32)
        kr_u = np.stack(kr_train_df["uds"].values, axis=0).astype(np.float32)
        scaler_inputs.append(extract_9channel_sequences(kr_h, kr_d, kr_u))

    chan_mean, chan_std = fit_channel_scaler(scaler_inputs)

    splits: Dict[str, SplitData] = {}
    for split_name, split_subjs in [
        ("train", train_subjs),
        ("val", val_subjs),
        ("test", test_subjs),
    ]:
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
        keyrecs_meta = {
            "total_valid_subjects": sum(len(v) for v in kr_splits_subjs.values()),
            "train_subjects": len(kr_splits_subjs["train"]),
            "train_sentence_chunks": int(len(sentence_splits["train"].subject_ids)),
            "train_fixed_strokes": int(len(keyrecs_fixed_train[4])) if keyrecs_fixed_train is not None else 0,
            "val_subjects": len(kr_splits_subjs["val"]),
            "val_sentence_chunks": int(len(sentence_splits["val"].subject_ids)),
            "test_subjects": len(kr_splits_subjs["test"]),
            "test_sentence_chunks": int(len(sentence_splits["test"].subject_ids)),
            "sentence_window_keys": 25,
            "sentence_digraphs": 24,
        }

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "dataset_rows": int(len(raw_df)),
        "dataset_cols": int(raw_df.shape[1]),
        "num_subjects_total": len(subjects),
        "split_counts": {
            "train_subjects": len(train_subjs),
            "train_samples": int(len(splits["train"].df)),
            "val_subjects": len(val_subjs),
            "val_samples": int(len(splits["val"].df)),
            "test_subjects": len(test_subjs),
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
            "before_mean_skewness": round(before_skew, 4),
            "after_mean_skewness": round(after_skew, 4),
            "before_max_skewness": round(before_max_skew, 4),
            "after_max_skewness": round(after_max_skew, 4),
            "before_mean_kurtosis": round(before_kurt, 4),
            "after_mean_kurtosis": round(after_kurt, 4),
            "rollover_ratio_pct": round(float((raw_df[ud_cols] < 0).values.mean() * 100), 2),
        },
        "channel_names": CHANNEL_NAMES,
        "digraph_labels": DIGRAPH_LABELS,
        "channel_mean": [round(float(x), 6) for x in chan_mean],
        "channel_std": [round(float(x), 6) for x in chan_std],
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
