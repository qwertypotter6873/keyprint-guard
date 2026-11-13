"""
User-chosen passwords: digraph-synthesised typing of arbitrary strings by real people, and the
dual-encoder password score calibrated on it.

Why: the CMU dataset has one password, so a cohort/threshold calibrated on it does not transfer to
the strings users pick. A person's typing of any string is approximated from their own free-text
keystrokes: for each consecutive key pair the interval (and holds) come from that person's real
instances of the same digraph, drawn around one random moment of the session ("coherent" draw, so
the attempt keeps that moment's tempo). Checked against real repeated words: impostor score
distributions match, genuine scores come out slightly lower (pessimistic).

Score (web: features.js dualPasswordZ): z = (raw9 - mu9)/sd9 + (raw17 - mu17)/sd17, where raw* is
the template similarity 0.65*cos(centroid) + 0.35*max cos(exemplar) of each encoder; mu/sd are the
impostor statistics on Validation strings, and tau is the Validation genuine quantile at the target FRR.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from behavior_profile import kind_of
from capture_sim import make_template
from keyboard_layout import code_for_label, key_channel_matrix, pairs_from_codes
from preprocessing import apply_channel_scaler, extract_sequences, touchify_timings

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DISPLAY_SLOPE = 1.0  # web similarity = sigmoid(slope * (z - tau)); 0.5 is the decision point


def digraph_tables(raw_sessions, people: Sequence[str], max_dd: float = 1.5) -> Dict:
    """{(person, session): {(a, b): [(t, dd, hold_a, hold_b, occ_id), ...]}} from clean letter->letter pairs."""
    tabs = {}
    next_occ_id = 0
    for p in people:
        for s in (1, 2):
            ev = raw_sessions.get((p, s))
            if not ev:
                continue
            tab = defaultdict(list)
            prev = None
            for label, t, h in ev:
                kind, _ = kind_of(label)
                ok = kind == "char" and len(label) == 1 and label.isalpha() and 0.008 < h < 1.2
                if ok and prev is not None and 0.006 < t - prev[1] < max_dd:
                    tab[(prev[0], label.lower())].append((prev[1], t - prev[1], prev[2], h, next_occ_id))
                    next_occ_id += 1
                prev = (label.lower(), t, h) if ok else None
            tabs[(p, s)] = dict(tab)
    return tabs


def covered(tab: Dict, word: str, min_per_occ: int = 1) -> bool:
    need = Counter(zip(word, word[1:]))
    return all(len(tab.get(pr, ())) >= min_per_occ * cnt for pr, cnt in need.items())


def covered_disjoint_s1(tabs: Dict, person: str, word: str) -> bool:
    """Ensure Session 1 has >= 2x occurrences per digraph (for disjoint enroll vs same-day query pools)
    and Session 2 has >= 1x occurrence per digraph."""
    return (
        (person, 1) in tabs
        and (person, 2) in tabs
        and covered(tabs[(person, 1)], word, min_per_occ=2)
        and covered(tabs[(person, 2)], word, min_per_occ=1)
    )


def partition_tab_disjoint(tab1: Dict, word: str, rng) -> Tuple[Dict, Dict]:
    """Partition a person's Session-1 digraph instances for `word` into two strictly disjoint occ_id pools."""
    enr_tab: Dict = {}
    same_tab: Dict = {}
    for pr in sorted(set(zip(word, word[1:]))):
        inst = list(tab1[pr])
        perm = rng.permutation(len(inst))
        mid = len(inst) // 2
        if mid < 1 or len(inst) - mid < 1:
            raise ValueError(f"Insufficient Session-1 occurrences to partition disjointly for digraph {pr}")
        enr_tab[pr] = [inst[int(i)] for i in perm[:mid]]
        same_tab[pr] = [inst[int(i)] for i in perm[mid:]]
    return enr_tab, same_tab


def synthesise(tab: Dict, word: str, rng, window: float = 120.0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """One typing of `word`: unclipped raw holds (n,), unclipped raw dds (n-1,), and source occ_ids (n-1,)."""
    pairs = list(zip(word, word[1:]))
    a0 = pairs[rng.integers(len(pairs))]
    anchor = tab[a0][rng.integers(len(tab[a0]))][0]
    holds, dds, occ_ids = [], [], []
    for i, pr in enumerate(pairs):
        inst = tab[pr]
        ts = np.array([x[0] for x in inst])
        near = np.where(np.abs(ts - anchor) <= window)[0]
        if len(near) == 0:
            near = np.argsort(np.abs(ts - anchor))[:2]
        x = inst[int(rng.choice(near))]
        dds.append(x[1])
        holds.append(x[2])
        occ_ids.append(int(x[4]) if len(x) > 4 else -1)
        if i == len(pairs) - 1:
            holds.append(x[3])
    return np.array(holds, dtype=np.float32), np.array(dds, dtype=np.float32), occ_ids


def synth(tab: Dict, word: str, rng, window: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
    """Two-tuple wrapper returning unclipped (holds, dds) in seconds."""
    holds, dds, _ = synthesise(tab, word, rng, window=window)
    return holds, dds


def choose_strings(
    raw_sessions,
    tabs,
    people: Sequence[str],
    n_words: int,
    n_random: int,
    rng,
    exclude=(),
    vocab_people: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """Dictionary words / word fragments (6-10 letters) and random letter strings (8 letters) that at
    least 70% of `people` can type with disjoint Session-1 occurrences and Session-2 coverage."""
    def cov_all(w: str) -> bool:
        return sum(covered_disjoint_s1(tabs, p, w) for p in people) >= max(6, int(0.7 * len(people)))

    allowed_vocab = set(vocab_people) if vocab_people is not None else None
    vocab = Counter()
    for (p, _), ev in raw_sessions.items():
        if allowed_vocab is not None and p not in allowed_vocab:
            continue
        cur = []
        for label, _, _ in list(ev) + [("Space", 0, 0)]:
            k, _ = kind_of(label)
            if k == "char" and len(label) == 1 and label.isalpha():
                cur.append(label.lower())
            else:
                if 6 <= len(cur) <= 10:
                    vocab["".join(cur)] += 1
                cur = []
    cands = [w for w, c in vocab.most_common(3000) if c >= 3 and w not in exclude]
    rng.shuffle(cands)
    words = []
    for w in cands:
        if cov_all(w):
            words.append(w)
        if len(words) >= n_words:
            break
    ok_pair = {
        a + b: sum(covered_disjoint_s1(tabs, p, a + b) for p in people) >= 0.75 * len(people)
        for a in LETTERS
        for b in LETTERS
    }
    strings, tries = [], 0
    while len(strings) < n_random and tries < 100000:
        tries += 1
        w = LETTERS[rng.integers(26)]
        while len(w) < 8:
            nxt = [b for b in LETTERS if ok_pair[w[-1] + b] and b != w[-1]]
            if not nxt:
                break
            w += nxt[rng.integers(len(nxt))]
        if len(w) == 8 and w not in vocab and w not in strings and w not in exclude and cov_all(w):
            strings.append(w)
    return {"word": words, "random": strings}


def typing_tensors(word: str, typings: Sequence[Tuple], bounds, mean, std, touch: bool, touch_hold: float) -> np.ndarray:
    """Convert unclipped (raw_holds, raw_dds[, occ_ids]) attempts into scaled 17-channel tensors.
    Raw UD = raw_DD - raw_H_curr is computed BEFORE clipping H, DD, UD independently."""
    raw_h = np.stack([t[0] for t in typings]).astype(np.float32)
    raw_d = np.stack([t[1] for t in typings]).astype(np.float32)
    c_d = np.clip(raw_d, *bounds["DD"]).astype(np.float32)
    kb = key_channel_matrix(pairs_from_codes([code_for_label(c) for c in word]))
    if touch:
        h2, u2 = touchify_timings(raw_h, raw_d, touch_hold, bounds)
        seq = extract_sequences(h2, c_d, u2, kb, np.zeros(len(c_d), dtype=np.float32))
    else:
        raw_u = (raw_d - raw_h[:, :-1]).astype(np.float32)
        c_h = np.clip(raw_h, *bounds["H"]).astype(np.float32)
        c_u = np.clip(raw_u, *bounds["UD"]).astype(np.float32)
        seq = extract_sequences(c_h, c_d, c_u, kb)
    return apply_channel_scaler(seq, mean, std)


def raw_similarity(Q: np.ndarray, tmpl: Dict) -> np.ndarray:
    return 0.65 * (Q @ tmpl["centroid"]) + 0.35 * np.max(Q @ tmpl["exemplars"].T, axis=1)


def as_norm_scores(Q: np.ndarray, tmpl: Dict, cohort: np.ndarray, top_m: int = 12) -> np.ndarray:
    raw = raw_similarity(Q, tmpl)
    u = np.sort(cohort @ tmpl["centroid"])[-top_m:]
    q = np.sort(Q @ cohort.T, axis=1)[:, -top_m:]
    z = (raw - 0.5 * (u.mean() + q.mean(1))) / (0.5 * (u.std() + 0.05 + q.std(1) + 0.05))
    return 1.0 / (1.0 + np.exp(-0.85 * (z - 0.15)))


def string_trials(
    raw_sessions,
    tabs,
    people,
    strings: Dict[str, List[str]],
    embed9,
    embed17,
    bounds,
    mean,
    std,
    touch_hold,
    cohorts9: Dict[str, np.ndarray],
    rng,
    k_enroll: int = 4,
) -> List[Dict]:
    """One row per attempt with raw9 / raw17 / deployed AS-Norm scores (keyboard + simulated touch),
    enforcing strictly disjoint Session-1 occ_ids between enroll and same-day genuine queries."""
    rows = []
    for kind, wl in strings.items():
        for w in wl:
            ppl = [p for p in people if covered_disjoint_s1(tabs, p, w)]
            if len(ppl) < 2:
                continue
            typ = {}
            enr_occ_map: Dict[str, List[int]] = {}
            for p in ppl:
                enr_tab, same_tab = partition_tab_disjoint(tabs[(p, 1)], w, rng)
                typ[(p, "enroll")] = [synthesise(enr_tab, w, rng) for _ in range(k_enroll)]
                typ[(p, "same")] = [synthesise(same_tab, w, rng) for _ in range(3)]
                typ[(p, "cross")] = [synthesise(tabs[(p, 2)], w, rng) for _ in range(3)]
                typ[(p, "imp")] = [synthesise(tabs[(p, int(rng.integers(1, 3)))], w, rng) for _ in range(2)]

                enr_occ_set = {oid for _, _, oids in typ[(p, "enroll")] for oid in oids}
                same_occ_set = {oid for _, _, oids in typ[(p, "same")] for oid in oids}
                if not enr_occ_set.isdisjoint(same_occ_set):
                    raise AssertionError(f"Non-disjoint Session-1 occ_ids detected for {p} on {w}")
                enr_occ_map[p] = sorted(enr_occ_set)

            keys = [(k, i) for k, v in typ.items() for i in range(len(v))]
            idx = defaultdict(list)
            for j, k in enumerate(keys):
                idx[k[0]].append(j)
            for mode in ("kb", "touch"):
                T = typing_tensors(w, [typ[k][i] for k, i in keys], bounds, mean, std, mode == "touch", touch_hold)
                E9, E17 = embed9(T).astype(np.float64), embed17(T).astype(np.float64)
                for p in ppl:
                    t9, t17 = make_template(E9[idx[(p, "enroll")]]), make_template(E17[idx[(p, "enroll")]])
                    for q, tag, lab in [(p, "same", 1), (p, "cross", 1)] + [(q, "imp", 0) for q in ppl if q != p]:
                        J = idx[(q, tag)]
                        r9, r17 = raw_similarity(E9[J], t9), raw_similarity(E17[J], t17)
                        dep = as_norm_scores(E9[J], t9, cohorts9[mode])
                        for j in range(len(J)):
                            q_oids = typ[(q, tag)][j][2]
                            rows.append(
                                {
                                    "kind": kind,
                                    "word": w,
                                    "mode": mode,
                                    "target": p,
                                    "query": q,
                                    "tag": tag,
                                    "label": lab,
                                    "raw9": float(r9[j]),
                                    "raw17": float(r17[j]),
                                    "deployed": float(dep[j]),
                                    "enroll_occ_ids": enr_occ_map[p],
                                    "query_occ_ids": list(q_oids),
                                }
                            )
    return rows


def synthesis_metadata(
    tabs: Dict,
    people: Sequence[str],
    strings: Dict[str, List[str]],
    rows: Sequence[Dict],
    k_enroll: int = 4,
) -> Dict:
    """Audit metadata for disjoint occ_id password synthesis."""
    active_people = [p for p in people if (p, 1) in tabs and (p, 2) in tabs]
    excluded_pairs = 0
    generated_attempts = 0
    for wl in strings.values():
        for w in wl:
            ppl = [p for p in active_people if covered_disjoint_s1(tabs, p, w)]
            excluded_pairs += len(active_people) - len(ppl)
            if len(ppl) >= 2:
                generated_attempts += len(ppl) * (k_enroll + 3 + 3 + 2)
    same_rows = [r for r in rows if r.get("tag") == "same" and "enroll_occ_ids" in r and "query_occ_ids" in r]
    disjoint_ok = all(set(r["enroll_occ_ids"]).isdisjoint(r["query_occ_ids"]) for r in same_rows)
    return {
        "disjoint_occ_enforced": bool(disjoint_ok),
        "same_day_rows_verified": int(len(same_rows)),
        "excluded_person_string_pairs": int(excluded_pairs),
        "total_generated_attempts": int(generated_attempts),
        "generated_attempts": int(generated_attempts),
    }


def calibrate_dual(val_rows: List[Dict], frr_target: float = 0.10) -> Dict[str, Dict[str, float]]:
    out = {}
    for mode in ("kb", "touch"):
        r = [x for x in val_rows if x["mode"] == mode]
        y = np.array([x["label"] for x in r])
        r9, r17 = np.array([x["raw9"] for x in r]), np.array([x["raw17"] for x in r])
        mu9, sd9 = float(r9[y == 0].mean()), float(r9[y == 0].std())
        mu17, sd17 = float(r17[y == 0].mean()), float(r17[y == 0].std())
        z = (r9 - mu9) / sd9 + (r17 - mu17) / sd17
        out[mode] = {
            "mu9": round(mu9, 5),
            "sd9": round(sd9, 5),
            "mu17": round(mu17, 5),
            "sd17": round(sd17, 5),
            "tau": round(float(np.quantile(z[y == 1], frr_target)), 4),
        }
    return out


def dual_z(raw9, raw17, prm: Dict[str, float]):
    return (np.asarray(raw9) - prm["mu9"]) / prm["sd9"] + (np.asarray(raw17) - prm["mu17"]) / prm["sd17"]


def _eer(y, s) -> float:
    f, t, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs(f - (1 - t))))
    return float((f[i] + 1 - t[i]) / 2)


def evaluate_strings(rows: List[Dict], calib: Dict, deployed_thr: Dict[str, float], speed: Dict[str, float], k_enroll: int = 4) -> Dict:
    """Test-people metrics: deployed AS-Norm (deployed threshold) vs dual score (calibrated tau)."""
    out = {}
    for mode in ("kb", "touch"):
        r = [x for x in rows if x["mode"] == mode]
        y = np.array([x["label"] for x in r])
        dep = np.array([x["deployed"] for x in r])
        z = dual_z([x["raw9"] for x in r], [x["raw17"] for x in r], calib[mode])
        tags = np.array([x["tag"] for x in r])
        kinds = np.array([x["kind"] for x in r])
        pair_sw = {(x["word"], x["target"]) for x in r}
        gen_attempts = int(len(pair_sw) * (k_enroll + 3 + 3 + 2))
        res = {
            "num_genuine": int(y.sum()),
            "num_impostor": int((y == 0).sum()),
            "num_strings": len({x["word"] for x in r}),
            "num_people": len({x["target"] for x in r}),
            "generated_attempts": gen_attempts,
        }
        for name, s, thr in (("deployed_as_norm", dep, deployed_thr[mode]), ("dual_raw", z, calib[mode]["tau"])):
            acc = s >= thr
            d = {
                "roc_auc": round(float(roc_auc_score(y, s)), 4),
                "eer": round(_eer(y, s), 4),
                "far": round(float(acc[y == 0].mean()), 4),
                "frr": round(float((~acc[y == 1]).mean()), 4),
                "frr_same_day": round(float((~acc[(y == 1) & (tags == "same")]).mean()), 4),
                "frr_cross_day": round(float((~acc[(y == 1) & (tags == "cross")]).mean()), 4),
                "far_words": round(float(acc[(y == 0) & (kinds == "word")].mean()), 4),
                "far_random": round(float(acc[(y == 0) & (kinds == "random")].mean()), 4),
            }
            dsp = np.array([abs(np.log(speed[x["target"]] / speed[x["query"]])) * 100 for x in r])
            for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 35), (35, 1000)):
                sel = (y == 0) & (dsp >= lo) & (dsp < hi)
                d[f"far_speed_diff_{lo}_{hi if hi < 1000 else 'up'}pct"] = round(float(acc[sel].mean()), 4) if sel.any() else None
            res[name] = d
        out[mode] = res
    return out


def person_speed(tabs, people) -> Dict[str, float]:
    return {
        p: float(np.median([x[1] for s in (1, 2) if (p, s) in tabs for v in tabs[(p, s)].values() for x in v]) * 1000)
        for p in people
    }


def adaptation_eval(
    h: np.ndarray,
    d: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    R: np.ndarray,
    word: str,
    embed9,
    embed17,
    bounds,
    mean,
    std,
    touch_hold,
    calib,
    deployed_thr,
    cohorts9,
    margins: Sequence[float],
    rng,
    mode: str,
    k_enroll: int = 4,
    keep: int = 12,
) -> Dict:
    """Fixed text typed 100x per session (KeyRecs): enrol on the first k attempts, then walk through the
    person's attempts in order; accepted attempts with z >= tau + margin join the template (last `keep`).
    `h` and `d` should be UNCLIPPED raw holds and DD intervals so raw UD is computed before clipping."""
    T = typing_tensors(word, list(zip(h, d)), bounds, mean, std, mode == "touch", touch_hold)
    E9, E17 = embed9(T).astype(np.float64), embed17(T).astype(np.float64)
    prm = calib[mode]
    buckets = {
        "s1_reps_5_14": (1, 4, 14),
        "s1_reps_51_100": (1, 50, 100),
        "s2_reps_1_10": (2, 0, 10),
        "s2_reps_91_100": (2, 90, 100),
    }
    out = {}
    for policy in ["deployed", "dual_no_adapt"] + [f"dual_adapt_margin_{m:g}" for m in margins]:
        frr = {b: [] for b in buckets}
        far = []
        for p in np.unique(P):
            own = np.where(P == p)[0]
            order = own[np.lexsort((R[own], S[own]))]
            enr = order[:k_enroll]
            ex9, ex17 = list(E9[enr]), list(E17[enr])
            dec = {}
            for i in order[k_enroll:]:
                t9, t17 = make_template(np.array(ex9[-keep:])), make_template(np.array(ex17[-keep:]))
                if policy == "deployed":
                    dec[i] = float(as_norm_scores(E9[i : i + 1], make_template(E9[enr]), cohorts9[mode])[0]) >= deployed_thr[mode]
                    continue
                z = float(dual_z(raw_similarity(E9[i : i + 1], t9), raw_similarity(E17[i : i + 1], t17), prm)[0])
                dec[i] = z >= prm["tau"]
                if policy.startswith("dual_adapt") and z >= prm["tau"] + float(policy.rsplit("_", 1)[1]):
                    ex9.append(E9[i])
                    ex17.append(E17[i])
            for b, (s, lo, hi) in buckets.items():
                frr[b] += [not dec[i] for i in order if S[i] == s and lo <= R[i] < hi and i in dec]
            others = np.where((P != p) & (S == 2))[0]
            oi = rng.choice(others, size=min(40, len(others)), replace=False)
            if policy == "deployed":
                far += list(as_norm_scores(E9[oi], make_template(E9[enr]), cohorts9[mode]) >= deployed_thr[mode])
            else:
                t9, t17 = make_template(np.array(ex9[-keep:])), make_template(np.array(ex17[-keep:]))
                far += list(dual_z(raw_similarity(E9[oi], t9), raw_similarity(E17[oi], t17), prm) >= prm["tau"])
        out[policy] = {
            **{f"frr_{b}": round(float(np.mean(v)), 4) for b, v in frr.items()},
            "far": round(float(np.mean(far)), 4),
        }
    return out


# ---------------------------------------------------------------------------------------
# Length-aware calibration
# ---------------------------------------------------------------------------------------
CALIB_LENGTHS = (8, 10, 12, 14, 16, 20)


def strings_of_length(
    tabs,
    people: Sequence[str],
    length: int,
    n: int,
    rng,
    exclude=(),
    min_share: float = 0.7,
) -> List[str]:
    """Random letter strings of an exact length built from digraphs most people typed with disjoint S1 pools."""
    pair_ok = {
        a + b: sum(covered_disjoint_s1(tabs, p, a + b) for p in people) >= 0.80 * len(people)
        for a in LETTERS
        for b in LETTERS
    }
    out, tries = [], 0
    while len(out) < n and tries < 200000:
        tries += 1
        w = LETTERS[rng.integers(26)]
        while len(w) < length:
            nxt = [b for b in LETTERS if pair_ok[w[-1] + b] and b != w[-1]]
            if not nxt:
                break
            w += nxt[rng.integers(len(nxt))]
        if len(w) != length or w in out or w in exclude:
            continue
        if sum(covered_disjoint_s1(tabs, p, w) for p in people) >= max(6, int(min_share * len(people))):
            out.append(w)
    return out


def calibrate_by_length(rows_by_len: Dict[int, List[Dict]], frr_target: float = 0.10) -> List[Dict]:
    table = []
    for L in sorted(rows_by_len):
        c = calibrate_dual(rows_by_len[L], frr_target)
        table.append({"length": int(L), "keyboard": c["kb"], "touch": c["touch"]})
    return table


def params_for_length(table: List[Dict], length: int, mode: str) -> Dict[str, float]:
    """Linear interpolation between calibrated lengths (clamped). Mirrors features.js dualParamsForLength."""
    key = "touch" if mode == "touch" else "keyboard"
    rows = sorted(table, key=lambda r: r["length"])
    if length <= rows[0]["length"]:
        return dict(rows[0][key])
    if length >= rows[-1]["length"]:
        return dict(rows[-1][key])
    for lo, hi in zip(rows, rows[1:]):
        if lo["length"] <= length <= hi["length"]:
            f = (length - lo["length"]) / (hi["length"] - lo["length"])
            return {k: lo[key][k] + f * (hi[key][k] - lo[key][k]) for k in lo[key]}
    return dict(rows[-1][key])


def evaluate_lengths(rows_by_len: Dict[int, List[Dict]], fixed: Dict, table: List[Dict]) -> Dict:
    """Test people: fixed (6-10 char) calibration vs length-aware calibration, per length and mode."""
    out = {}
    for L, rows in sorted(rows_by_len.items()):
        out[str(L)] = {}
        for mode in ("kb", "touch"):
            r = [x for x in rows if x["mode"] == mode]
            y = np.array([x["label"] for x in r])
            r9, r17 = [x["raw9"] for x in r], [x["raw17"] for x in r]
            res = {"num_genuine": int(y.sum()), "num_impostor": int((y == 0).sum())}
            for name, prm in (("fixed_6_10", fixed[mode]), ("length_aware", params_for_length(table, L, mode))):
                z = dual_z(r9, r17, prm)
                acc = z >= prm["tau"]
                res[name] = {
                    "far": round(float(acc[y == 0].mean()), 4),
                    "frr": round(float((~acc[y == 1]).mean()), 4),
                }
            z = dual_z(r9, r17, params_for_length(table, L, mode))
            res["roc_auc"] = round(float(roc_auc_score(y, z)), 4)
            res["eer"] = round(_eer(y, z), 4)
            out[str(L)][mode] = res
    return out


# ---------------------------------------------------------------------------------------
# Capital letters: Shift vs CapsLock habit
# ---------------------------------------------------------------------------------------
def capital_method_rates(raw_sessions, min_capitals: int = 5) -> Dict[Tuple[str, int], float]:
    """{(person, session): share of capitals typed with CapsLock (vs Shift)}."""
    counts = defaultdict(lambda: [0, 0])
    for key, ev in raw_sessions.items():
        caps_on = False
        for i in range(len(ev) - 1):
            label = ev[i][0]
            if label == "CapsLock":
                caps_on = not caps_on
            nxt = ev[i + 1][0]
            if len(nxt) == 1 and nxt.isalpha() and nxt.isupper():
                if label == "Shift":
                    counts[key][1] += 1
                elif caps_on:
                    counts[key][0] += 1
    return {k: v[0] / sum(v) for k, v in counts.items() if sum(v) >= min_capitals}


def capital_llr(q_caps: bool, n_caps: int, n_total: int, caps_share: float, alpha: float = 2.0) -> float:
    """log P(method | this person's enrolment, Beta-smoothed) - log P(method | population)."""
    r = (n_caps + alpha * caps_share) / (n_total + alpha)
    return float(np.log((r if q_caps else 1 - r) / (caps_share if q_caps else 1 - caps_share)))


def capital_trials(
    tabs,
    rates,
    people,
    length: int,
    n_words: int,
    embed9,
    embed17,
    bounds,
    mean,
    std,
    touch_hold,
    prm: Dict[str, float],
    rng,
    k_enroll: int = 4,
) -> List[Tuple[int, float, bool, int]]:
    """Keyboard passwords of `length` letters whose first letter is a capital; each attempt's method is
    drawn from that person's own CapsLock rate in that session, with disjoint Session-1 enroll vs query pools.
    Rows: (label, z, query_caps, enrol_caps)."""
    people = [p for p in people if (p, 1) in tabs and (p, 2) in tabs and (p, 1) in rates and (p, 2) in rates]
    words = strings_of_length(tabs, people, length, n_words, rng)
    rows = []
    for w in words:
        ppl = [p for p in people if covered_disjoint_s1(tabs, p, w)]
        if len(ppl) < 2:
            continue
        typ = {}
        for p in ppl:
            enr_tab, same_tab = partition_tab_disjoint(tabs[(p, 1)], w, rng)
            enr_att = [(synthesise(enr_tab, w, rng), bool(rng.random() < rates[(p, 1)])) for _ in range(k_enroll)]
            gen_s1 = [(synthesise(same_tab, w, rng), bool(rng.random() < rates[(p, 1)])) for _ in range(2)]
            gen_s2 = [(synthesise(tabs[(p, 2)], w, rng), bool(rng.random() < rates[(p, 2)])) for _ in range(2)]
            imp_s2 = [(synthesise(tabs[(p, 2)], w, rng), bool(rng.random() < rates[(p, 2)])) for _ in range(2)]

            enr_occ = {oid for (_, _, oids), _ in enr_att for oid in oids}
            same_occ = {oid for (_, _, oids), _ in gen_s1 for oid in oids}
            if not enr_occ.isdisjoint(same_occ):
                raise AssertionError(f"Non-disjoint capital_trials Session-1 occ_ids for {p} on {w}")
            typ[p] = {"e": enr_att, "g": gen_s1 + gen_s2, "i": imp_s2}

        keys = [(p, k, i) for p in ppl for k in ("e", "g", "i") for i in range(len(typ[p][k]))]
        T = typing_tensors(w, [typ[p][k][i][0] for p, k, i in keys], bounds, mean, std, False, touch_hold)
        E9, E17 = embed9(T).astype(np.float64), embed17(T).astype(np.float64)
        ix = defaultdict(list)
        for j, kk in enumerate(keys):
            ix[kk[:2]].append(j)
        for p in ppl:
            t9, t17 = make_template(E9[ix[(p, "e")]]), make_template(E17[ix[(p, "e")]])
            n_caps = sum(c for _, c in typ[p]["e"])
            for q in ppl:
                tag = "g" if q == p else "i"
                J = ix[(q, tag)]
                z = dual_z(raw_similarity(E9[J], t9), raw_similarity(E17[J], t17), prm)
                for zz, (_, c) in zip(z, typ[q][tag]):
                    rows.append((int(q == p), float(zz), bool(c), int(n_caps)))
    return rows


def capital_fusion(
    val_rows,
    test_rows,
    caps_share: float,
    base_tau: float,
    k_enroll: int = 4,
    frr_target: float = 0.10,
    grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> Dict:
    """Select (weight, tau_offset) strictly on Validation where decision threshold is `base_tau + tau_offset`
    (matching biometrics.js:261 `prm.tau += cm.tau_offset`), and evaluate Test at exact deployed thresholds."""
    def arr(rows, w):
        y = np.array([r[0] for r in rows])
        s = np.array([r[1] + w * capital_llr(r[2], r[3], k_enroll, caps_share) for r in rows])
        return y, s

    def at(rows, w, thr):
        y, s = arr(rows, w)
        return float(np.mean(s[y == 0] >= thr)), float(np.mean(s[y == 1] < thr))

    base_tau_r = round(float(base_tau), 4)
    offsets: Dict[float, float] = {}
    thr: Dict[float, float] = {}
    val_far: Dict[float, float] = {}
    val_frr: Dict[float, float] = {}
    for w in grid:
        yv, sv = arr(val_rows, w)
        if w == 0.0:
            off_w = 0.0
        else:
            off_w = round(float(np.quantile(sv[yv == 1], frr_target) - base_tau_r), 4)
        offsets[w] = off_w
        thr[w] = round(base_tau_r + off_w, 4)
        far_w, frr_w = at(val_rows, w, thr[w])
        val_far[w] = far_w
        val_frr[w] = frr_w

    best = min(grid, key=lambda w: (val_far[w], w))
    tau_offset = offsets[best]
    deployed_thr_no_cap = base_tau_r
    deployed_thr_with_cap = round(base_tau_r + tau_offset, 4)

    far0, frr0 = at(test_rows, 0.0, deployed_thr_no_cap)
    far1, frr1 = at(test_rows, best, deployed_thr_with_cap)
    return {
        "weight": float(best),
        "tau_offset": tau_offset,
        "caps_share": round(float(caps_share), 4),
        "deployed_threshold_no_capitals": deployed_thr_no_cap,
        "deployed_threshold_with_capitals": deployed_thr_with_cap,
        "validation_far_by_weight": {str(w): round(val_far[w], 4) for w in grid},
        "validation_frr_by_weight": {str(w): round(val_frr[w], 4) for w in grid},
        "test_without": {"far": round(far0, 4), "frr": round(frr0, 4)},
        "test_with": {"far": round(far1, 4), "frr": round(frr1, 4)},
        "num_test_genuine": int(sum(r[0] for r in test_rows)),
        "num_test_impostor": int(sum(1 - r[0] for r in test_rows)),
    }
