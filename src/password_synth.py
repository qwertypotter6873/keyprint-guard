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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from behavior_profile import kind_of
from capture_sim import make_template
from keyboard_layout import code_for_label, key_channel_matrix, pairs_from_codes
from preprocessing import apply_channel_scaler, extract_sequences, touchify_timings

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DISPLAY_SLOPE = 1.0  # web similarity = sigmoid(slope * (z - tau)); 0.5 is the decision point


def digraph_tables(raw_sessions, people: Sequence[str], max_dd: float = 1.5) -> Dict:
    """{(person, session): {(a, b): [(t, dd, hold_a, hold_b), ...]}} from clean letter->letter pairs."""
    tabs = {}
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
                    tab[(prev[0], label.lower())].append((prev[1], t - prev[1], prev[2], h))
                prev = (label.lower(), t, h) if ok else None
            tabs[(p, s)] = dict(tab)
    return tabs


def covered(tab: Dict, word: str) -> bool:
    return all(len(tab.get((a, b), ())) >= 1 for a, b in zip(word, word[1:]))


def synth(tab: Dict, word: str, rng, window: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
    """One typing of `word`: holds (n,) and digraph intervals (n-1,) in seconds."""
    pairs = list(zip(word, word[1:]))
    a0 = pairs[rng.integers(len(pairs))]
    anchor = tab[a0][rng.integers(len(tab[a0]))][0]
    holds, dds = [], []
    for i, pr in enumerate(pairs):
        inst = tab[pr]
        ts = np.array([x[0] for x in inst])
        near = np.where(np.abs(ts - anchor) <= window)[0]
        if len(near) == 0:
            near = np.argsort(np.abs(ts - anchor))[:2]
        x = inst[int(rng.choice(near))]
        dds.append(x[1])
        holds.append(x[2])
        if i == len(pairs) - 1:
            holds.append(x[3])
    return np.array(holds, dtype=np.float32), np.array(dds, dtype=np.float32)


def choose_strings(raw_sessions, tabs, people: Sequence[str], n_words: int, n_random: int, rng, exclude=()) -> Dict[str, List[str]]:
    """Dictionary words / word fragments (6-10 letters) and random letter strings (8 letters) that at
    least 70% of `people` can type with real instances of every digraph in both sessions."""
    def cov_all(w):
        return sum(all(covered(tabs[(p, s)], w) for s in (1, 2)) for p in people) >= max(6, int(0.7 * len(people)))

    vocab = Counter()
    for (p, s), ev in raw_sessions.items():
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
    ok_pair = {a + b: sum(covered(tabs[(p, s)], a + b) for p in people for s in (1, 2)) >= 0.8 * 2 * len(people) for a in LETTERS for b in LETTERS}
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


def typing_tensors(word: str, typings: Sequence[Tuple[np.ndarray, np.ndarray]], bounds, mean, std, touch: bool, touch_hold: float) -> np.ndarray:
    h = np.clip(np.stack([t[0] for t in typings]), *bounds["H"]).astype(np.float32)
    d = np.clip(np.stack([t[1] for t in typings]), *bounds["DD"]).astype(np.float32)
    kb = key_channel_matrix(pairs_from_codes([code_for_label(c) for c in word]))
    if touch:
        h2, u2 = touchify_timings(h, d, touch_hold, bounds)
        seq = extract_sequences(h2, d, u2, kb, np.zeros(len(d), dtype=np.float32))
    else:
        seq = extract_sequences(h, d, np.clip(d - h[:, :-1], *bounds["UD"]), kb)
    return apply_channel_scaler(seq, mean, std)


def raw_similarity(Q: np.ndarray, tmpl: Dict) -> np.ndarray:
    return 0.65 * (Q @ tmpl["centroid"]) + 0.35 * np.max(Q @ tmpl["exemplars"].T, axis=1)


def as_norm_scores(Q: np.ndarray, tmpl: Dict, cohort: np.ndarray, top_m: int = 12) -> np.ndarray:
    raw = raw_similarity(Q, tmpl)
    u = np.sort(cohort @ tmpl["centroid"])[-top_m:]
    q = np.sort(Q @ cohort.T, axis=1)[:, -top_m:]
    z = (raw - 0.5 * (u.mean() + q.mean(1))) / (0.5 * (u.std() + 0.05 + q.std(1) + 0.05))
    return 1.0 / (1.0 + np.exp(-0.85 * (z - 0.15)))


def string_trials(raw_sessions, tabs, people, strings: Dict[str, List[str]], embed9, embed17, bounds, mean, std, touch_hold,
                  cohorts9: Dict[str, np.ndarray], rng, k_enroll: int = 4) -> List[Dict]:
    """One row per attempt with raw9 / raw17 / deployed AS-Norm scores (keyboard + simulated touch)."""
    rows = []
    for kind, wl in strings.items():
        for w in wl:
            ppl = [p for p in people if all(covered(tabs[(p, s)], w) for s in (1, 2))]
            typ = {}
            for p in ppl:
                typ[(p, "enroll")] = [synth(tabs[(p, 1)], w, rng) for _ in range(k_enroll)]
                typ[(p, "same")] = [synth(tabs[(p, 1)], w, rng) for _ in range(3)]
                typ[(p, "cross")] = [synth(tabs[(p, 2)], w, rng) for _ in range(3)]
                typ[(p, "imp")] = [synth(tabs[(p, int(rng.integers(1, 3)))], w, rng) for _ in range(2)]
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
                            rows.append({"kind": kind, "word": w, "mode": mode, "target": p, "query": q, "tag": tag, "label": lab,
                                         "raw9": float(r9[j]), "raw17": float(r17[j]), "deployed": float(dep[j])})
    return rows


def calibrate_dual(val_rows: List[Dict], frr_target: float = 0.10) -> Dict[str, Dict[str, float]]:
    out = {}
    for mode in ("kb", "touch"):
        r = [x for x in val_rows if x["mode"] == mode]
        y = np.array([x["label"] for x in r])
        r9, r17 = np.array([x["raw9"] for x in r]), np.array([x["raw17"] for x in r])
        mu9, sd9 = float(r9[y == 0].mean()), float(r9[y == 0].std())
        mu17, sd17 = float(r17[y == 0].mean()), float(r17[y == 0].std())
        z = (r9 - mu9) / sd9 + (r17 - mu17) / sd17
        out[mode] = {"mu9": round(mu9, 5), "sd9": round(sd9, 5), "mu17": round(mu17, 5), "sd17": round(sd17, 5),
                     "tau": round(float(np.quantile(z[y == 1], frr_target)), 4)}
    return out


def dual_z(raw9, raw17, prm: Dict[str, float]):
    return (np.asarray(raw9) - prm["mu9"]) / prm["sd9"] + (np.asarray(raw17) - prm["mu17"]) / prm["sd17"]


def _eer(y, s) -> float:
    f, t, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs(f - (1 - t))))
    return float((f[i] + 1 - t[i]) / 2)


def evaluate_strings(rows: List[Dict], calib: Dict, deployed_thr: Dict[str, float], speed: Dict[str, float]) -> Dict:
    """Test-people metrics: deployed AS-Norm (deployed threshold) vs dual score (calibrated tau)."""
    out = {}
    for mode in ("kb", "touch"):
        r = [x for x in rows if x["mode"] == mode]
        y = np.array([x["label"] for x in r])
        dep = np.array([x["deployed"] for x in r])
        z = dual_z([x["raw9"] for x in r], [x["raw17"] for x in r], calib[mode])
        tags = np.array([x["tag"] for x in r])
        kinds = np.array([x["kind"] for x in r])
        res = {"num_genuine": int(y.sum()), "num_impostor": int((y == 0).sum()), "num_strings": len({x["word"] for x in r}),
               "num_people": len({x["target"] for x in r})}
        for name, s, thr in (("deployed_as_norm", dep, deployed_thr[mode]), ("dual_raw", z, calib[mode]["tau"])):
            acc = s >= thr
            d = {"roc_auc": round(float(roc_auc_score(y, s)), 4), "eer": round(_eer(y, s), 4),
                 "far": round(float(acc[y == 0].mean()), 4), "frr": round(float((~acc[y == 1]).mean()), 4),
                 "frr_same_day": round(float((~acc[(y == 1) & (tags == "same")]).mean()), 4),
                 "frr_cross_day": round(float((~acc[(y == 1) & (tags == "cross")]).mean()), 4),
                 "far_words": round(float(acc[(y == 0) & (kinds == "word")].mean()), 4),
                 "far_random": round(float(acc[(y == 0) & (kinds == "random")].mean()), 4)}
            # impostor pass rate by typing-speed difference between impostor and target
            dsp = np.array([abs(np.log(speed[x["target"]] / speed[x["query"]])) * 100 for x in r])
            for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 35), (35, 1000)):
                sel = (y == 0) & (dsp >= lo) & (dsp < hi)
                d[f"far_speed_diff_{lo}_{hi if hi < 1000 else 'up'}pct"] = round(float(acc[sel].mean()), 4) if sel.any() else None
            res[name] = d
        out[mode] = res
    return out


def person_speed(tabs, people) -> Dict[str, float]:
    return {p: float(np.median([x[1] for s in (1, 2) if (p, s) in tabs for v in tabs[(p, s)].values() for x in v]) * 1000) for p in people}


def adaptation_eval(h: np.ndarray, d: np.ndarray, P: np.ndarray, S: np.ndarray, R: np.ndarray, word: str, embed9, embed17,
                    bounds, mean, std, touch_hold, calib, deployed_thr, cohorts9, margins: Sequence[float], rng, mode: str,
                    k_enroll: int = 4, keep: int = 12) -> Dict:
    """Fixed text typed 100x per session (KeyRecs): enrol on the first k attempts, then walk through the
    person's attempts in order; accepted attempts with z >= tau + margin join the template (last `keep`).
    Returns FRR by repetition bucket and impostor FAR against the final template, per margin."""
    T = typing_tensors(word, list(zip(h, d)), bounds, mean, std, mode == "touch", touch_hold)
    E9, E17 = embed9(T).astype(np.float64), embed17(T).astype(np.float64)
    prm = calib[mode]
    buckets = {"s1_reps_5_14": (1, 4, 14), "s1_reps_51_100": (1, 50, 100), "s2_reps_1_10": (2, 0, 10), "s2_reps_91_100": (2, 90, 100)}
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
                    dec[i] = float(as_norm_scores(E9[i:i + 1], make_template(E9[enr]), cohorts9[mode])[0]) >= deployed_thr[mode]
                    continue
                z = float(dual_z(raw_similarity(E9[i:i + 1], t9), raw_similarity(E17[i:i + 1], t17), prm)[0])
                dec[i] = z >= prm["tau"]
                if policy.startswith("dual_adapt") and z >= prm["tau"] + float(policy.rsplit("_", 1)[1]):
                    ex9.append(E9[i]); ex17.append(E17[i])
            for b, (s, lo, hi) in buckets.items():
                frr[b] += [not dec[i] for i in order if S[i] == s and lo <= R[i] < hi and i in dec]
            others = np.where((P != p) & (S == 2))[0]
            oi = rng.choice(others, size=min(40, len(others)), replace=False)
            if policy == "deployed":
                far += list(as_norm_scores(E9[oi], make_template(E9[enr]), cohorts9[mode]) >= deployed_thr[mode])
            else:
                t9, t17 = make_template(np.array(ex9[-keep:])), make_template(np.array(ex17[-keep:]))
                far += list(dual_z(raw_similarity(E9[oi], t9), raw_similarity(E17[oi], t17), prm) >= prm["tau"])
        out[policy] = {**{f"frr_{b}": round(float(np.mean(v)), 4) for b, v in frr.items()}, "far": round(float(np.mean(far)), 4)}
    return out


# ---------------------------------------------------------------------------------------
# Length-aware calibration: the dual z grows with password length, so a single tau fitted on
# 6-10 character strings is too lenient for 12+ characters (impostors pass at the same rate while
# only the owner's rejections fall). tau / mu / sigma are fitted per length and interpolated.
# ---------------------------------------------------------------------------------------
CALIB_LENGTHS = (8, 10, 12, 14, 16, 20)


def strings_of_length(tabs, people: Sequence[str], length: int, n: int, rng, exclude=(), min_share: float = 0.7) -> List[str]:
    """Random letter strings of an exact length built from digraphs most people typed in both sessions."""
    pair_ok = {a + b: sum(covered(tabs[(p, s)], a + b) for p in people for s in (1, 2)) >= 0.85 * 2 * len(people)
               for a in LETTERS for b in LETTERS}
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
        if sum(all(covered(tabs[(p, s)], w) for s in (1, 2)) for p in people) >= max(6, int(min_share * len(people))):
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
                res[name] = {"far": round(float(acc[y == 0].mean()), 4), "frr": round(float((~acc[y == 1]).mean()), 4)}
            z = dual_z(r9, r17, params_for_length(table, L, mode))
            res["roc_auc"] = round(float(roc_auc_score(y, z)), 4)
            res["eer"] = round(_eer(y, z), 4)
            out[str(L)][mode] = res
    return out


# ---------------------------------------------------------------------------------------
# Capital letters: Shift vs CapsLock habit. In KeyRecs ~31% of capitals are typed with CapsLock and
# 99% of people keep the same habit across days, while Shift lead/release timing of a single capital
# is too noisy to help. The habit enters the password score as a predictive log-likelihood ratio.
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


def capital_trials(tabs, rates, people, length: int, n_words: int, embed9, embed17, bounds, mean, std, touch_hold,
                   prm: Dict[str, float], rng, k_enroll: int = 4) -> List[Tuple[int, float, bool, int]]:
    """Keyboard passwords of `length` letters whose first letter is a capital; each attempt's method is
    drawn from that person's own CapsLock rate in that session. Rows: (label, z, query_caps, enrol_caps)."""
    people = [p for p in people if (p, 1) in tabs and (p, 2) in tabs and (p, 1) in rates and (p, 2) in rates]
    words = strings_of_length(tabs, people, length, n_words, rng)
    rows = []
    for w in words:
        ppl = [p for p in people if all(covered(tabs[(p, s)], w) for s in (1, 2))]
        att = lambda p, s: (synth(tabs[(p, s)], w, rng), bool(rng.random() < rates[(p, s)]))  # noqa: E731
        typ = {p: {"e": [att(p, 1) for _ in range(k_enroll)], "g": [att(p, s) for s in (1, 1, 2, 2)], "i": [att(p, 2) for _ in range(2)]}
               for p in ppl}
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


def capital_fusion(val_rows, test_rows, caps_share: float, k_enroll: int = 4, frr_target: float = 0.10,
                   grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> Dict:
    """Weight w of z + w*LLR chosen on Validation (lowest FAR at the target FRR) and the tau offset that
    keeps the FRR there; Test FAR/FRR with and without the cue."""
    def arr(rows, w):
        y = np.array([r[0] for r in rows])
        s = np.array([r[1] + w * capital_llr(r[2], r[3], k_enroll, caps_share) for r in rows])
        return y, s

    def at(rows, w, thr):
        y, s = arr(rows, w)
        return float(np.mean(s[y == 0] >= thr)), float(np.mean(s[y == 1] < thr))

    thr = {}
    for w in grid:
        y, s = arr(val_rows, w)
        thr[w] = float(np.quantile(s[y == 1], frr_target))
    best = min(grid, key=lambda w: at(val_rows, w, thr[w])[0])
    far0, frr0 = at(test_rows, 0.0, thr[0.0])
    far1, frr1 = at(test_rows, best, thr[best])
    return {"weight": best, "tau_offset": round(thr[best] - thr[0.0], 4), "caps_share": round(caps_share, 4),
            "validation_far_by_weight": {str(w): round(at(val_rows, w, thr[w])[0], 4) for w in grid},
            "test_without": {"far": round(far0, 4), "frr": round(frr0, 4)},
            "test_with": {"far": round(far1, 4), "frr": round(frr1, 4)},
            "num_test_genuine": int(sum(r[0] for r in test_rows)), "num_test_impostor": int(sum(1 - r[0] for r in test_rows))}
