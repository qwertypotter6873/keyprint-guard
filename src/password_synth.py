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

import hashlib
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from behavior_profile import kind_of
from capture_sim import make_template
from keyboard_layout import code_for_label, key_channel_matrix, pairs_from_codes
from preprocessing import apply_channel_scaler, extract_sequences, touchify_timings

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DISPLAY_SLOPE = 1.0  # web similarity = sigmoid(slope * (z - tau)); 0.5 is the decision point
SourceKey = Tuple[str, int, int]  # participant, session, index in the source event stream
SOURCE_EVENT_SCOPE = "Session 1 enrollment and same-day genuine queries for each participant and string."
_RNG_SPLITS = {"val": 0, "test": 1}
_RNG_STAGES = {
    "word_selection": 0,
    "word_trials": 1,
    "length_selection": 2,
    "length_trials": 3,
    "capital_selection": 4,
    "capital_trials": 5,
    "adaptation_queries": 6,
    "fixed_comparison_queries": 7,
}


def password_rng(seed: int, split: str, stage: str, length: int = 0) -> np.random.Generator:
    """Provide a reproducible stream independent of other splits, stages and lengths."""
    return np.random.default_rng(np.random.SeedSequence([seed, _RNG_SPLITS[split], _RNG_STAGES[stage], length]))


class DigraphOccurrence(NamedTuple):
    time: float
    dd: float
    hold_a: float
    hold_b: float
    start: SourceKey  # also uniquely identifies the occurrence, which ends at the next event


def _end_key(start: SourceKey) -> SourceKey:
    person, session, index = start
    return person, session, index + 1


def digraph_tables(raw_sessions, people: Sequence[str], max_dd: float = 1.5) -> Dict:
    """Index clean consecutive letter pairs by their physical source events."""
    tabs = {}
    for p in people:
        for s in (1, 2):
            ev = raw_sessions.get((p, s))
            if not ev:
                continue
            tab = defaultdict(list)
            prev = None
            for i, (label, t, h) in enumerate(ev):
                kind, _ = kind_of(label)
                ok = kind == "char" and len(label) == 1 and label.isalpha() and 0.008 < h < 1.2
                if ok and prev is not None and 0.006 < t - prev[1] < max_dd:
                    tab[(prev[0], label.lower())].append(
                        DigraphOccurrence(prev[1], t - prev[1], prev[2], h, (p, s, i - 1))
                    )
                prev = (label.lower(), t, h) if ok else None
            tabs[(p, s)] = dict(tab)
    return tabs


def covered(tab: Dict, word: str, min_per_occ: int = 1) -> bool:
    need = Counter(zip(word, word[1:]))
    return all(len(tab.get(pr, ())) >= min_per_occ * cnt for pr, cnt in need.items())


def _partition_witnesses(tab: Dict, need: Counter, rng=None):
    """Find enough occurrences for every digraph on both sides by reserving their source events."""
    pairs = sorted(need, key=lambda pair: (len(tab.get(pair, ())) - 2 * need[pair], pair))
    if any(len(tab.get(pair, ())) < 2 * need[pair] for pair in pairs):
        return None
    choices = {pair: tab[pair] for pair in pairs}
    if rng is not None:
        choices = {pair: [instances[i] for i in rng.permutation(len(instances))] for pair, instances in choices.items()}
    groups = [(pair, role) for pair in pairs for role in (0, 1)]
    pools = [{pair: [] for pair in pairs} for _ in range(2)]
    occupied = [Counter(), Counter()]

    def reserve(group: int, offset: int) -> bool:
        if group == len(groups):
            return True
        pair, role = groups[group]
        pool = pools[role][pair]
        missing = need[pair] - len(pool)
        instances = choices[pair]
        keys, other = occupied[role], occupied[1 - role]
        for i in range(offset, len(instances) - missing + 1):
            occ = instances[i]
            start = occ.start[2]
            if other[start] or other[start + 1]:
                continue
            pool.append(occ)
            keys[start] += 1
            keys[start + 1] += 1
            next_group, next_offset = (group + 1, 0) if missing == 1 else (group, i + 1)
            if reserve(next_group, next_offset):
                return True
            keys[start] -= 1
            keys[start + 1] -= 1
            pool.pop()
        return False

    return (pools, occupied) if reserve(0, 0) else None


def _coverage_exclusion(tabs: Dict, person: str, word: str) -> Optional[str]:
    if (person, 1) not in tabs or (person, 2) not in tabs:
        return "missing_session"
    tab1 = tabs[(person, 1)]
    need = Counter(zip(word, word[1:]))
    if not covered(tabs[(person, 2)], word) or any(len(tab1.get(pair, ())) < 2 * count for pair, count in need.items()):
        return "insufficient_digraph_occurrences"

    # A single cut proves eligibility for common digraphs. Search the remaining cases exactly.
    if need:
        first = min(tab1[pair][0].start[2] for pair in need)
        last = max(tab1[pair][-1].start[2] for pair in need)
        cut = (first + last + 1) // 2
        if all(
            bisect_left(tab1[pair], cut - 1, key=lambda occ: occ.start[2]) >= count
            and len(tab1[pair]) - bisect_left(tab1[pair], cut, key=lambda occ: occ.start[2]) >= count
            for pair, count in need.items()
        ):
            return None
    return "no_disjoint_source_events" if _partition_witnesses(tab1, need) is None else None


def covered_disjoint_s1(tabs: Dict, person: str, word: str) -> bool:
    """Require physical key event separation in Session 1 and digraph coverage in Session 2."""
    return _coverage_exclusion(tabs, person, word) is None


def partition_tab_disjoint(tab1: Dict, word: str, rng) -> Tuple[Dict, Dict]:
    """Assign source events to enroll or same-day pools for one person's word."""
    witnesses = _partition_witnesses(tab1, Counter(zip(word, word[1:])), rng)
    if witnesses is None:
        raise ValueError(f"Session 1 cannot supply source-disjoint digraphs for {word!r}")
    pools, occupied = witnesses
    reserved = {occ.start for pool in pools for instances in pool.values() for occ in instances}
    remaining = [(pair, occ) for pair in pools[0] for occ in tab1[pair] if occ.start not in reserved]
    for i in rng.permutation(len(remaining)):
        pair, occ = remaining[i]
        start = occ.start[2]
        roles = [role for role in (0, 1) if not (occupied[1 - role][start] or occupied[1 - role][start + 1])]
        if not roles:
            continue  # the two endpoints already belong to opposite pools
        if len(roles) == 2 and len(pools[0][pair]) == len(pools[1][pair]):
            role = int(rng.integers(2))
        else:
            role = min(roles, key=lambda k: len(pools[k][pair]))
        pools[role][pair].append(occ)
        occupied[role][start] += 1
        occupied[role][start + 1] += 1
    return pools[0], pools[1]


def synthesise(tab: Dict, word: str, rng, window: float = 120.0) -> Tuple[np.ndarray, np.ndarray, List[SourceKey]]:
    """Return raw holds, raw DD intervals and a source occurrence ID per interval."""
    pairs = list(zip(word, word[1:]))
    a0 = pairs[rng.integers(len(pairs))]
    anchor = tab[a0][rng.integers(len(tab[a0]))].time
    holds, dds, occ_ids = [], [], []
    for i, pr in enumerate(pairs):
        inst = tab[pr]
        ts = np.array([x.time for x in inst])
        near = np.where(np.abs(ts - anchor) <= window)[0]
        if len(near) == 0:
            near = np.argsort(np.abs(ts - anchor))[:2]
        x = inst[int(rng.choice(near))]
        dds.append(x.dd)
        holds.append(x.hold_a)
        occ_ids.append(x.start)
        if i == len(pairs) - 1:
            holds.append(x.hold_b)
    return np.array(holds, dtype=np.float32), np.array(dds, dtype=np.float32), occ_ids


def synth(tab: Dict, word: str, rng, window: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
    """Two-tuple wrapper returning unclipped (holds, dds) in seconds."""
    holds, dds, _ = synthesise(tab, word, rng, window=window)
    return holds, dds


def _source_keys(occ_ids: Sequence[SourceKey]) -> set[SourceKey]:
    return {key for start in occ_ids for key in (start, _end_key(start))}


def _enrollment_provenance(enroll, same, person: str, word: str):
    occ_ids = sorted({occ for _, _, sources in enroll for occ in sources})
    source_keys = _source_keys(occ_ids)
    if any(not source_keys.isdisjoint(_source_keys(sources)) for _, _, sources in same):
        raise AssertionError(f"Session 1 enrollment and query share source key events for {person} on {word!r}")
    return occ_ids, sorted(source_keys)


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
    least 70% of `people` can type with disjoint Session-1 source events and Session-2 coverage."""
    def cov_all(w: str) -> bool:
        return sum(covered_disjoint_s1(tabs, p, w) for p in people) >= max(6, int(0.7 * len(people)))

    allowed_vocab = set(vocab_people) if vocab_people is not None else None
    vocab = Counter()
    vocab_sessions = sorted(key for key in raw_sessions if allowed_vocab is None or key[0] in allowed_vocab)
    for key in vocab_sessions:
        cur = []
        for label, _, _ in list(raw_sessions[key]) + [("Space", 0, 0)]:
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
    enforcing disjoint physical key events between enrollment and same-day queries for each word."""
    rows = []
    for kind, wl in strings.items():
        for w in wl:
            ppl = [p for p in people if covered_disjoint_s1(tabs, p, w)]
            if len(ppl) < 2:
                continue
            typ = {}
            enroll_provenance = {}
            for p in ppl:
                enr_tab, same_tab = partition_tab_disjoint(tabs[(p, 1)], w, rng)
                typ[(p, "enroll")] = [synthesise(enr_tab, w, rng) for _ in range(k_enroll)]
                typ[(p, "same")] = [synthesise(same_tab, w, rng) for _ in range(3)]
                typ[(p, "cross")] = [synthesise(tabs[(p, 2)], w, rng) for _ in range(3)]
                typ[(p, "imp")] = [synthesise(tabs[(p, int(rng.integers(1, 3)))], w, rng) for _ in range(2)]

                enroll_provenance[p] = _enrollment_provenance(typ[(p, "enroll")], typ[(p, "same")], p, w)

            keys = [(k, i) for k, v in typ.items() for i in range(len(v))]
            query_sources = {(k, i): sorted(_source_keys(typ[k][i][2])) for k, i in keys if k[1] != "enroll"}
            enrollment_sources = {p: [attempt[2] for attempt in typ[(p, "enroll")]] for p in ppl}
            enrollment_ids = {p: [f"{kind}|{w}|{p}|enroll|{k}" for k in range(k_enroll)] for p in ppl}
            idx = defaultdict(list)
            for j, k in enumerate(keys):
                idx[k[0]].append(j)
            for mode in ("kb", "touch"):
                T = typing_tensors(w, [typ[k][i] for k, i in keys], bounds, mean, std, mode == "touch", touch_hold)
                E9, E17 = embed9(T).astype(np.float64), embed17(T).astype(np.float64)
                for p in ppl:
                    t9, t17 = make_template(E9[idx[(p, "enroll")]]), make_template(E17[idx[(p, "enroll")]])
                    enroll_occ_ids, enroll_source_keys = enroll_provenance[p]
                    for q, tag, lab in [(p, "same", 1), (p, "cross", 1)] + [(q, "imp", 0) for q in ppl if q != p]:
                        J = idx[(q, tag)]
                        r9, r17 = raw_similarity(E9[J], t9), raw_similarity(E17[J], t17)
                        dep = as_norm_scores(E9[J], t9, cohorts9[mode])
                        for j in range(len(J)):
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
                                    "enroll_occ_ids": enroll_occ_ids,
                                    "query_occ_ids": typ[(q, tag)][j][2],
                                    "enroll_source_key_ids": enroll_source_keys,
                                    "query_source_key_ids": query_sources[((q, tag), j)],
                                    "attempt_id": f"{kind}|{w}|{q}|{tag}|{j}",
                                    "enroll_attempt_ids": enrollment_ids[p],
                                    "enroll_attempt_occ_ids": enrollment_sources[p],
                                }
                            )
    return rows


def _source_digest(attempts):
    return hashlib.sha256(json.dumps(attempts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _coverage_metadata(tabs, people, strings, requested_strings):
    exclusions = Counter()
    participants_by_string = {}
    for kind, words in strings.items():
        participants_by_string[kind] = {}
        for word in words:
            n = 0
            for person in people:
                reason = _coverage_exclusion(tabs, person, word)
                if reason is None:
                    n += 1
                else:
                    exclusions[reason] += 1
            if n < 2:
                exclusions["fewer_than_two_participants"] += n
                n = 0
            participants_by_string[kind][word] = n
    return {
        "num_participants_requested": len(people),
        "requested_strings_by_kind": dict(requested_strings),
        "selected_strings_by_kind": {kind: len(words) for kind, words in strings.items()},
        "scored_participants_by_string": participants_by_string,
        "excluded_person_string_pairs": sum(exclusions.values()),
        "exclusion_counts_by_reason": dict(exclusions),
    }


def synthesis_metadata(
    tabs: Dict,
    people: Sequence[str],
    strings: Dict[str, List[str]],
    rows: Sequence[Dict],
    *,
    requested_strings: Optional[Dict[str, int]] = None,
) -> Dict:
    """Audit physical key separation, sample counts and source eligibility."""
    same_rows = [r for r in rows if r.get("tag") == "same"]
    checked = [
        r for r in same_rows
        if r.get("enroll_source_key_ids") and r.get("query_source_key_ids")
        and set(r["enroll_source_key_ids"]).isdisjoint(r["query_source_key_ids"])
    ]
    coverage = _coverage_metadata(tabs, people, strings, requested_strings or {k: len(v) for k, v in strings.items()})
    counts = synthetic_input_counts(rows)
    sources_by_attempt = {}
    for r in rows:
        if "query_occ_ids" in r:
            sources_by_attempt[r["attempt_id"]] = r["query_occ_ids"]
        sources_by_attempt.update(zip(r.get("enroll_attempt_ids", ()), r.get("enroll_attempt_occ_ids", ())))
    return {
        **counts,
        **coverage,
        "num_participants_scored": len({r["target"] for r in rows}),
        "disjoint_occ_enforced": bool(same_rows) and all(
            r.get("enroll_occ_ids") and r.get("query_occ_ids") and set(r["enroll_occ_ids"]).isdisjoint(r["query_occ_ids"])
            for r in same_rows
        ),
        "disjoint_source_events_enforced": bool(same_rows) and len(checked) == len(same_rows),
        "source_event_disjointness_scope": SOURCE_EVENT_SCOPE,
        "source_trial_sha256": _source_digest(sources_by_attempt),
        "same_day_rows_verified": len(checked),
        "same_day_generated_inputs_verified": len({r["attempt_id"] for r in checked}),
        "total_generated_attempts": counts["generated_attempts"],
    }


def synthetic_input_counts(rows):
    """Count generated input IDs separately from their many template comparisons.

    Keyboard/touch representations share an input ID and are not new captures.
    IDs count generation operations, not statistically independent raw events.
    """
    enroll = {aid for r in rows for aid in r.get("enroll_attempt_ids", [])}
    gen = {r["attempt_id"] for r in rows if r["label"] == 1 and "attempt_id" in r}
    imp = {r["attempt_id"] for r in rows if r["label"] == 0 and "attempt_id" in r}
    return {"generated_attempts": len(enroll | gen | imp),
            "generated_enrollment_inputs": len(enroll),
            "generated_genuine_query_inputs": len(gen),
            "generated_impostor_query_inputs": len(imp),
            "comparison_trials": len(rows),
            "counting_note": "Generation IDs, not independent raw observations; keyboard/touch share IDs."}


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
        res = {
            "num_genuine": int(y.sum()),
            "num_impostor": int((y == 0).sum()),
            "num_strings": len({x["word"] for x in r}),
            "num_people": len({x["target"] for x in r}),
            **synthetic_input_counts(r),
            "num_genuine_trials": int(y.sum()),
            "num_impostor_trials": int((y == 0).sum()),
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


def select_adaptation_policy(validation_by_mode, margins):
    """Select off/on using Validation only; ties favor no adaptation."""
    candidates = ["dual_no_adapt"] + [f"dual_adapt_margin_{m:g}" for m in margins]
    costs = {}
    for policy in candidates:
        costs[policy] = sum(
            0.5 * (np.mean([v for k, v in values[policy].items() if k.startswith("frr_")])
                   + values[policy]["far"])
            for values in validation_by_mode.values()
        )
    chosen = min(candidates, key=lambda policy: costs[policy])
    return chosen, costs


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
    # Fix the attack queries before policy evaluation so every policy sees the
    # exact same people/attempts. No policy can improve by drawing easier attacks.
    impostor_indices = {}
    for p in np.unique(P):
        others = np.where((P != p) & (S == 2))[0]
        impostor_indices[p] = rng.choice(others, size=min(40, len(others)), replace=False)
    query_hash = hashlib.sha256(b"".join(impostor_indices[p].tobytes() for p in sorted(impostor_indices))).hexdigest()
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
            oi = impostor_indices[p]
            if policy == "deployed":
                far += list(as_norm_scores(E9[oi], make_template(E9[enr]), cohorts9[mode]) >= deployed_thr[mode])
            else:
                t9, t17 = make_template(np.array(ex9[-keep:])), make_template(np.array(ex17[-keep:]))
                far += list(dual_z(raw_similarity(E9[oi], t9), raw_similarity(E17[oi], t17), prm) >= prm["tau"])
        out[policy] = {
            **{f"frr_{b}": round(float(np.mean(v)), 4) for b, v in frr.items()},
            "far": round(float(np.mean(far)), 4),
            "num_impostor_trials": len(far),
            "paired_impostor_sha256": query_hash,
            "limitation": "Only genuine updates precede final impostor probes; not a poisoning-resistance test.",
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
    drawn from that person's CapsLock rate in that session, with separate Session-1 source events.
    Rows: (label, z, query_caps, enrol_caps)."""
    people = _capital_people(tabs, rates, people)
    words = strings_of_length(tabs, people, length, n_words, rng)
    rows, _ = _score_capital_trials(tabs, rates, people, words, embed9, embed17, bounds, mean, std,
                                    touch_hold, prm, rng, k_enroll, n_words)
    return rows


def _capital_people(tabs, rates, people):
    return [p for p in people if all((p, session) in tabs and (p, session) in rates for session in (1, 2))]


def _score_capital_trials(
    tabs, rates, people, words, embed9, embed17, bounds, mean, std, touch_hold, prm, rng, k_enroll, n_words,
):
    rows = []
    counts = Counter()
    scored_people = set()
    sources_by_attempt = {}
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

            _enrollment_provenance([typing for typing, _ in enr_att], [typing for typing, _ in gen_s1], p, w)
            typ[p] = {"e": enr_att, "g": gen_s1 + gen_s2, "i": imp_s2}
            counts["generated_enrollment_inputs"] += len(enr_att)
            counts["generated_genuine_query_inputs"] += len(gen_s1) + len(gen_s2)
            counts["generated_impostor_query_inputs"] += len(imp_s2)
            counts["same_day_generated_inputs_verified"] += len(gen_s1)
            scored_people.add(p)
            for tag, attempts in typ[p].items():
                for i, (typing, method) in enumerate(attempts):
                    sources_by_attempt[f"{w}|{p}|{tag}|{i}"] = (typing[2], method)

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
    generated = sum(counts[k] for k in ("generated_enrollment_inputs", "generated_genuine_query_inputs", "generated_impostor_query_inputs"))
    audit = {
        **_coverage_metadata(tabs, people, {"capital": words}, {"capital": n_words}),
        **counts,
        "num_participants_scored": len(scored_people),
        "generated_attempts": generated,
        "comparison_trials": len(rows),
        "disjoint_source_events_enforced": counts["same_day_generated_inputs_verified"] > 0,
        "source_event_disjointness_scope": SOURCE_EVENT_SCOPE,
        "source_trial_sha256": _source_digest(sources_by_attempt),
        "same_day_rows_verified": counts["same_day_generated_inputs_verified"],
    }
    return rows, audit


def _capital_scores(rows, weight, caps_share, k_enroll):
    y = np.array([r[0] for r in rows])
    s = np.array([r[1] + weight * capital_llr(r[2], r[3], k_enroll, caps_share) for r in rows])
    return y, s


def _capital_error_rates(labels, scores, threshold):
    return float(np.mean(scores[labels == 0] >= threshold)), float(np.mean(scores[labels == 1] < threshold))


def _fit_capital_fusion(
    val_rows,
    caps_share: float,
    base_tau: float,
    k_enroll: int = 4,
    frr_target: float = 0.10,
    grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> Dict:
    """Select the weight and threshold offset on Validation using the exported population prior."""
    base_tau_r = round(float(base_tau), 4)
    caps_share = round(float(caps_share), 4)
    offsets: Dict[float, float] = {}
    val_far: Dict[float, float] = {}
    val_frr: Dict[float, float] = {}
    for w in grid:
        yv, sv = _capital_scores(val_rows, w, caps_share, k_enroll)
        offsets[w] = round(float(np.quantile(sv[yv == 1], frr_target) - base_tau_r), 4) if w else 0.0
        val_far[w], val_frr[w] = _capital_error_rates(yv, sv, base_tau_r + offsets[w])

    best = min(grid, key=lambda w: (val_far[w], w))
    tau_offset = offsets[best]
    return {
        "weight": float(best),
        "tau_offset": tau_offset,
        "caps_share": caps_share,
        "deployed_threshold_no_capitals": base_tau_r,
        "deployed_threshold_with_capitals": round(base_tau_r + tau_offset, 4),
        "validation_far_by_weight": {str(w): round(val_far[w], 4) for w in grid},
        "validation_frr_by_weight": {str(w): round(val_frr[w], 4) for w in grid},
    }


def _evaluate_capital_fusion(test_rows, calibration, k_enroll=4):
    base_tau = calibration["deployed_threshold_no_capitals"]
    caps_share = calibration["caps_share"]
    labels, raw = _capital_scores(test_rows, 0.0, caps_share, k_enroll)
    _, fused = _capital_scores(test_rows, calibration["weight"], caps_share, k_enroll)
    far0, frr0 = _capital_error_rates(labels, raw, base_tau)
    far1, frr1 = _capital_error_rates(labels, fused, base_tau + calibration["tau_offset"])
    return {
        **calibration,
        "test_without": {"far": round(far0, 4), "frr": round(frr0, 4)},
        "test_with": {"far": round(far1, 4), "frr": round(frr1, 4)},
        "num_test_genuine": int(sum(r[0] for r in test_rows)),
        "num_test_impostor": int(sum(1 - r[0] for r in test_rows)),
    }


def capital_fusion(
    val_rows, test_rows, caps_share: float, base_tau: float, k_enroll: int = 4, frr_target: float = 0.10,
    grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> Dict:
    calibration = _fit_capital_fusion(val_rows, caps_share, base_tau, k_enroll, frr_target, grid)
    return _evaluate_capital_fusion(test_rows, calibration, k_enroll)


def run_password_synthesis(
    raw_sessions, split_people, embed9, embed17, bounds, mean, std, touch_hold, cohorts9, deployed_thr,
    *, seed: int, frr_target: float, min_length: int,
) -> Dict:
    """Freeze the free-text password protocol on Train/Validation, then generate and evaluate Test trials."""
    def sessions_for(people):
        return {(p, s): raw_sessions[(p, s)] for p in people for s in (1, 2) if (p, s) in raw_sessions}

    vocab_sessions = sessions_for(sorted(split_people["train"] + split_people["val"]))

    def draw_split(split, lengths, excluded):
        people = sorted(split_people[split])
        sessions = sessions_for(people)
        tabs = digraph_tables(sessions, people)
        available = [p for p in people if (p, 1) in tabs and (p, 2) in tabs]
        used = set(excluded)
        strings = choose_strings(vocab_sessions, tabs, available, 30, 30, password_rng(seed, split, "word_selection"), exclude=used)
        used.update(w for words in strings.values() for w in words)
        rows = string_trials(sessions, tabs, available, strings, embed9, embed17, bounds, mean, std,
                             touch_hold, cohorts9, password_rng(seed, split, "word_trials"))
        metadata = synthesis_metadata(tabs, people, strings, rows, requested_strings={"word": 30, "random": 30})
        length_strings, length_rows, length_metadata = {}, {}, {}
        for length in lengths:
            words = strings_of_length(tabs, available, length, 15, password_rng(seed, split, "length_selection", length), exclude=used)
            used.update(words)
            length_strings[length] = words
            length_rows[length] = string_trials(
                sessions, tabs, available, {"random": words}, embed9, embed17, bounds, mean, std,
                touch_hold, cohorts9, password_rng(seed, split, "length_trials", length),
            )
            length_metadata[length] = synthesis_metadata(
                tabs, people, {"random": words}, length_rows[length], requested_strings={"random": 15},
            )
        return {
            "people": people, "sessions": sessions, "tabs": tabs, "used": used,
            "strings": strings, "rows": rows, "metadata": metadata,
            "length_strings": length_strings, "length_rows": length_rows, "length_metadata": length_metadata,
        }

    def draw_capitals(split, data, rates, prm):
        n = 12 if split == "val" else 15
        tabs = data["tabs"]
        people = _capital_people(tabs, rates, data["people"])
        words = strings_of_length(tabs, people, min_length, n, password_rng(seed, split, "capital_selection", min_length),
                                  exclude=data["used"])
        rows, audit = _score_capital_trials(
            tabs, rates, people, words, embed9, embed17, bounds, mean, std, touch_hold, prm,
            password_rng(seed, split, "capital_trials", min_length), 4, n,
        )
        audit["num_participants_in_split"] = len(data["people"])
        audit["participants_without_capital_coverage"] = sorted(set(data["people"]) - set(people))
        return words, rows, audit

    validation = draw_split("val", CALIB_LENGTHS, ())
    dual = calibrate_dual(validation["rows"], frr_target)
    table = calibrate_by_length(validation["length_rows"], frr_target)
    capital_params = params_for_length(table, min_length, "kb")
    calibration_rates = capital_method_rates(vocab_sessions)
    caps_share = float(np.mean([calibration_rates[(p, 1)] for p in split_people["train"] if (p, 1) in calibration_rates]))
    val_capital_words, val_capital_rows, val_capital_meta = draw_capitals("val", validation, calibration_rates, capital_params)
    capital = _fit_capital_fusion(val_capital_rows, caps_share, capital_params["tau"], frr_target=frr_target)
    excluded = frozenset(validation["used"].union(val_capital_words))

    test = draw_split("test", (8, 12, 16, 20), excluded)
    test_rates = capital_method_rates(test["sessions"])
    test_capital_words, test_capital_rows, test_capital_meta = draw_capitals("test", test, test_rates, capital_params)
    capital = _evaluate_capital_fusion(test_capital_rows, capital)
    rates = {**calibration_rates, **test_rates}
    both_days = [p for p, s in rates if s == 1 and (p, 2) in rates]
    capital.update({
        "habit_same_across_days": round(float(np.mean([(rates[(p, 1)] > 0.5) == (rates[(p, 2)] > 0.5) for p in both_days])), 4),
        "people_mostly_capslock": sum(rates[(p, 1)] > 0.5 for p in both_days),
        "people_with_capitals": len(both_days),
        "strings": {"val": val_capital_words, "test": test_capital_words},
        "synthesis_metadata": {"val": val_capital_meta, "test": test_capital_meta},
        "selection_split": "val_only",
    })
    phases = {"val": validation, "test": test}
    speed = person_speed(test["tabs"], sorted({r["target"] for r in test["rows"]}))
    return {
        "password_user_strings": {
            "strings": {sp: data["strings"] for sp, data in phases.items()},
            "synthesis_metadata": {sp: data["metadata"] for sp, data in phases.items()},
            "frr_target": frr_target,
            "dual_calibration": dual,
            "calibration_split": "val_only",
            "vocabulary_splits": ["train", "val"],
            "rng_seed": seed,
            "test": evaluate_strings(test["rows"], dual, deployed_thr, speed),
        },
        "password_length_calibration": {
            "min_length": min_length,
            "table": table,
            "strings": {sp: {str(length): words for length, words in data["length_strings"].items()} for sp, data in phases.items()},
            "synthesis_metadata": {
                sp: {str(length): audit for length, audit in data["length_metadata"].items()} for sp, data in phases.items()
            },
            "test": evaluate_lengths(test["length_rows"], dual, table),
        },
        "password_capital_method": capital,
    }
