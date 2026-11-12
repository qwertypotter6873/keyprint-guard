"""
Why are impostors with almost the same typing speed hard to reject? (README section 5.6)

1) Tempo-matching experiment on user-chosen strings (test people, digraph-synthesised typing):
   each impostor attempt is re-timed so its mean interval (and, separately, its mean hold) equals the
   target's enrolment mean; the rhythm shape is untouched. A rise in the pass rate shows how much the
   decision leans on overall tempo / hold length rather than on per-key rhythm.
2) Variance decomposition of real free-text timing (all 99 KeyRecs people, session 1):
   log interval / log hold = key-layout effect + person speed + person x key-pair rhythm + attempt noise.
3) Information limit: with speed equal, an ideal Gaussian detector separates two people only through the
   person x key-pair rhythm; its EER for 9 intervals (password) vs ~60 / ~120 (one / two sentences).

    python src/eval_typing_style.py      -> models/typing_style_analysis.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_profile import load_keyrecs_sessions  # noqa: E402
from capture_sim import make_template, ms_rounded  # noqa: E402
from feature_eval import encode_variable  # noqa: E402
from models import KeyPrintNet  # noqa: E402
from password_synth import as_norm_scores, covered, digraph_tables, dual_z, raw_similarity, synth, typing_tensors  # noqa: E402
from preprocessing import (  # noqa: E402
    N_CHANNELS,
    N_TIMING_CHANNELS,
    SEED,
    apply_channel_scaler,
    extract_sequences,
    get_feature_columns,
    prepare_full_pipeline,
    touchify_timings,
)
from train_and_evaluate import compute_cohort_centroids, encode_sequences  # noqa: E402


def decompose(cells):
    """cells {(person, item): log values} -> variance components (method of moments, two-way layout)."""
    P = sorted({p for p, _ in cells}); G = sorted({g for _, g in cells})
    M = np.full((len(P), len(G)), np.nan); N = np.zeros_like(M); W = []
    for (p, g), v in cells.items():
        i, j = P.index(p), G.index(g)
        M[i, j] = v.mean(); N[i, j] = len(v)
        if len(v) >= 2:
            W.append((np.var(v, ddof=1), len(v) - 1))
    noise = sum(a * b for a, b in W) / sum(b for _, b in W)
    keep = ~np.isnan(M).any(axis=0); M, N = M[:, keep], N[:, keep]
    mu = M.mean(); a = M.mean(1) - mu; b = M.mean(0) - mu
    resid = M - mu - a[:, None] - b[None, :]
    rhythm = max(resid.var() - np.mean(noise / N), 0.0)
    speed = max(a.var() - rhythm / M.shape[1] - np.mean(noise / N) / M.shape[1], 0.0)
    layout = b.var()
    tot = layout + speed + rhythm + noise
    return {"people": int(M.shape[0]), "items": int(M.shape[1]),
            "share": {k: round(float(v / tot), 4) for k, v in (("key_layout", layout), ("person_speed", speed), ("person_rhythm", rhythm), ("attempt_noise", noise))},
            "var": {"layout": float(layout), "speed": float(speed), "rhythm": float(rhythm), "noise": float(noise)}}


def ideal_eer(rho: float, m: int) -> float:
    """EER between sigma^2*chi2_m (genuine) and rho*sigma^2*chi2_m (impostor) squared distances."""
    lo, hi, t = 0.0, 1e4, 1.0
    for _ in range(100):
        t = (lo + hi) / 2
        frr, far = chi2.sf(t, m), chi2.cdf(t / rho, m)
        lo, hi = (t, hi) if frr > far else (lo, t)
    return float(chi2.sf(t, m))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cpu")
    cfg = json.loads((root / "web/public/models/model_config.json").read_text(encoding="utf-8"))
    report = json.loads((root / "models/evaluation_report.json").read_text(encoding="utf-8"))
    dual = {"kb": cfg["password_dual"]["keyboard"], "touch": cfg["password_dual"]["touch"]}
    prev_thr = {"kb": cfg["calibrated_threshold"], "touch": cfg["calibrated_threshold_touch"]}
    with tempfile.TemporaryDirectory() as tmp:  # keep data/processed untouched
        splits, meta = prepare_full_pipeline(root / "data/raw/DSL-StrongPasswordData.csv", Path(tmp), seed=SEED)
    kr = meta["_keyrecs_splits_subjs"]
    mean, std = np.array(meta["channel_mean"], np.float32), np.array(meta["channel_std"], np.float32)
    bounds = {k: tuple(v) for k, v in meta["outlier_bounds"].items()}
    touch_hold = float(meta["touch_hold_impute_sec"])

    m9 = KeyPrintNet(in_channels=N_TIMING_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18)
    m9.load_state_dict(torch.load(root / "models/legacy/keyprint_net_9ch.pt", map_location="cpu", weights_only=True)); m9.eval()
    m17 = KeyPrintNet(in_channels=N_CHANNELS, hidden_channels=80, embed_dim=64, dropout=0.18)
    m17.load_state_dict(torch.load(root / "models/keyprint_net.pt", map_location="cpu", weights_only=True)); m17.eval()
    e9 = lambda T: encode_variable(m9, list(T), device, N_TIMING_CHANNELS).astype(np.float64)  # noqa: E731
    e17 = lambda T: encode_variable(m17, list(T), device, N_CHANNELS).astype(np.float64)  # noqa: E731
    _, hold_cols, dd_cols, _ = get_feature_columns()
    tr = splits["train"]
    h, d = tr.df[hold_cols].values.astype(np.float32), tr.df[dd_cols].values.astype(np.float32)
    h2, u2 = touchify_timings(h, d, touch_hold, bounds)
    cohorts = {"kb": compute_cohort_centroids(encode_sequences(m9, tr.seq_scaled[:, :N_TIMING_CHANNELS], device), tr.subject_ids),
               "touch": compute_cohort_centroids(encode_sequences(m9, apply_channel_scaler(extract_sequences(h2, d, u2, tr.key_block, np.zeros(len(d), np.float32)), mean, std)[:, :N_TIMING_CHANNELS], device), tr.subject_ids)}

    raw = {k: ms_rounded(v) for k, v in load_keyrecs_sessions(root / "data/raw/KeyRecs-free-text.csv").items()}
    out = {}

    # 1) tempo matching
    tabs = digraph_tables(raw, kr["test"])
    strings = report["feature_upgrade"]["password_user_strings"]["strings"]["test"]
    words = strings["word"][:12] + strings["random"][:12]
    rng = np.random.default_rng(SEED)
    acc = {m: {f"{v}_{r}": [] for v in ("original", "tempo_matched", "tempo_hold_matched") for r in ("previous", "dual")} for m in ("kb", "touch")}
    for w in words:
        ppl = [p for p in kr["test"] if (p, 1) in tabs and (p, 2) in tabs and all(covered(tabs[(p, s)], w) for s in (1, 2))]
        typ = {p: {"enroll": [synth(tabs[(p, 1)], w, rng) for _ in range(4)], "imp": [synth(tabs[(p, 2)], w, rng) for _ in range(2)]} for p in ppl}
        for mode in ("kb", "touch"):
            for p in ppl:
                eh = np.mean([x[0].mean() for x in typ[p]["enroll"]]); ed = np.mean([x[1].mean() for x in typ[p]["enroll"]])
                q = {"original": [], "tempo_matched": [], "tempo_hold_matched": []}
                for o in ppl:
                    if o == p:
                        continue
                    for hh, dd in typ[o]["imp"]:
                        q["original"].append((hh, dd))
                        q["tempo_matched"].append((hh, dd * ed / dd.mean()))
                        q["tempo_hold_matched"].append((hh * eh / hh.mean(), dd * ed / dd.mean()))
                allx = typ[p]["enroll"] + q["original"] + q["tempo_matched"] + q["tempo_hold_matched"]
                T = typing_tensors(w, allx, bounds, mean, std, mode == "touch", touch_hold)
                E9, E17 = e9(T), e17(T)
                t9, t17 = make_template(E9[:4]), make_template(E17[:4])
                n = len(q["original"])
                for k, name in enumerate(("original", "tempo_matched", "tempo_hold_matched")):
                    sl = slice(4 + k * n, 4 + (k + 1) * n)
                    acc[mode][f"{name}_previous"] += list(as_norm_scores(E9[sl], t9, cohorts[mode]) >= prev_thr[mode])
                    acc[mode][f"{name}_dual"] += list(dual_z(raw_similarity(E9[sl], t9), raw_similarity(E17[sl], t17), dual[mode]) >= dual[mode]["tau"])
    out["tempo_matching_impostor_pass_rate"] = {m: {k: round(float(np.mean(v)), 4) for k, v in acc[m].items()} for m in acc}
    out["tempo_matching_strings"] = words

    # 2) variance decomposition (session 1 of every KeyRecs person)
    people = kr["train"] + kr["val"] + kr["test"]
    tabs_all = digraph_tables(raw, people)
    cnt = Counter(g for (p, s), t in tabs_all.items() if s == 1 for g, v in t.items() if len(v) >= 4)
    common = [g for g, c in cnt.items() if c >= 0.9 * len(people)]
    dd_cells = {(p, g): np.log([x[1] for x in tabs_all[(p, 1)][g]]) for p in people if (p, 1) in tabs_all
                for g in common if len(tabs_all[(p, 1)].get(g, [])) >= 4}
    hold_cells = {}
    for p in people:
        if (p, 1) not in tabs_all:
            continue
        hk = {}
        for g, v in tabs_all[(p, 1)].items():
            hk.setdefault(g[0], []).extend(x[2] for x in v)
        for k, v in hk.items():
            if len(v) >= 8:
                hold_cells[(p, k)] = np.log(v)
    cntk = Counter(k for (_, k) in hold_cells)
    hold_cells = {key: v for key, v in hold_cells.items() if cntk[key[1]] >= 0.9 * len(people)}
    dd_c, h_c = decompose(dd_cells), decompose(hold_cells)
    out["variance_decomposition"] = {"digraph_interval": {k: v for k, v in dd_c.items() if k != "var"}, "key_hold": {k: v for k, v in h_c.items() if k != "var"}}

    # 3) information limit for two people of the same speed (4 enrolment reps)
    k = 4
    rho_dd = 1 + 2 * dd_c["var"]["rhythm"] / (dd_c["var"]["noise"] * (1 + 1 / k))
    rho_h = 1 + 2 * h_c["var"]["rhythm"] / (h_c["var"]["noise"] * (1 + 1 / k))
    out["same_speed_ideal_eer"] = {
        "password_9_intervals": round(ideal_eer(rho_dd, 9), 4),
        "password_9_intervals_plus_10_holds": round(ideal_eer(float(np.exp((9 * np.log(rho_dd) + 10 * np.log(rho_h)) / 19)), 19), 4),
        "one_sentence_60_intervals": round(ideal_eer(rho_dd, 60), 4),
        "two_sentences_120_intervals": round(ideal_eer(rho_dd, 120), 4),
    }
    path = root / "models/typing_style_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
