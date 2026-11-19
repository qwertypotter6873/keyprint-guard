"""Reproduce the Korean FACT timing study using the two published Arena releases.

Run with Git, NumPy and web dependencies installed (`npm ci --prefix web`):
  python src/eval_korean_fact.py > /tmp/korean_fact_ab.json

The scorer obtains missing frozen Git revisions in a temporary directory.
The report contains aggregate outcomes and provenance.
"""

import argparse
import codecs
import hashlib
import io
import json
import pickle
from collections import Counter, defaultdict
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npy
try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import _reconstruct


ROOT = Path(__file__).resolve().parents[1]
PROMPT_STROKES = 76
SUPPLEMENT = "https://ars.els-cdn.com/content/image/1-s2.0-S0031320320303599-mmc{}.zip"
ARCHIVE_HASHES = {
    2: "183f9b86161b787adc7aeac7281c104b8af447843fa6c041bfac8d1647a5a259",
    4: "44cfd09bf5778897bd9892021676648e436be9c113584509d2e8d69c64f98b99",
    6: "9ba9ea627bb1eedf7ebac25618b1c97ac0160d55d985e289a984eec029b6ee89",
    8: "ef915244c172a52ddf5004bb4251add9c8ae9d04971732d08a9968ff2ef4ce1b",
}
RELEASES = {
    "enroll": (4, 6),
    "query": (2, 8),
}


class SupplementArrayUnpickler(pickle.Unpickler):
    """Read the pinned publisher array using only its four NumPy/codec globals."""

    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("_codecs", "encode"): codecs.encode,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unexpected global in the publisher supplement.")
        return allowed[module, name]


def supplement(number, archive_dir):
    archive_name = f"mmc{number}.zip"
    if archive_dir:
        content = (archive_dir / archive_name).read_bytes()
    else:
        request = Request(SUPPLEMENT.format(number), headers={"User-Agent": "keyprint-guard/1.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_HASHES[number]:
        raise ValueError(f"The publisher checksum does not match for {archive_name}.")
    with ZipFile(io.BytesIO(content)) as archive, archive.open(f"mmc{number}.npy") as file:
        if number >= 5:
            return np.load(file, allow_pickle=False)
        if npy.read_magic(file) != (1, 0):
            raise ValueError("Unexpected NumPy header version in the publisher supplement.")
        shape, fortran, dtype = npy.read_array_header_1_0(file)
        if shape != (500, 642, 32) or fortran or not dtype.hasobject:
            raise ValueError("Unexpected shape or type in the publisher supplement.")
        return SupplementArrayUnpickler(file).load()


def touch_capture(recording_id, intervals):
    """Keep passable DD intervals while representing unavailable browser data."""
    dds = []
    stitches = []
    gap = False
    for dd in intervals:
        sec = float(dd) / 1_000_000_000
        if not 0.006 <= sec <= 2.2:
            gap = True
            continue
        stitches.append(bool(dds) and gap)
        dds.append(sec)
        gap = False
    count = len(dds)
    if count < 8:
        raise ValueError("A preselected source recording has fewer than eight in-bounds intervals.")
    hold_count = count + 1 + sum(stitches)
    return {
        "recording_id": recording_id,
        "inputMode": "touch",
        "cleanDigraphCount": count,
        "cpm": round(60 * count / sum(dds)),
        "holdMs": 0,
        "rolloverPct": 0,
        "behavior": None,
        "stroke": {
            "holds": [None] * hold_count,
            "holdValid": [False] * hold_count,
            "hCurr": [None] * count,
            "hNext": [None] * count,
            "hvCurr": [False] * count,
            "hvNext": [False] * count,
            "dds": dds,
            "uds": [None] * count,
            "stitchFlags": stitches,
            "pairKeys": [[None, None]] * count,
        },
    }


def extract_split(role, archive_dir):
    sequence_file, label_file = RELEASES[role]
    data = supplement(sequence_file, archive_dir)
    labels = supplement(label_file, archive_dir).astype(int)
    if labels.shape != (500, 2) or Counter(labels[:, 0]) != Counter({i: 10 for i in range(1, 51)}):
        raise ValueError("The participant labels do not match the expected 50 × 10 split.")

    by_writer = defaultdict(list)
    key_sequences = set()
    raw_recordings = set()
    interval_counts = {
        name: {"before_range_filter": 0, "after_range_filter": 0}
        for name in ("prompt_length", "full_recording")
    }
    for index, (sequence, (writer, length)) in enumerate(zip(data, labels, strict=True)):
        # Label column 1 is the number of unpadded digraph rows. Adding one
        # recovers the per-recording keystroke count from dissertation Table 21.
        keys = sequence[:length, :2].astype(int)
        if (np.any(sequence[length:, 0] != 0.0) or np.any(keys < 0) or np.any(keys > 31)
                or not np.array_equal(keys[:-1, 1], keys[1:, 0])):
            raise ValueError("The label lengths do not align with the paired keystroke rows.")

        # Six nanosecond timing columns. Adjacent rows share the previous
        # destination hold and the next source hold. DD = UD + source hold.
        hold1, hold2, dd, uu, du, ud = sequence[:length, 14:20].astype(float).T
        if (not np.array_equal(hold2[:-1], hold1[1:]) or not np.array_equal(dd, ud + hold1)
                or not np.array_equal(du, dd + hold2) or not np.array_equal(du, uu + hold1)):
            raise ValueError("Unexpected FACT timing column order.")
        key_sequences.add(np.append(keys[:, 0], keys[-1, 1]).astype("uint8").tobytes())
        raw_recordings.add(hashlib.sha256(dd.astype("<f8").tobytes()).digest())

        # Every writer receives the same enrollment budget. Preserve the
        # publisher's record order within each writer in both partitions.
        captures = by_writer[writer]
        if role == "enroll" and len(captures) >= 5:
            continue
        if len(dd) < PROMPT_STROKES - 1:
            raise ValueError("A selected recording is shorter than the fixed keystroke prefix.")
        versions = {}
        for name, intervals in (("prompt_length", dd[:PROMPT_STROKES - 1]), ("full_recording", dd)):
            capture = touch_capture(f"{role}-{index}", intervals)
            interval_counts[name]["before_range_filter"] += len(intervals)
            interval_counts[name]["after_range_filter"] += capture["cleanDigraphCount"]
            versions[name] = capture
        captures.append(versions)

    if len(raw_recordings) != len(labels):
        raise ValueError("Duplicate human timing observations in the publisher partition.")

    return by_writer, key_sequences, raw_recordings, {
        "reported_keystrokes_per_recording_mean": round(float(np.mean(labels[:, 1] + 1)), 3),
        "min_source_keystrokes": int(np.min(labels[:, 1])) + 1,
        "available_recordings": len(labels),
        "selected_recordings": sum(map(len, by_writer.values())),
        "digraphs": interval_counts,
    }


def study_observations(enroll, query, variant):
    return {
        "schema": "keyprint-korean-eval/v1",
        "language": "ko",
        "input_mode": "touch",
        "capture_protocol": "published_recording_split",
        "participants": [
            {
                "id": str(writer),
                "sessions": [
                    {"id": "publisher-reference", "role": "enroll",
                     "captures": [record[variant] for record in enroll[writer]]},
                    {"id": "publisher-test", "role": "query",
                     "captures": [record[variant] for record in query[writer]]},
                ],
            }
            for writer in sorted(enroll)
        ],
    }


def score_variants(enroll, query, args):
    with tempfile.TemporaryDirectory(prefix="keyprint-fact-") as temp:
        command = [
            "node", str(ROOT / "web/scripts/eval_korean_ab.mjs"),
            "--five-person-rosters",
        ]
        for variant in ("prompt_length", "full_recording"):
            input_path = Path(temp) / f"{variant}.json"
            input_path.write_text(json.dumps(study_observations(enroll, query, variant)), encoding="utf8")
            command.extend(["--input", str(input_path)])
        if args.main_ref:
            command.extend(["--main-ref", args.main_ref])
        if args.candidate_ref:
            command.extend(["--candidate-ref", args.candidate_ref])
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True, cwd=ROOT)
    return json.loads(result.stdout)


def run(args):
    enroll, enroll_keys, enroll_timings, enroll_metadata = extract_split("enroll", args.archive_dir)
    query, query_keys, query_timings, query_metadata = extract_split("query", args.archive_dir)
    if enroll_timings & query_timings:
        raise ValueError("The reference and test partitions share timing observations.")
    report, full_recording = score_variants(enroll, query, args)
    report["full_recording"] = {
        "enrolled_probe_roster_size": full_recording["enrolled_probe_roster_size"],
        "unknown_probe_roster_size": full_recording["unknown_probe_roster_size"],
        "main": full_recording["main"],
        "candidate": full_recording["candidate"],
        "paired_change": full_recording["paired_change"],
        "full_roster": full_recording["full_roster"],
    }
    report["dataset"] = {
        "name": "FACT (Korean subset), Kim and Kang (2020)",
        "doi": "10.1016/j.patcog.2020.107556",
        "paper": "https://pure.korea.ac.kr/en/publications/freely-typed-keystroke-dynamics-based-user-authentication-for-mob/",
        "methods": "https://dcollection.korea.ac.kr/srch/srchDetail/000000127101",
        "method_reference": "Junhong Kim's dissertation, Chapter 4, Tables 16, 19 and 21",
        "device": "Samsung Galaxy S8, Android 9",
        "archives": {
            role: [
                {"url": SUPPLEMENT.format(number), "sha256": ARCHIVE_HASHES[number]}
                for number in numbers
            ]
            for role, numbers in RELEASES.items()
        },
        "reference": enroll_metadata,
        "test": query_metadata,
        "exact_shared_timing_recordings": 0,
        "exact_shared_key_sequences": len(enroll_keys & query_keys),
        "prompt_ids_available": False,
        "session_dates_available": False,
    }
    report["protocol"] = {
        "model_inputs": "Observed touch DD only. Key geometry is unknown. Holds and UD use each release's touch imputation. Behavior is unavailable.",
        "source_prefix_keystrokes": PROMPT_STROKES,
        "source_prefix_dd_intervals": PROMPT_STROKES - 1,
        "prefix_length_selection": "First consecutive keystrokes in each recording, fixed to the median two-set keystroke count of the 64 Korean Arena prompts.",
        "full_recording_sensitivity": "The same enrollments, query pairs and rosters with whole source recordings.",
        "filter": "0.006 to 2.2 seconds per DD, inclusive, with a stitch after discarded intervals.",
        "enrollment": "First five recordings per writer in the publisher reference split, in archive order.",
        "query": "All ten recordings per writer in the publisher test split, grouped into five nonoverlapping pairs in archive order.",
        "primary_roster": "Ten fixed groups of five by ascending pseudonymous writer ID. Enrolled probes use their group; unknown probes use the next group cyclically.",
        "full_roster": "Enrolled probes use all 50 writers; unknown probes use the 49 other writers. Same query pool as primary.",
        "paired_intervals": "Percentile bootstrap over 50 writer blocks, 2000 draws, seed 1729. Descriptive for these fixed rosters and recordings.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, help="Read existing mmc{2,4,6,8}.zip files from here.")
    parser.add_argument("--main-ref", help="Override the deployed baseline commit.")
    parser.add_argument("--candidate-ref", help="Override the reviewed candidate commit.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
