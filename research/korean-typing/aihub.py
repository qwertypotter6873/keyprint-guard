"""Read checksum-pinned Korean keyboard data and build the same-day cohorts."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
from statistics import median
from zipfile import ZipFile

import numpy as np


SPLITS = ('TS', 'VS')
PROTOCOLS = ((40, 'chronological'), (76, 'chronological'), (40, 'global_question_holdout'))
ARCHIVE_HASHES = {
    'TS': '49df847ff3637bc5e952ef20f4583a6f88941e1fab521e74c5f60c1dc2295af4',
    'VS': 'e394e99cf38185d5545e777cb821c2618046fecc8b2526241f3908d88df4044f',
}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
ONE_US = timedelta(microseconds=1)
KST = timezone(timedelta(hours=9))


def text_key(key):
    if key == 'SPACE':
        return True
    if isinstance(key, int):
        return 0 <= key <= 9
    return isinstance(key, str) and len(key) == 1 and key.isprintable()


def hangul(key):
    return isinstance(key, str) and bool(re.fullmatch('[가-힣ㄱ-ㆎᄀ-ᇿ]', key))


def parse_record(row, part, question, digest, touch_capture):
    events = row['data']['STAGE_BASE']
    down = [(datetime.fromisoformat(e['time']), e['key']) for e in events if e['action'] == 'KEY_DOWN']
    if not down:
        return None, 'no_base_downs'
    assert all(ts.utcoffset() is not None for ts, _ in down)
    if any(a[0] > b[0] for a, b in zip(down, down[1:])):
        return None, 'out_of_order_keydowns'
    day = down[0][0].astimezone(KST).date()
    if down[-1][0].astimezone(KST).date() != day:
        return None, 'cross_midnight_record'
    first = (down[0][0] - EPOCH) // ONE_US
    last = (down[-1][0] - EPOCH) // ONE_US
    caps = {}
    intervals = []
    taps = 0
    korean = 0
    previous = None
    boundary = False
    for ts, key in down:
        if not text_key(key):
            boundary = True
            continue
        taps += 1
        korean += hangul(key)
        if previous is not None:
            # A sentinel makes the existing touch adapter start a new segment.
            # Do not bridge across deletion, toggles, submit, or another non-text tap.
            dt = (ts - previous) // ONE_US
            intervals.append(0 if boundary else dt * 1000)
        previous = ts
        boundary = False
        if taps in (40, 76) and korean * 2 >= taps:
            try:
                capture = touch_capture(f'q{question}', intervals)
            except ValueError:
                continue
            # test the timing actually passed to the model for duplicates as well
            timing = [capture['stroke']['dds'], capture['stroke']['stitchFlags']]
            stamp = hashlib.sha256(json.dumps(timing, separators=(',', ':')).encode()).digest()
            caps[taps] = (capture, stamp)
        if taps == 76:
            break
    return dict(writer=row['student_id'], day=day, q=question, start=first, end=last,
                part=part, digest=digest, caps=caps), None


def verified_archives(root):
    if not root.is_dir():
        raise ValueError('Pass the local AI Hub dataset directory with --aihub.')
    archives = {}
    for split, expected in ARCHIVE_HASHES.items():
        candidates = sorted(path for path in root.rglob(f'{split}.zip') if path.is_file())
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one {split}.zip inside --aihub, found {len(candidates)}.')
        archive = candidates[0]
        with archive.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'{split}.zip SHA-256 mismatch. Expected {expected}, received {actual}.')
        archives[split] = archive
    return archives


def signal_stamp(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).digest()


def describe_raw(rows):
    dates = defaultdict(set)
    lengths = []
    long_counts = Counter()
    for student, date, length, _ in rows:
        dates[student].add(date)
        lengths.append(length)
        if length >= 76:
            long_counts[student] += 1
    return {
        'records': len(lengths),
        'pseudonymous_students': len(dates),
        'students_on_multiple_KST_dates': sum(len(ds) >= 2 for ds in dates.values()),
        'median_base_keydowns_per_record': median(lengths),
        'students_with_at_least_seven_records_of_at_least_76_base_keydowns': sum(n >= 7 for n in long_counts.values()),
    }


def audit_raw(groups, owners):
    rows = [row for group in groups.values() for row in group]
    duplicates = [group for group in groups.values() if len(group) > 1]
    cross_split = [group for group in duplicates if len({row[3] for row in group}) > 1]
    unique = [group[0] for group in groups.values() if len(group) == 1]
    return {
        'source': 'AI Hub 71518, Korean subject / keyboard sensor only, locally downloaded TS.zip and VS.zip',
        'comparison': 'Exact ordered lists of (timestamp, action, key) per stage across all stages. For the conservative view, drop every record in a duplicate group. Counts and median keystrokes describe STAGE_BASE only.',
        'split': {split: describe_raw(row for row in rows if row[3] == split) for split in SPLITS},
        'student_ids_shared_across_splits': len(owners['TS'] & owners['VS']),
        'combined': describe_raw(rows),
        'distinct_duplicated_signals': len(duplicates),
        'duplicate_groups_with_multiple_student_ids': sum(len({row[0] for row in group}) > 1 for group in duplicates),
        'all_records_in_duplicate_groups': sum(map(len, duplicates)),
        'distinct_signals_present_in_both_splits': len(cross_split),
        'affected_records_present_in_both_splits': dict(Counter(row[3] for group in cross_split for row in group)),
        'excluding_all_members_of_duplicate_groups': describe_raw(unique),
    }


def event_tuples(events):
    return [(event['time'], event['action'], event['key']) for event in events]


def load_records(archives, touch_capture):
    files = Counter()
    invalid = Counter()
    hashes = Counter()
    owners = defaultdict(set)
    questions = defaultdict(set)
    raw_groups = defaultdict(list)
    raw_owners = defaultdict(set)
    records = []
    for part, archive in archives.items():
        with ZipFile(archive) as bundle:
            for info in bundle.infolist():
                path = PurePosixPath(info.filename)
                if (info.is_dir() or path.parts[:1] != ('sensorData',)
                        or 'KOREAN' not in path.parts or 'keyboard' not in info.filename.lower()):
                    continue
                with bundle.open(info) as stream:
                    row = json.load(stream)
                if row['sensor_type'].lower() != 'keyboard':
                    continue
                data = row['data']
                ordered_stages = {stage: event_tuples(data[stage]) for stage in sorted(data)}
                all_stages = signal_stamp(list(ordered_stages.items()))
                base = ordered_stages['STAGE_BASE']
                student = row['student_id']
                date = datetime.fromisoformat(row['start_time']).astimezone(KST).date()
                keydowns = sum(action == 'KEY_DOWN' for _, action, _ in base)
                raw_groups[all_stages].append((student, date, keydowns, part))
                raw_owners[part].add(student)

                if path.name != 'keyboard.json':
                    continue
                question = int(re.fullmatch(r'Question(\d+)', path.parts[-2]).group(1))
                questions[question].add(row['question_id'])
                owners[part].add(student)
                files[part] += 1
                signature = signal_stamp(base)
                hashes[signature] += 1
                record, reason = parse_record(row, part, question, signature, touch_capture)
                if reason:
                    invalid[reason] += 1
                else:
                    records.append(record)
    assert not (owners['TS'] & owners['VS'])
    assert set(questions) == set(range(1, 21))
    assert all(len(ids) == 1 for ids in questions.values()) and len(set.union(*questions.values())) == len(questions)
    clean = [r for r in records if hashes[r['digest']] == 1]
    used_days = defaultdict(set)
    for r in clean:
        used_days[r['writer']].add(r['day'])
    counts = dict(archive_sha256=ARCHIVE_HASHES, files_by_split=dict(files), distinct_source_writers_by_split={p: len(s) for p, s in owners.items()},
                  source_writers_shared_by_split=len(owners['TS'] & owners['VS']),
                  source_questions=len(questions), identical_base_groups=sum(n > 1 for n in hashes.values()),
                  all_records_in_identical_base_groups=sum(n for n in hashes.values() if n > 1),
                  invalid_raw_records=dict(invalid), valid_nondeduplicated_records=len(records),
                  valid_unique_base_records=len(clean), valid_unique_base_writers=len(used_days),
                  writers_with_multiple_local_days_after_dedup=sum(len(ds) > 1 for ds in used_days.values()))
    return clean, counts, audit_raw(raw_groups, raw_owners)


def select(rows, rule):
    # Use the first complete qualifying enrollment/probe sequence. Never use performance to select it.
    ordered = sorted(rows, key=lambda r: (r['start'], r['q']))
    if rule == 'chronological':
        chosen = []
        for r in ordered:
            if all(r['q'] != v['q'] and r['start'] > v['end'] for v in chosen):
                chosen.append(r)
                if len(chosen) == 7:
                    return chosen[:5], chosen[5:]
    else:
        # Global holdout: multiples of 3 can never enter anyone's enrollment set.
        # Roles are defined by question number, with no temporal constraint between roles.
        by_question = {}
        for r in ordered:
            by_question.setdefault(r['q'], r)
        ref = [r for q, r in by_question.items() if q % 3][:5]
        probe = [r for q, r in by_question.items() if q % 3 == 0][:2]
        if len(ref) == 5 and len(probe) == 2:
            # Disallow actual shared time windows, including between nonadjacent question numbers.
            chronological = sorted(ref + probe, key=lambda r: r['start'])
            if all(a['end'] < b['start'] for a, b in zip(chronological, chronological[1:])):
                return ref, probe
    return None


def build_study(records, length, rule, part):
    stamp_counts = Counter(r['caps'][length][1] for r in records if length in r['caps'])
    by_person = defaultdict(list)
    for r in records:
        if r['part'] == part and length in r['caps'] and stamp_counts[r['caps'][length][1]] == 1:
            by_person[r['writer'], r['day']].append(r)
    seen = set()
    participants, speed, intervals, gap_minutes = [], [], [], []
    enrollment_questions, probe_questions = set(), set()
    enrolled_first = 0
    for key in sorted(by_person, key=lambda k: (hashlib.sha256(k[0].encode()).digest(), k[1])):
        match = select(by_person[key], rule)
        if match is None or key[0] in seen:
            continue
        seen.add(key[0])
        ref, probe = match
        assert len({r['q'] for r in ref + probe}) == 7
        assert len({(r['writer'], r['day']) for r in ref + probe}) == 1
        gap_minutes.append((min(r['start'] for r in probe) - max(r['end'] for r in ref)) / 60_000_000)
        enrolled_first += gap_minutes[-1] > 0
        enrollment_questions.update(r['q'] for r in ref)
        probe_questions.update(r['q'] for r in probe)
        enrollment, probes = ([r['caps'][length][0] for r in group] for group in (ref, probe))
        intervals.extend(c['cleanDigraphCount'] for c in enrollment + probes)
        speed.append([np.mean([np.median(np.log(c['stroke']['dds'])) for c in group]) for group in (enrollment, probes)])
        writer = f'w{len(participants):05d}'
        participants.append({'id': writer, 'sessions': [
            {'id': 'enroll_questions', 'role': 'enroll', 'captures': enrollment},
            {'id': 'probe_questions', 'role': 'query', 'captures': probes}]})
    if rule == 'global_question_holdout':
        assert enrollment_questions.isdisjoint(probe_questions)
    elif participants:
        assert enrolled_first == len(participants)
    design = dict(split=part, prefix_text_taps=length, question_selection=rule, writers=len(participants),
                  enrolled_first_in_time=enrolled_first, distinct_enrollment_questions=len(enrollment_questions),
                  distinct_probe_questions=len(probe_questions), questions_shared_between_roles_globally=len(enrollment_questions & probe_questions),
                  all_files_dropped_for_duplicate_prefix_in_both_splits=sum(n for n in stamp_counts.values() if n > 1),
                  median_valid_intervals_per_capture=float(np.median(intervals)) if intervals else None,
                  median_gap_minutes_after_enrollment=round(float(np.median(gap_minutes)), 2) if intervals else None)
    study = dict(schema='keyprint-korean-eval/v1', language='ko', input_mode='touch',
                 capture_protocol='published_recording_split', participants=participants)
    return study, np.array(speed), design
