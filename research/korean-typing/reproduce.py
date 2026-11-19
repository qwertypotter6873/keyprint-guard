"""Recreate the aggregate Korean typing evaluation from the original datasets."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from aihub import PROTOCOLS, SPLITS, build_study, load_records, verified_archives
from early_cutoff import run as evaluate_early_cutoffs
from frozen import EARLY, MAIN, PR, fact_studies, git_file, load_adapter, pinned_sources, score_studies
from margin_sweep import MARGINS, THRESHOLD, all_rosters, fact_rosters, plot_validation
from metrics import calibration_report, summarize, verify_formulas
from ranking_pilot import assemble_features, run as rank


ROOT = Path(__file__).resolve().parents[2]
AIHUB = 'https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71518&topMenu=100'
FACT = 'https://doi.org/10.1016/j.patcog.2020.107556'


def evaluate(archives, archives_fact, work):
    verify_formulas()
    sources = pinned_sources(ROOT, work)
    adapter = load_adapter(sources, work)
    records, audit, raw_audit = load_records(archives, adapter.touch_capture)
    fact, fact_design, fact_hashes, historical = fact_studies(adapter, archives_fact, work)

    designs = []
    ready = []
    for length, rule in PROTOCOLS:
        for split in SPLITS:
            case = build_study(records, length, rule, split)
            design = case[2]
            designs.append(design)
            print(f'AI Hub {split}, {length} taps, {rule}: {design["writers"]} qualified writers', flush=True)
            if design['writers'] >= 6:
                ready.append(case)

    print('Scoring the AI Hub and FACT studies with the frozen releases.', flush=True)
    scores = score_studies(ROOT, sources, work, [study for study, _, _ in ready], fact, historical)
    curves = []
    primary = {}
    rank_cases = {}
    for (study, speed, design), answer in zip(ready, scores['aihub'], strict=True):
        design['results'] = summarize(answer, speed)
        details = {key: value for key, value in design.items() if key != 'results'}
        curves.append(dict(source='AI Hub 71518', **details, curve=all_rosters(answer['pr'], design['results'])))
        if design['prefix_text_taps'] == 40:
            split = design['split']
            if design['question_selection'] == 'chronological':
                primary[split] = answer
                label = 'train' if split == 'TS' else 'validation'
                rank_cases[label] = assemble_features(study, answer['pr'], 'all_five_person_rosters')
            elif split == 'VS':
                rank_cases['nonoverlapping_prompts'] = assemble_features(study, answer['pr'], 'all_five_person_rosters')

    for language in ('ko', 'en'):
        study = fact[language]
        answer = scores['fact'][language]
        n = len(study['participants'])
        curves.append(dict(source='FACT', language=language, writers=n, known_trials=n * 5, outsider_trials=n * 5,
                           prefix_text_taps=76, curve=fact_rosters(answer, n)))
        rank_cases['fact_' + language] = assemble_features(study, answer, 'published_project_fixed_rosters')
    fact_reports = [scores['fact'][language]['report'] for language in ('en', 'ko')]

    published_text = git_file(sources, 'models/korean_fact_ab.json')
    published = json.loads(published_text)
    for release in ('main', 'candidate'):
        for field in ('enrolled_probes', 'unknown_writer_probes'):
            assert fact_reports[1][release][field] == published[release][field]

    protocol = {
        'corpus': 'Korean subject keyboard.json, STAGE_BASE, dates in Asia/Seoul. No labels or non-keyboard sensor files are opened.',
        'inputs': 'First 40 (primary) or 76 (sensitivity) printable or space taps per question; at least half are Hangul. Adjacent KEY_DOWN differences of 6 to 2200 ms only, cutting across intervening controls. No physical key coordinates, measured key holds, behavior, answer text, absolute timestamp or identifier in the input to the scorer.',
        'task': 'Five enrollment answers and one probe of two answers, seven distinct questions per writer. No overlap between actual timing windows. Chronological primary: all five reference answers precede both probes. Alternate global question holdout: question numbers divisible by three are always test questions; reference may follow probe.',
        'sampling': 'Use every eligible writer once per protocol and evaluate rosters within one publisher split. For each writer, average exactly over all choices of four other candidates for a known probe, or five other candidates for an outsider probe. Then give every writer equal weight. Separate fixed-roster counts use the web engine directly, seed 71518.',
        'metrics': 'top1 is the ranking before rejection. accept_correct, accept_wrong and reject_known have all known-writer probes as their denominator. accept_outsider has outsider probes as its denominator. Intervals resample writers 5000 times conditional on the observed candidate pool. Rates are fractions, not percentages.',
        'data_use': 'Both deployed networks are frozen. The exploratory two-cutoff diagnostic uses TS only to select settings and evaluates them on VS. Each capture uses at least eight eligible inter-tap intervals. This is an offline smart-device timing assessment, without a copied-text or browser-event reconstruction.',
    }
    result = dict(source=AIHUB, main=MAIN, pr_5=PR, protocol=protocol, audit=audit, studies=designs,
                  exploratory_cutoff_fit=calibration_report(primary['TS'], primary['VS']))
    margin = dict(pr_5_commit=PR, input_mode='touch', similarity_threshold=THRESHOLD, default_margin=MARGINS[0],
                  aihub_archive_sha256=audit['archive_sha256'],
                  protocol='AI Hub uses every eligible 5-person roster within the original participant split. FACT uses the five-person rosters of the prior published-project benchmark. Enrollment is five captures, each probe is two captures. No weight or similarity cutoff is fit.',
                  units='Rates and paired differences are in fractions. Uncertainty resamples writers, conditional on the given candidate pool. Known and outsider measures have separate denominators.',
                  groups=curves)
    bilingual = dict(source=FACT, design=fact_design,
                     protocol='same 50 smartphone typists; within-language 5 enrolled, 2 query captures, 5 candidates; first 76 taps; down-down timing only; enrollment and probe are distinct publisher partitions with dates unavailable',
                     published_archive_sha256=fact_hashes, reports=fact_reports)
    original_touch_rule = json.loads(git_file(sources, 'models/evaluation_report.json'))['feature_upgrade']['arena_protocol']['touch']
    print('Fitting the exploratory ranking pilot on the AI Hub development split.', flush=True)
    pilot = rank(rank_cases, audit['archive_sha256'], original_touch_rule, PR)
    return {
        'results.json': result,
        'aihub-71518-audit.json': raw_audit,
        'fact-en-ko-comparison.json': bilingual,
        'fact-early-cutoff-review.json': evaluate_early_cutoffs(scores['historical'], EARLY, published_text),
        'margin-sweep.json': margin,
        'ranking-pilot.json': pilot,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aihub', type=Path, required=True, help='Approved local AI Hub 71518 directory containing TS.zip and VS.zip.')
    parser.add_argument('--fact-dir', type=Path, help='Directory with the eight FACT mmc*.zip supplements. Defaults to the publisher download.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Write the aggregate JSON reports to this directory.')
    parser.add_argument('--plot', action='store_true', help='Render margin-sweep.png from the calculated rates using Matplotlib.')
    args = parser.parse_args()
    try:
        archives = verified_archives(args.aihub)
        with tempfile.TemporaryDirectory(prefix='keyprint-korean-reproduce-') as directory:
            reports = evaluate(archives, args.fact_dir, Path(directory))
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        for filename, report in reports.items():
            (output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if args.plot:
            plot_validation(reports['margin-sweep.json'], output / 'margin-sweep.png')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Wrote aggregate reports to {output}.', flush=True)


if __name__ == '__main__':
    main()
