"""Load immutable model releases and verified FACT supplements."""

import importlib.util
import json
from pathlib import Path
import subprocess


MAIN = '4afdaa2a4a3e6063a6cfbeb82b893481866dd471'
PR = 'b999eccc0d0adbba2e0b386443f290cd4fb728c5'
EARLY = 'e7529d3f6be9b89f5f71a6958dadaa887e113c8f'
UPSTREAM = 'https://github.com/qwertypotter6873/keyprint-guard.git'
FACT_PROTOCOLS = {
    'en': ((3, 5), (1, 7), (211.81, 211.68)),
    'ko': ((4, 6), (2, 8), (235.11, 235.26)),
}
ENGLISH_HASHES = {
    1: 'f068d211097f4e4e0ea6236a56bf6c178e977939b369a96cb4f9b387112ab025',
    3: '3434adaba70ec155ff57915fce40ef22878c22482bb59d362c5db4bc40dea556',
    5: '98c8b07bdc7be4eeaa03037aa4283a914e50ed174f7e78de0322f2656f5ffe19',
    7: '20765c0d290e80f1fce0c1fe9f4b226e25e804d8c2de082db0c6624481f0a2f1',
}


def git(directory, *args, **options):
    return subprocess.run(['git', '-C', str(directory), *args], **options)


def has_revision(directory, revision):
    return git(directory, 'cat-file', '-e', f'{revision}^{{commit}}',
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def pinned_sources(repo, work):
    revisions = (MAIN, PR, EARLY)
    if all(has_revision(repo, revision) for revision in revisions):
        return repo

    cache = work / 'frozen-releases.git'
    git(work, 'init', '--quiet', '--bare', str(cache), check=True)
    git(cache, 'remote', 'add', 'origin', UPSTREAM, check=True)
    git(cache, 'config', 'remote.origin.promisor', 'true', check=True)
    git(cache, 'fetch', '--quiet', '--no-tags', '--depth=1', '--filter=blob:none',
        'origin', *revisions, check=True, timeout=120)
    for revision in revisions:
        fetched = git(cache, 'rev-parse', '--verify', f'{revision}^{{commit}}',
                      check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        if fetched != revision:
            raise ValueError(f'Expected immutable Git commit {revision}, received {fetched}.')
    return cache


def git_file(directory, path):
    return git(directory, 'show', f'{PR}:{path}', check=True, text=True, encoding='utf-8',
               stdout=subprocess.PIPE, timeout=120).stdout


def load_adapter(sources, work):
    module_file = work / 'eval_korean_fact.py'
    module_file.write_text(git_file(sources, 'src/eval_korean_fact.py'), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('eval_korean_fact', module_file)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def fact_studies(adapter, archive_dir, work):
    adapter.ARCHIVE_HASHES.update(ENGLISH_HASHES)
    staged = None
    if archive_dir is not None:
        staged = work / 'supplements'
        staged.mkdir()
        for number in adapter.ARCHIVE_HASHES:
            filename = f'mmc{number}.zip'
            archive = archive_dir / filename
            if not archive.is_file():
                archive = archive_dir / ('fact.zip' if number == 1 else f'fact{number}.zip')
            if not archive.is_file():
                raise ValueError(f'Could not find {filename} inside --fact-dir.')
            (staged / filename).symlink_to(archive.resolve())

    studies = {}
    design = {}
    for language, (enrollment, query, published_means) in FACT_PROTOCOLS.items():
        # The pilot adds probes from four of the Korean writer's five unused reference records.
        role = 'all_reference' if language == 'ko' else 'enroll'
        adapter.RELEASES = {role: enrollment, 'query': query}
        reference, _, reference_signals, reference_meta = adapter.extract_split(role, staged)
        probe, _, probe_signals, probe_meta = adapter.extract_split('query', staged)
        if reference_signals & probe_signals:
            raise ValueError(f'The FACT {language} partitions share paired timing observations.')
        means = [meta['reported_keystrokes_per_recording_mean'] for meta in (reference_meta, probe_meta)]
        if tuple(round(value, 2) for value in means) != published_means:
            raise ValueError(f'The FACT {language} archive layout differs from the published design.')
        enrolled = {writer: records[:5] for writer, records in reference.items()}
        study = adapter.study_observations(enrolled, probe, 'prompt_length')
        study['language'] = language
        studies[language] = study
        design[language] = {
            'supplement_reference': enrollment,
            'supplement_probe': query,
            'mean_full_keystrokes': means,
        }
        if language == 'ko':
            expanded_probes = {writer: probe[writer] + records[5:9] for writer, records in reference.items()}
            historical = adapter.study_observations(enrolled, expanded_probes, 'prompt_length')
    return studies, design, dict(sorted(adapter.ARCHIVE_HASHES.items())), historical


def score_studies(repo, sources, work, aihub, fact, historical):
    # Amend only harness setup and language metadata. Preserve model inference,
    # capture validation, template formation, rosters and report calculations.
    harness = git_file(sources, 'web/scripts/eval_korean_ab.mjs')
    entry = 'try {\n  const { values } = parseArgs('
    if harness.count(entry) != 1:
        raise ValueError('Unexpected entry point in the frozen web harness.')
    harness = harness.split(entry, 1)[0]
    for old, new in (
        ("const root = fileURLToPath(new URL('../../', import.meta.url));", f'const root = {json.dumps(str(repo))};'),
        ("const local = optionalGit(root, ['rev-parse', '--show-toplevel']);", f'const local = {json.dumps(str(sources))};'),
        ("study.language === 'ko'", "['ko', 'en'].includes(study.language)"),
        ('    captureProtocol,\n', '    captureProtocol,\n    language: study.language,\n'),
        ("    language: 'ko',", '    language: study.language,'),
    ):
        if harness.count(old) != 1:
            raise ValueError('Unexpected structure in the frozen web harness.')
        harness = harness.replace(old, new, 1)
    driver = Path(__file__).with_name('score-driver.mjs').read_text(encoding='utf-8')
    script = work / 'score.mjs'
    script.write_text(harness + '\n' + driver, encoding='utf-8')
    studies = {'aihub': aihub, 'fact': fact, 'historical': historical}
    run = subprocess.run(['node', str(script), MAIN, PR, EARLY], cwd=repo, check=True,
                         input=json.dumps(studies, separators=(',', ':'), allow_nan=False),
                         stdout=subprocess.PIPE, text=True, encoding='utf-8')
    return json.loads(run.stdout)
