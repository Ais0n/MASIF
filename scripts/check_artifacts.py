"""Validate released records and recompute the published trace-subset counts offline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPES = {'failed_tool_use', 'wrong_action', 'unnecessary_action', 'hallucination',
         'incomplete_action', 'reasoning_error', 'misunderstanding', 'malformed_output'}
ABSORPTIONS = {'self_corrected', 'self_workaround', 'peer_repair',
               'orchestrator_intervention', 'unresolved_propagation', 'trivial_issue'}
PAPER_COUNTS = {
    'webwalker_webwalkerqa': (680, 71, 609, 1090),
    'mrlx_browsecomp_plus': (100, 4, 96, 119),
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_path(value):
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'data':
        raise ValueError(f'Invalid release path: {value}')
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f'Missing release path: {value}')
    return path


def normalized_console(raw):
    blocks = []
    for message in raw.get('messages') or raw.get('trace') or []:
        role = str(message.get('role') or message.get('source') or message.get('tool_name') or 'trace')
        value = message.get('content', message)
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        blocks.append(f'----- {role} -----\n{content}')
    return '\n\n'.join(blocks)


def is_artifact(path):
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.name in ('ARTIFACT_MANIFEST.json', '.DS_Store'):
        return False
    if path.suffix == '.pyc':
        return False
    return not any(part.startswith('.') or part.endswith('.egg-info')
                   or part in ('__pycache__', 'outputs', 'checkpoints')
                   for part in relative.parts[:-1])


def check():
    from audit_artifacts import audit
    privacy = audit(ROOT)
    if privacy['findings']:
        raise ValueError('Distribution audit failed: ' + json.dumps(privacy['counts']))
    annotations = ROOT / 'data/q1_annotations'
    traces = load_jsonl(annotations / 'traces.jsonl')
    failures = load_jsonl(annotations / 'if_records.jsonl')
    excluded = load_jsonl(annotations / 'excluded_records.jsonl')
    cohort = json.loads((annotations / 'cohort.json').read_text(encoding='utf-8'))['traces']
    statistics = json.loads((annotations / 'statistics.json').read_text(encoding='utf-8'))
    assert len(traces) == 780 and len(failures) == 1209 and len(excluded) == 1
    assert len({(t['configuration'], t['task_id']) for t in traces}) == len(traces)
    assert len({f['if_id'] for f in failures}) == len(failures)
    assert not {f['if_id'] for f in failures} & {f['if_id'] for f in excluded}
    assert [f for t in traces for f in t['implicit_failures']] == failures
    assert len(cohort) == len(traces)
    for c, t in zip(cohort, traces):
        assert c == {k: t[k] for k in c}, t['task_id']

    truncated_prompts = []
    no_judge = Counter()
    raw_paths = set()
    for t in traces:
        assert t['configuration'] in PAPER_COUNTS
        assert type(t['semantic_success']) is bool
        assert t['num_implicit_failures'] == len(t['implicit_failures'])
        path = safe_path(t['raw_trace_path'])
        assert path not in raw_paths
        raw_paths.add(path)
        raw = json.loads(path.read_text(encoding='utf-8'))
        console = safe_path(t['console_path'])
        if not t.get('judge'):
            no_judge[t['configuration']] += 1
        if t['configuration'] == 'webwalker_webwalkerqa':
            steps = raw['events']
            result = json.loads(path.with_name('result.json').read_text(encoding='utf-8'))
            assert all(result[k] == t[k] for k in ('task_id', 'prompt', 'expected_answer',
                                                   'final_answer', 'semantic_success'))
        else:
            steps = raw.get('messages') or raw.get('trace') or []
            assert raw['metadata']['model'].casefold() == 'qwen3-8b'
            full_prompt = raw['metadata']['query']
            if t['prompt'] != full_prompt:
                assert len(t['prompt']) == 300 and full_prompt.startswith(t['prompt'])
                truncated_prompts.append(t['task_id'])
            assert console.read_text(encoding='utf-8') == normalized_console(raw)
        assert len(steps) == t['num_steps'], t['task_id']
        for failure in t['implicit_failures']:
            assert failure['task_id'] == t['task_id']
            assert failure['configuration'] == t['configuration']
            assert failure['error_type'] in TYPES
            assert failure['absorption_pattern'] in ABSORPTIONS
            assert type(failure['error_step']) is int
            assert 1 <= failure['error_step'] <= len(steps)
            step = steps[failure['error_step'] - 1]
            # MrlX serializes browser feedback as a user-role tool response.
            role = step['role']
            assert failure['error_agent'] == role or (
                failure['error_agent'] == 'browser' and role == 'user'
                and '<tool_response>' in step['content'])
    assert len(truncated_prompts) == 4
    expected_raw = set((ROOT / 'data/raw_traces').rglob('trace.json'))
    expected_raw.update((ROOT / 'data/raw_traces/browsecomp_plus').glob('*.json'))
    assert raw_paths == expected_raw

    groups = []
    for config, expected in PAPER_COUNTS.items():
        ts = [t for t in traces if t['configuration'] == config]
        fs = [f for t in ts for f in t['implicit_failures']]
        successes = sum(t['semantic_success'] for t in ts)
        observed = (len(ts), successes, len(ts) - successes, len(fs))
        assert observed == expected, (config, observed, expected)
        stat = dict(configuration=config, traces=len(ts), success=successes,
                    failure=len(ts) - successes, IFs=len(fs))
        assert stat in statistics
        outcomes = {}
        for value, label in ((True, 'success'), (False, 'failure')):
            subset = [t for t in ts if t['semantic_success'] == value]
            count = sum(t['num_implicit_failures'] for t in subset)
            bearing = sum(bool(t['implicit_failures']) for t in subset)
            outcomes[label] = dict(traces=len(subset), if_bearing_traces=bearing,
                                   IFs=count, prevalence=bearing / len(subset),
                                   mean_ifs_per_trace=count / len(subset))
        groups.append(dict(**stat, outcomes=outcomes,
                           error_types=dict(sorted(Counter(f['error_type'] for f in fs).items())),
                           absorption_patterns=dict(sorted(Counter(f['absorption_pattern'] for f in fs).items()))))
    manifest_path = ROOT / 'ARTIFACT_MANIFEST.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        actual_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if is_artifact(p)}
        assert actual_files == set(manifest['files']), 'Manifest file inventory differs'
        for name, expected in manifest['files'].items():
            path = ROOT / name
            assert path.stat().st_size == expected['bytes'], name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
    return dict(traces=len(traces), IFs=len(failures), excluded_records=len(excluded),
                configurations=groups, truncated_summary_prompts=truncated_prompts,
                records_without_retained_outcome_judge=dict(no_judge))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write recomputed JSON statistics')
    args = parser.parse_args()
    report = check()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
