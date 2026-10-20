"""Matched no-op and single-IF repair from the same runtime boundary."""
import random
from collections import defaultdict

from .common import RemotePolicy, arguments, evaluate, grade, load_hook, read_tasks, write_json
from .judge import Judge
from .runtime import budget_for, run


def trace_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['task_id'], row.get('rerun', 0))].append(row)
    traces = []
    for (task_id, rerun), interventions in grouped.items():
        original = interventions[0]['original_success']
        if any(r['original_success'] != original for r in interventions):
            raise ValueError('Inconsistent original outcomes within one trace')
        n = len(interventions)
        traces.append(dict(task_id=task_id, rerun=rerun, original_success=original, pairs=n,
                           noop_success=sum(r['noop_success'] for r in interventions) / n,
                           repair_success=sum(r['repair_success'] for r in interventions) / n))
    return traces


def summarize(rows):
    traces = trace_means(rows)
    result = {}
    for original, name in [(1, 'success_preservation'), (0, 'failure_rescue')]:
        subset = [r for r in traces if r['original_success'] == original]
        if subset:
            result[name] = {condition: sum(r[condition] for r in subset) / len(subset)
                            for condition in ('noop_success', 'repair_success')}
            result[name]['traces'] = len(subset)
            result[name]['pairs'] = sum(r['pairs'] for r in subset)
    return result


def rescue_interval(rows, seed, resamples=5000):
    clusters = defaultdict(list)
    for row in trace_means(rows):
        if not row['original_success']:
            clusters[row['task_id']].append(row['repair_success'] - row['noop_success'])
    if not clusters:
        return None
    rng = random.Random(seed)
    groups = list(clusters.values())
    differences = []
    for _ in range(resamples):
        sample = [value for group in rng.choices(groups, k=len(groups)) for value in group]
        differences.append(100 * sum(sample) / len(sample))
    differences.sort()
    return dict(delta_pp=100 * sum(map(sum, groups)) / sum(map(len, groups)),
                ci95_pp=[differences[int(0.025 * resamples)], differences[int(0.975 * resamples)]])


def main():
    parser = arguments(__doc__)
    parser.add_argument('--sample-per-outcome', type=int, default=50)
    parser.add_argument('--reruns', type=int, default=1, help='Fresh runs per task before sampling')
    args = parser.parse_args()
    if args.sample_per_outcome < 1 or args.reruns < 1:
        parser.error('Sample size and reruns must be positive')
    policy, judge = RemotePolicy(), Judge()
    grader = load_hook(args.grader) if args.grader else grade
    pools = {0: [], 1: []}
    for task in read_tasks(args.tasks):
        for rerun in range(args.reruns):
            trace = run(policy, task['prompt'], args.architecture, args.seed + rerun, capture=True)
            errors = judge.annotate(task['prompt'], trace['steps'])
            eligible = [e for e in errors if str(e['error_step']) in trace['snapshots']]
            if eligible:
                success = evaluate(grader, task, trace['final'])
                pools[success].append((task, rerun, trace, eligible))
    rng = random.Random(args.seed)
    rows = []
    for success, pool in pools.items():
        if len(pool) < args.sample_per_outcome:
            raise ValueError(f'Need {args.sample_per_outcome} IF-bearing traces per outcome; '
                             f'found {len(pools[1])} successful and {len(pools[0])} failed. '
                             'Add tasks or reruns, or reduce --sample-per-outcome.')
        for task, rerun, trace, errors in rng.sample(pool, args.sample_per_outcome):
            # Every IF is repaired separately from the original boundary, not cumulatively.
            for error in errors:
                index = error['error_step']
                state = trace['snapshots'][str(index)]
                step = trace['steps'][index - 1]
                noop = run(policy, state=state, replacement=step['content'],
                           replacement_tokens=step['generated_tokens'], replacement_truncated=step['truncated'])
                correction = judge.repair(task['prompt'], trace['steps'][:index - 1], step, error,
                                          max_tokens=budget_for(state))
                if judge.last_tokens is None:
                    raise RuntimeError('Repair endpoint must report completion_tokens')
                repair = run(policy, state=state, replacement=correction,
                             replacement_tokens=judge.last_tokens,
                             replacement_truncated=judge.last_truncated)
                rows.append(dict(task_id=task['id'], rerun=rerun, error_step=index,
                                 original_success=success, noop_success=evaluate(grader, task, noop['final']),
                                 repair_success=evaluate(grader, task, repair['final'])))
    write_json(args.output, dict(pairs=rows, rates=summarize(rows),
                                failure_rescue_difference=rescue_interval(rows, args.seed)))


if __name__ == '__main__':
    main()
