"""Fresh best-of-N generation, trace-only IF scoring, and tuning-only PRM selection."""
import math
import random

from .common import RemotePolicy, arguments, evaluate, grade, load_hook, read_tasks, write_json
from .judge import Judge, cost
from .runtime import run

POOLINGS = ('min', 'mean', 'geometric', 'last')


def pool(scores, method):
    if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('PRM must supply nonempty finite normalized step scores')
    if method == 'min':
        return min(scores)
    if method == 'mean':
        return sum(scores) / len(scores)
    if method == 'last':
        return scores[-1]
    if method == 'geometric':
        return 0.0 if 0 in scores else math.exp(sum(math.log(s) for s in scores) / len(scores))
    raise ValueError(method)


def select(candidates, method, pooling='mean', weight=1.0):
    if method == 'if':
        return min(candidates, key=lambda c: c['risk'])
    return max(candidates, key=lambda c: weight * pool(c['prm'], pooling)
               + (1 - weight) * (1 - c['risk']))


def accuracy(groups, method, pooling='mean', weight=1.0):
    return sum(select(cs, method, pooling, weight)['success'] for cs in groups) / len(groups)


def main():
    parser = arguments(__doc__)
    parser.add_argument('--prm', help='Optional module:function(prompt, steps) -> step scores')
    parser.add_argument('--prm-scale', choices=['unit', 'prints'], default='unit')
    args = parser.parse_args()
    tasks = read_tasks(args.tasks)
    if len(tasks) < 5:
        parser.error('At least five tasks are required for a 20%/80% tuning/test split')
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    tune_count = len(tasks) // 5
    policy, judge = RemotePolicy(), Judge()
    grader = load_hook(args.grader) if args.grader else grade
    prm = load_hook(args.prm) if args.prm else None
    groups = []
    for task in tasks:
        candidates = []
        for slot in range(16):
            trace = run(policy, task['prompt'], args.architecture, args.seed + slot)
            steps = [{'agent': s['agent'], 'content': s['content']} for s in trace['steps']]
            errors = judge.annotate(task['prompt'], steps)
            scores = list(map(float, prm(task['prompt'], steps))) if prm else None
            if scores is not None:
                if args.prm_scale == 'prints':
                    scores = [(s + 4) / 8 for s in scores]
                pool(scores, 'mean')  # Reject malformed scores before tuning.
            candidates.append(dict(id=slot, risk=cost(errors) / max(1, len(steps)), prm=scores,
                                   success=evaluate(grader, task, trace['final'])))
        groups.append(candidates)
    reports = {}
    for n in (4, 8, 16):
        tune = [cs[:n] for cs in groups[:tune_count]]
        test = [cs[:n] for cs in groups[tune_count:]]
        report = dict(random_accuracy=sum(sum(c['success'] for c in cs) / n for cs in test) / len(test),
                      if_accuracy=accuracy(test, 'if'),
                      if_selected=[select(cs, 'if')['id'] for cs in test])
        if prm:
            best_pool = max(POOLINGS, key=lambda p: accuracy(tune, 'prm', p))
            settings = [(p, w / 10) for p in POOLINGS for w in range(11)]
            mixed_pool, weight = max(settings, key=lambda s: accuracy(tune, 'mixed', *s))
            report.update(prm_accuracy=accuracy(test, 'prm', best_pool), prm_pooling=best_pool,
                          mixed_accuracy=accuracy(test, 'mixed', mixed_pool, weight),
                          mixed_pooling=mixed_pool, mixed_prm_weight=weight)
        reports[n] = report
    write_json(args.output, dict(tuning_tasks=[t['id'] for t in tasks[:tune_count]],
                                test_tasks=[t['id'] for t in tasks[tune_count:]], results=reports))


if __name__ == '__main__':
    main()
