"""Offline regression checks for the third audit; no credentials required."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masif.common import grade, prediction
from masif.replay import rescue_interval, summarize, trace_means

def row(task, rerun, step, original, noop, repair):
    return dict(task_id=task, rerun=rerun, error_step=step,
                original_success=original, noop_success=noop, repair_success=repair)

rows = [row('a', 0, 1, 0, 0, 1)]
rows += [row('b', 0, i, 0, 0, 0) for i in range(1, 10)]
rates = summarize(rows)['failure_rescue']
assert rates == dict(noop_success=0.0, repair_success=0.5, traces=2, pairs=10)
assert rescue_interval(rows, 42)['delta_pp'] == 50.0
assert rescue_interval(rows, 42) == rescue_interval([rows[0]] + rows[1:] * 2, 42)
print('PASS unequal IF counts, trace/pair counts, and within-trace replication invariance')

# Two failed reruns of one task and one of another must remain separate traces,
# but reruns of the same task must move together in bootstrap resampling.
rows += [row('a', 1, 1, 0, 1, 0), row('a', 2, 1, 1, 1, 1)]
assert len(trace_means(rows)) == 4
assert summarize(rows)['success_preservation']['traces'] == 1
assert summarize(rows)['failure_rescue']['repair_success'] == 1 / 3
rng = random.Random(42)
draws = []
for _ in range(5000):
    selected = rng.choices([[1, -1], [0]], k=2)
    flat = [v for group in selected for v in group]
    draws.append(100 * sum(flat) / len(flat))
draws.sort()
assert rescue_interval(rows, 42) == dict(delta_pp=0.0,
                                        ci95_pp=[draws[125], draws[4875]])
assert rescue_interval([row('x', 0, 1, 1, 1, 0)], 42) is None
assert summarize([]) == {}
try:
    trace_means([row('x', 0, 1, 0, 0, 0), row('x', 0, 2, 1, 0, 0)])
except ValueError:
    pass
else:
    raise AssertionError('Inconsistent outcomes accepted')
print('PASS rerun separation, task-cluster bootstrap reference, empty strata, inconsistent outcomes')

for text, expected in [
    ('FINAL: 5', '5'), ('Reasoning\nFINAL: 5\n \n', '5'),
    ('FINAL:\t5\t', '5'), ('FINAL: 4\nFINAL: 5', '5'),
    ('', None), ('FINAL:', None), ('FINAL:   \n', None),
    ('FINAL: 5\nI retract that answer; my tentative result is 5', None),
    ('FINAL: 5\nMore commentary', None), ('FINAL:\n5', None),
]:
    assert prediction(text) == expected, repr(text)
assert grade({'answer': '5'}, 'FINAL: 5\nI retract that answer; my tentative result is 5') == 0
assert grade({'answer': '0.5'}, 'FINAL: 1/2') == 1
print('PASS final-line contract, retracted answers, whitespace, empty markers, rational grading')
