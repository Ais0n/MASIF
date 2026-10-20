"""Outcome GRPO and outcome-conditioned IF-GRPO (paper equations 1-4)."""
import math
import random
import statistics
from pathlib import Path

from .common import arguments, evaluate, grade, load_hook, read_tasks, write_json
from .judge import Judge, cost
from .runtime import run


def advantages(outcomes, costs=None):
    if len(outcomes) != 8 or any(y not in (0, 1) for y in outcomes):
        raise ValueError('Exactly eight binary outcomes required')
    successes = [i for i, y in enumerate(outcomes) if y]
    if len(successes) in (0, 8):
        return [0.0] * 8
    mean = statistics.mean(outcomes)
    std = statistics.stdev(outcomes)  # sample standard deviation, ddof=1
    result = [(y - mean) / (std + 1e-6) for y in outcomes]
    if costs is not None:
        if len(costs) != 8:
            raise ValueError('Exactly eight costs required')
        values = [costs[i] for i in successes]
        if any(c is None or not math.isfinite(c) or c < 0 for c in values):
            raise ValueError('Finite nonnegative successful IF costs required')
        low, high = min(values), max(values)
        if len(successes) > 1 and high > low:
            for i in successes:
                z = (high - costs[i]) / (high - low) - 0.5
                result[i] *= 1 + z
    return result


def update(policy, optimizer, traces, values, epochs):
    import torch
    from .policy import token_logprobs
    if not any(values):
        return False  # No optimizer.step(): AdamW decay must also be skipped.
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for trace, advantage in zip(traces, values):
            if advantage == 0:
                continue
            token_count = sum(len(s['old_logprobs']) for s in trace['segments'])
            if token_count == 0:
                raise ValueError('No policy tokens in training trajectory')
            for segment in trace['segments']:
                current = token_logprobs(policy.model, segment['ids'], segment['prompt_length'])
                old = torch.tensor(segment['old_logprobs'], device=current.device)
                ratio = (current - old).exp()
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.28) * advantage)
                # Mean over generated tokens per trajectory, then over the eight trajectories.
                loss = -surrogate.sum() / (token_count * len(traces))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite PPO objective')
                loss.backward()
        optimizer.step()
    return True


def evaluate_heldout(policy, judge, tasks, grader, architecture, seed):
    policy.temperature = 0.0
    policy.record_segments = False
    rows = []
    for task in tasks:
        trace = run(policy, task['prompt'], architecture, seed)
        errors = judge.annotate(task['prompt'], trace['steps'])
        rows.append(dict(task_id=task['id'], success=evaluate(grader, task, trace['final']),
                         if_cost=cost(errors), if_count=len(errors),
                         generated_tokens=trace['generated_tokens']))
    n = len(rows)
    tokens = sum(r['generated_tokens'] for r in rows)
    total_cost = sum(r['if_cost'] for r in rows)
    total_count = sum(r['if_count'] for r in rows)
    return dict(tasks=rows, accuracy=sum(r['success'] for r in rows) / n,
                if_cost_per_task=total_cost / n, if_count_per_task=total_count / n,
                if_cost_per_1k_tokens=1000 * total_cost / tokens if tokens else None,
                if_count_per_1k_tokens=1000 * total_count / tokens if tokens else None)


def main():
    parser = arguments(__doc__)
    parser.add_argument('--model', required=True, help='Local or public causal-LM model path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--objective', choices=['grpo', 'if_grpo'], default='if_grpo')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--ppo-epochs', type=int, default=1)
    parser.add_argument('--eval-tasks', help='Optional held-out task JSONL, excluded from updates')
    args = parser.parse_args()
    if args.steps < 1 or args.ppo_epochs < 1:
        parser.error('Steps and PPO epochs must be positive')
    import torch
    from .policy import LocalPolicy
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tasks = read_tasks(args.tasks)
    heldout = read_tasks(args.eval_tasks) if args.eval_tasks else []
    if {t['id'] for t in tasks} & {t['id'] for t in heldout}:
        raise ValueError('Training and held-out task ids overlap')
    grader = load_hook(args.grader) if args.grader else grade
    policy = LocalPolicy(args.model, args.device)
    judge = Judge() if args.objective == 'if_grpo' or heldout else None
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-6, betas=(0.9, 0.98), weight_decay=0.1)
    for step in range(args.steps):
        task = rng.choice(tasks)
        traces = [run(policy, task['prompt'], args.architecture, args.seed + step * 8 + slot)
                  for slot in range(8)]
        outcomes = [evaluate(grader, task, t['final']) for t in traces]
        costs = None
        if args.objective == 'if_grpo' and 0 < sum(outcomes) < 8:
            costs = [cost(judge.annotate(task['prompt'], t['steps'])) if y else None
                     for y, t in zip(outcomes, traces)]
        values = advantages(outcomes, costs)
        updated = update(policy, optimizer, traces, values, args.ppo_epochs)
        print(f'step={step + 1} successes={sum(outcomes)}/8 updated={updated}', flush=True)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    policy.model.save_pretrained(target)
    policy.tokenizer.save_pretrained(target)
    if heldout:
        report = evaluate_heldout(policy, judge, heldout, grader, args.architecture, args.seed)
        write_json(target / 'evaluation.json', report)
        print(f"heldout_accuracy={report['accuracy']:.6f}")


if __name__ == '__main__':
    main()
