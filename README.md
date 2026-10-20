# MASIF

Code and data for **MASIF: Implicit Failures for Process Observability in Multi-Agent Systems**.

The repository provides implementations of counterfactual replay, IF-based
best-of-N selection, and IF-GRPO, together with execution traces and IF
annotations for WebWalkerQA and BrowseComp-Plus. The [data guide](data/README.md)
describes the dataset, annotation schema, and file organization.

| Path | Contents |
| --- | --- |
| `masif/`, `pyproject.toml` | Installable code for replay, best-of-N selection, and IF-GRPO |
| `data/raw_traces/` | 780 raw traces |
| `data/q1_annotations/` | Trace outcomes and cleaned IF annotations |
| `data/traces/` | Normalized MrlX console files |
| `scripts/check_artifacts.py` | Offline data and file-integrity validation |
| `scripts/audit_artifacts.py` | Credential and operational-metadata checks |
| `ARTIFACT_MANIFEST.json` | File inventory and SHA-256 checksums |

To validate the dataset and recompute its statistics:

```bash
python scripts/check_artifacts.py
```

To build a distribution archive:

```bash
python scripts/build_release.py --output ../masif-publication.zip
```

## Choose a workflow

| Workflow | What it does | What you need |
| --- | --- | --- |
| Counterfactual replay | Compares continuing a trace with its original step against continuing with a corrected step. | Tasks, a policy endpoint, and a judge endpoint. |
| Best-of-N selection | Generates candidate traces and selects the one with the lowest IF risk. | At least five tasks, a policy endpoint, and a judge endpoint. PRM comparisons are optional. |
| IF-GRPO training | Trains a local shared policy using task outcomes and IF costs. | Training tasks, a local causal model, training dependencies, and a judge endpoint. Held-out tasks are optional. |

## Set up

### 1. Install the package

Use Python 3.10 or later. Clone this repository or extract the combined release.
From its root (the directory containing `pyproject.toml`), create and activate
a virtual environment, then install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For local training or the included local math PRM adapters, also install:

```bash
python -m pip install -e '.[train]'
```

Run the commands below from this directory with the environment activated.

The included offline code checks use mocked endpoints and need no credentials:

```bash
python tests/smoke_check.py
python tests/regression_checks.py
```

### 2. Prepare tasks

Supply a UTF-8 JSONL file with one task per line. Each task needs a unique `id`, a nonempty `prompt`, and an `answer`:

```json
{"id":"example-1","prompt":"What is 2 + 3?","answer":"5"}
```

Include any answer options in `prompt`. The `answer` field is used only by the outcome evaluator. It is never passed to the policy or IF judge. Agents are instructed to end their response with `FINAL: <answer>` on the last line.

The built-in grader checks exact option/text matches and exact rational-number equivalence. For symbolic math or benchmark-specific evaluation, provide a custom grader:

```text
--grader your_module:grade
```

The importable function must implement `grade(task, final_text) -> 0 or 1`. All three workflows accept this option.

### 3. Configure endpoints

Replay and selection require both a policy endpoint and a judge endpoint.
Set the following environment variables using your own chat-completions service:

```bash
export POLICY_URL='<full policy chat-completions URL>'
export POLICY_MODEL='<policy model>'
export POLICY_KEY='<credential, if required>'

export JUDGE_URL='<full judge chat-completions URL>'
export JUDGE_MODEL='<judge model>'
export JUDGE_KEY='<credential, if required>'
```

Use full endpoint URLs, not service roots. The service must accept the chat-completions request format and Bearer authentication when a key is needed. You may omit `POLICY_KEY` or `JUDGE_KEY` if the endpoint requires no credential. Credentials are read from the environment and are never written to output.

Policy and repair responses must report `usage.completion_tokens`. Judge annotations must be valid JSON with the requested schema. Full task and trace contexts are sent without silent truncation, so choose models with enough context capacity. Transport failures and invalid annotation JSON are retried up to three times; persistent failures stop the run.

Local training uses a local policy and does not require `POLICY_*` variables. It requires `JUDGE_*` for IF-GRPO or whenever `--eval-tasks` is supplied, including held-out evaluation of the outcome-only GRPO baseline.

## Available architectures

Choose a runtime with `--architecture`. All agents within a runtime share one policy.

| Architecture | Interaction | Generation limits |
| --- | --- | --- |
| `mathchat` | An assistant works with a proxy that executes Python computations. | At most 15 assistant calls; 2,048 tokens per call; 8,192 generated tokens per trace. |
| `teamwork` | Four private initial answers, four discussion rounds with immediate peer broadcast, then one final team answer. | 21 calls; 1,024 tokens per call. |
| `self_moa` | Six independent proposals followed by one aggregation. | 7 calls; 1,024 tokens per call. |


## Counterfactual replay

Replay tests whether correcting a local IF changes the final outcome compared with continuing from the same point without changing the step.

```bash
python -m masif.replay --tasks tasks.jsonl --architecture mathchat \
  --sample-per-outcome 50 --reruns 2 --output outputs/replay.json
```

The command generates fresh traces, annotates their IFs, and randomly samples 50 originally successful and 50 originally failed traces that contain eligible IFs. If either group is too small, add tasks, increase `--reruns`, or reduce `--sample-per-outcome`. A smaller sample is useful for checking your setup.

Each eligible IF is evaluated independently:

1. Restore the scheduler, per-agent histories, token budget, and successful computation programs immediately before the faulty agent step.
2. Continue once using the original message (**no-op**) and once using a local correction (**repair**).
3. Grade both final outcomes.

The correction uses the task, trace prefix, original message, and local error annotation. It receives no gold answer or future trace messages. Tool-feedback IFs are excluded because the editable boundary is an agent message. Corrected Python requests are executed by the proxy, and replacement messages must fit both the per-call and remaining trace budgets.

The output contains paired binary outcomes, success-preservation and failure-rescue rates averaged first over IF interventions within each trace and then over traces, and the repair-minus-no-op rescue difference with a 95% paired task-cluster bootstrap interval (5,000 resamples, unadjusted). Runtime seeds control scheduling and trace sampling; remote model generation remains stochastic.

**Python execution:** the proxy restores computation state by replaying successful deterministic programs in a fresh subprocess. Its timeout and temporary working directory are not a security sandbox. Use an isolated execution container for untrusted model code.

## Best-of-N selection

Selection generates 16 fresh candidate traces per task and compares selectors using the first 4, 8, and 16 candidates. It splits tasks into 20% tuning and 80% test with a fixed seed. Supply at least five tasks.

```bash
python -m masif.selection \
  --tasks tasks.jsonl \
  --architecture mathchat \
  --output outputs/selection.json
```

The judge sees only the prompt and visible trace steps. The IF selector chooses the candidate with the lowest risk:

```text
risk = sum(type_weight * absorption_weight) / max(1, parsed_trace_steps)
```

Weights follow Tables 7–8. Parsed steps include agent and proxy messages, with no extra task-header step. Risk is not clamped to [0, 1]. The output also reports the expected accuracy of choosing a candidate uniformly at random.

### Optional PRM comparisons

A process reward model (PRM) scores reasoning steps. To add the included local math PRM adapters, install the training dependencies and supply a compatible model path:

```bash
export PRM_KIND=qwen_math  # Or math_shepherd
export PRM_MODEL='<compatible math PRM model path>'
export PRM_DEVICE=cpu

python -m masif.selection \
  --tasks tasks.jsonl \
  --architecture mathchat \
  --prm masif.prm:score \
  --output outputs/selection_with_prm.json
```

Alternatively, use `--prm your_module:score`. The importable function must implement `score(prompt, steps) -> list[float]`; it receives no answer or outcome. Scores must lie in [0, 1]. For PRInTS raw scores in [-4, 4], also pass `--prm-scale prints`. PRInTS inference must be supplied through this interface. Its upstream implementation is not bundled.

The tuning set determines how step scores are pooled: minimum, mean, geometric mean, or last step. Geometric mean implements the paper's product/geometric-mean option. The combined PRM+IF selector separately tunes pooling and the weight in:

```text
combined_score = w * prm_score + (1 - w) * (1 - risk)
w = 0, 0.1, ..., 1
```

The settings stay fixed on the test set. Ties select the first candidate or setting.

## IF-GRPO training

Training uses a local shared policy. Each group contains eight complete multi-agent trajectories for the same task.

```bash
python -m masif.train \
  --tasks train.jsonl \
  --eval-tasks validation.jsonl \
  --model '<causal policy model path>' \
  --device cpu \
  --architecture mathchat \
  --objective if_grpo \
  --steps 200 \
  --output checkpoints/if_grpo
```

Choose a device suited to your model and available hardware. The CPU setting above is supported, but full-parameter training of a large model requires substantial memory and compute. The model's tokenizer must have a compatible chat template. Separate thinking mode is disabled where supported.

For the matched outcome-only baseline, use `--objective grpo` with the same settings and a separate output directory. Both objectives require a judge when `--eval-tasks` is supplied. Held-out evaluation uses greedy decoding and never updates the policy.

The output directory contains the model and tokenizer. With `--eval-tasks`, it also contains `evaluation.json`, with per-task outcomes, IF costs/counts, generated-token counts, and these aggregate metrics:

- Accuracy.
- IF cost and count per task.
- IF cost and count per 1,000 generated tokens.

The package does not perform the paper's paired statistical tests or Holm correction.

### How IFs affect training

The outcome advantage is:

```text
outcome_advantage = (y - mean(y)) / (sample_std(y) + 1e-6)
```

For groups containing both successes and failures, the judge scores only successful traces. Their **unnormalized** weighted IF cost, C, adjusts the positive outcome advantage:

```text
z = (max_success_cost - C) / (max_success_cost - min_success_cost) - 0.5
successful_advantage = outcome_advantage * (1 + z)
failed_advantage = outcome_advantage
```

Set z to zero when fewer than two traces succeed or successful costs are equal. All-success and all-failure groups receive zero advantages and skip the optimizer update, including weight decay.

### Training settings

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | Constant 1e-6 |
| Betas | (0.9, 0.98) |
| Weight decay | 0.1 |
| PPO ratio clipping | [0.8, 1.28] |
| Reference-policy KL coefficient | 0 |
| Entropy regularization coefficient | 0 |
| Training decoding | Temperature 0.7; top-p 0.95 |
| Held-out decoding | Greedy |

Loss applies only to agent-generated tokens, averaged within each trajectory and then across the eight trajectories. Prompt and tool tokens receive no direct loss. All agent calls in a trajectory share its advantage. `--ppo-epochs` controls optimization passes over each collected group. Truncated final responses and unfinished MathChat computations count as failures.

IF detection and absorption labeling are separate judge calls at temperature zero. They use the eight error types and six absorption patterns in Appendix D.
