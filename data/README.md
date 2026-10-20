# Trace dataset

Execution traces and implicit failure (IF) annotations for WebWalkerQA and
BrowseComp-Plus, corresponding to the two web-agent configurations in
Sections 2.1–2.3. MAST-derived traces and annotations are not included.

| Configuration | Backbone | Traces | Success | Failure | Retained IFs |
| --- | --- | ---: | ---: | ---: | ---: |
| WebAgent (WebWalker) / WebWalkerQA | Qwen3-8B | 680 | 71 | 609 | 1,090 |
| MrlX / BrowseComp-Plus | Qwen3-8B | 100 | 4 | 96 | 119 |
| Total | | 780 | 75 | 705 | 1,209 |

## Files and joins

- `q1_annotations/traces.jsonl`: one record per trace, with the stored outcome,
  task information, raw/console paths, and its nested retained IF annotations.
- `q1_annotations/if_records.jsonl`: the same retained annotations flattened to
  one record per IF. Join on `(configuration, task_id)`; `if_id` is unique.
- `q1_annotations/cohort.json`: the same 780 trace summaries without nested IFs.
- `q1_annotations/statistics.json`: per-configuration trace/outcome/IF counts.
- `q1_annotations/excluded_records.jsonl`: one access-restriction event excluded
  from the retained IF set. Do not add it to the counts above.
- `raw_traces/webwalkerqa/`: 680 task directories,
  each containing `trace.json`, `console_log.txt`, and `result.json`.
- `raw_traces/browsecomp_plus/`: 100 raw JSON
  traces with `metadata`, `messages`, tool outputs, and final results.
- `traces/browsecomp_plus/`: console representations of the 100 MrlX raw
  message lists, referenced by the summaries' `console_path` fields.

Paths beginning with `data/` are relative to the repository root.
`source_ref` values are opaque annotation-batch identifiers. Records from the
same annotation batch share an identifier.

## Reading the records

`semantic_success` is the outcome used for the dataset statistics. WebWalker
`result.json` records contain the corresponding outcomes and task information.
`original_outcome_before_rejudge` stores an earlier assessment, where present;
the dataset statistics use the final stored outcome. Outcome judge metadata is
available for 75 successful traces; the 705 failed summaries have `judge: null`.

For MrlX, use `metadata.query` in the raw JSON as the complete task prompt.
Four successful trace summaries (`229`, `231`, `245`, and `820`) retain only the
first 300 characters in `prompt`. Their full questions remain in the raw files.

`error_step` is one-based: it indexes WebWalker `events` or MrlX `messages`.
The initial system/user messages count when present. For two MrlX annotations
in task `844`, `error_agent: browser` identifies tool feedback serialized with
chat role `user`.

The eight `error_type` values and six `absorption_pattern` values use the
paper's taxonomy. Six source `other` types were mapped to `malformed_output`;
the excluded event remains in `excluded_records.jsonl`. The original labels and
cleaning actions are retained. `source_absorption_pattern` may retain the older
`trivial_issue_or_unknown` spelling; the canonical field uses `trivial_issue`.

The annotation records include error descriptions and absorption rationales.
The `severity`, `confidence`, and `error_evidence` fields in the live judge
response schema are not included in this dataset export.

## Validation and statistics

From the repository root, with Python 3.10 or later:

```bash
python scripts/check_artifacts.py
```

This needs no credentials or third-party packages. It verifies file hashes,
joins, raw-trace coverage, step indices, normalized console files, and dataset
counts, and reports prevalence/intensity and taxonomy counts for this subset.
Per-configuration counts are provided in `q1_annotations/statistics.json`.

The full-corpus prevalence and Tables 2–3 require the omitted MAST-derived
records. Replay outcomes, best-of-N candidates, RL checkpoints, human audit
records, and cross-judge audit records are not included. The released web
traces are not directly accepted by the code's task-JSONL or runtime-snapshot
interfaces; the code README describes the separate workflows.

The repository's MIT license covers its code. Benchmark content and retrieved
web/document text retain their respective upstream rights; this release does
not assign a new license to third-party content.
