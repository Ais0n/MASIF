"""Appendix D, Listings 1-6, with PDF line wrapping and typography normalized."""
import re

DETECT_SYSTEM = '''You are an expert analyzer of LLM-based multi-agent system execution traces.

Identify implicit failures using only the task prompt and execution trace. An implicit failure is a concrete, trace-observable process error that does not individually determine the final outcome of the trace. Determine whether the event meets this definition, but do not classify its absorption pattern.

Return only valid JSON matching the requested schema, with no markdown or extra text.'''

DETECT_USER = '''Analyze this multi-agent execution trace for concrete local process errors.

## Task
{{TASK_PROMPT}}

## Execution Trace
{{EXECUTION_TRACE}}

## Instructions

Identify all clear local implicit errors visible in the trace. An implicit error is a specific mistake, failed action, unsupported claim, malformed tool use, invalid inference, misunderstanding, omission, hallucination, or materially unnecessary action at a particular trace step, which does NOT individually determine the final outcome of the trace.

For each implicit error, return:
- error_step: the step index at which the error occurs
- error_agent: the agent responsible for the error
- error_type: one of failed_tool_use, wrong_action, unnecessary_action, hallucination,
  incomplete_action, reasoning_error, misunderstanding, malformed_output
- error_description: a concise description of what is wrong, grounded in trace evidence
- error_evidence: a short excerpt or close paraphrase of the trace evidence
- severity: low, medium, or high, based only on the local process error
- confidence: high, medium, or low

Detailed explanation of the error type taxonomy:

| Type | Description | Example |
| --- | --- | --- |
| Failed Tool Use | A tool call fails, is blocked, times out, cannot execute, or returns unusable output because of tool or environment issues. | FileSurfer attempts to read an MP3 audio file as text and receives a conversion exception. |
| Wrong Action | The agent takes an action that is inappropriate for the current task state or unlikely to make progress. | WebSurfer clicks page text rather than a valid link or play button, leaving the browser on the same page. |
| Unnecessary Action | The agent performs redundant, low-value, or repetitive actions that consume steps without advancing the task. | WebSurfer handles a cookie prompt instead of following the already identified target page. |
| Hallucination | The agent fabricates information, tool behavior, source content, or intermediate evidence not supported by the trace. | The agent invents a ranked list item that is not present in the retrieved source. |
| Incomplete Action | The agent attempts an action but omits required details, formatting, inputs, or follow-through. | FileSurfer repeatedly returns the first PDF page instead of searching or navigating to the relevant pages. |
| Reasoning Error | The agent makes an incorrect inference, introduces or propagates incorrect information, repeats a known-bad strategy, misses relevant evidence, or terminates from insufficient reasoning. | The assistant swaps document statuses across several listed books, or the orchestrator identifies the wrong single from an album despite available evidence. |
| Misunderstanding | The agent misunderstands the task, role, constraints, or operational setting. | The orchestrator fails to recognize that another agent has already reached the correct page and restarts the search. |
| Malformed Output | The agent emits syntactically malformed text or a corrupted structured block, but execution continues. | WebWalker emits a memory block with a stray closing quote and brace before recovering in the next step. |

Return this JSON schema:
{
  "implicit_failures": [
    {
      "error_step": 5,
      "error_agent": "AgentName",
      "error_type": "reasoning_error",
      "error_description": "The agent reports an unsupported and internally inconsistent count.",
      "error_evidence": "The relevant trace evidence.",
      "severity": "medium",
      "confidence": "high"
    }
  ],
  "summary": {
    "total_implicit_failures": 1,
    "overall_process_assessment": "Brief trace-only assessment."
  }
}

If no clear local errors are visible, return an empty implicit_failures list.'''

ABSORB_SYSTEM = '''You are an expert analyzer of how multi-agent systems handle local process errors.

Classify the absorption pattern using only the task prompt, the given local-error annotation, and execution-trace evidence. Do not use or infer the expected answer, final-answer correctness, trace outcome, success/failure split, or decisive-error annotation. Do not assume that a correct final answer makes an error trivial.

Return only valid JSON matching the requested schema, with no markdown or extra text.'''

ABSORB_USER = '''Classify how the execution trace handles the annotated local error.

## Task
{{TASK_PROMPT}}

## Local error annotation
{{ERROR_ANNOTATION}}

## Trace context
{{EXECUTION_TRACE}}

## Absorption labels

- self_corrected: The same agent recognizes and repairs its own local error.
- self_workaround: The originating agent bypasses the local error through a genuinely
  different route, query, tool, source, or implementation strategy.
- peer_repair: A different non-managerial agent corrects the error or supplies a
  replacement result that is used downstream.
- orchestrator_intervention: A manager, planner, or orchestrator recognizes the
  problem and redirects or corrects the trajectory.
- unresolved_propagation: The local error is not repaired before its erroneous artifact
  is accepted or used by at least one downstream step as evidence, a premise, memory,
  goal, calculation input, action/tool input, patch input, or justification.
- trivial_issue: A catch-all for absorption patterns with low analytic value, including
  mechanical retries that do not change strategy, duplicate work, ignored or abandoned
  actions, tool- or environment-level absorption, runtime blocking, timeout, and trace
  termination. Also use it when the available trace evidence is insufficient to
  distinguish a trivial issue from another absorption pattern.

## Clarification
- To distinguish between self_corrected, self_workaround, and trivial_issue, the decisive dimension is how the originating agent proceeds. Apply self_corrected when the same agent repairs the faulty output while keeping the same core tool, query, implementation logic, or data source--for example, by adjusting meaningful input parameters or recalculating within the same reasoning chain. Apply self_workaround only when the agent abandons the original failing method and instead pursues a genuinely different route--such as switching to an alternative tool, querying a different data source, or adopting a different implementation path. An exact or merely mechanical retry, including a superficial formatting-only retry, belongs to trivial_issue.
- To distinguish between peer_repair and orchestrator_intervention, the decisive dimension is the hierarchical role of the corrective agent and the scope of the correction. Apply peer_repair when a non-managerial agent (i.e., at the same or equivalent execution level as the original agent) provides a corrected output or replacement result, and this correction is integrated locally without triggering a global re-plan or trajectory redirection. Apply orchestrator_intervention only when a manager, planner, or orchestrator explicitly detects the error and imposes a high-level change--such as issuing a revised plan, reassigning the subtask to a different agent, aborting and restarting the workflow with new instructions, or actively overriding the erroneous step as part of a systemic trajectory adjustment. In practice, look for evidence of top-down command or plan modification for orchestrator_intervention; if the fix comes from a same-level collaborator without altering the overall task topology, use peer_repair.
- To distinguish between unresolved_propagation and trivial_issue, the decisive factor is whether the erroneous artifact--the faulty output, incorrect calculation, or flawed premise--is demonstrably consumed by at least one downstream step. Apply unresolved_propagation only when the trace provides clear evidence that the artifact is explicitly used as an input, evidence, premise, memory update, or justification in a subsequent action, regardless of whether the overall task ultimately succeeds. Conversely, apply trivial_issue if the issue is contained locally by a mechanical retry, ignored, or occurs in a terminated or abandoned branch and does not demonstrably feed into later processing.

Choose exactly one label. Return:
{
  "absorption_pattern": "self_corrected|self_workaround|peer_repair|orchestrator_intervention|unresolved_propagation|trivial_issue",
  "confidence": "high|medium|low",
  "rationale": "One concise sentence grounded in the downstream trace evidence."
}'''

REPAIR_SYSTEM = '''You are an expert in LLM-based Multi-Agent Systems. Given a partial execution trace where a specific agent step was wrong, you produce a corrected replacement of that single step in the agent's voice. You must:

- Preserve the agent's persona, tool-call format, and writing style as evident from the trace.
- Fix only the identified error; do not anticipate or pre-empt later steps.
- Output ONLY the corrected message body. No commentary, no JSON wrapping, no leading "FixedMessage:" labels. Just the literal text the agent should have produced.'''

REPAIR_USER = '''## Task being solved by the multi-agent system
{{QUESTION}}

## Conversation history up to (but NOT including) the buggy step

{{PREFIX}}

## Buggy step (step {{STEP_INDEX}}, agent: {{AGENT}}) -- this is what the agent actually said and what we want to replace
<<<BUGGY MESSAGE BEGIN>>>
{{BUGGY_MESSAGE}}
<<<BUGGY MESSAGE END>>>

## What was wrong with this step (annotation from an analyzer)
- error_type: {{ERROR_TYPE}}
- description: {{ERROR_DESCRIPTION}}
- severity: {{SEVERITY}}

## Your job
Produce a single corrected version of the buggy step that the {{AGENT}} agent should have output instead. Stay in the same persona and format. Do not include any text other than the message body itself.'''


def fill(template, **values):
    # Replace placeholders only in the template, never inside user-provided values.
    return re.sub(r'\{\{([A-Z_]+)\}\}', lambda m: str(values[m[1]]), template)
