"""Two-stage, trace-only IF annotation and Appendix A severity weights."""
import json
import time
from .common import Endpoint
from .prompts import ABSORB_SYSTEM, ABSORB_USER, DETECT_SYSTEM, DETECT_USER, REPAIR_SYSTEM, REPAIR_USER, fill

TYPE_WEIGHTS = dict(hallucination=1.30, reasoning_error=1.20, wrong_action=1.10,
                    misunderstanding=1.10, failed_tool_use=1.00, malformed_output=1.00,
                    incomplete_action=0.90, unnecessary_action=0.60)
ABS_WEIGHTS = dict(trivial_issue=0.35, self_corrected=0.55, self_workaround=0.65,
                   peer_repair=0.75, orchestrator_intervention=0.80, unresolved_propagation=1.40)


def format_trace(steps):
    return '\n\n'.join(f"[Step {i}] Agent: {s['agent']}\n{s['content']}"
                        for i, s in enumerate(steps, 1))


def cost(errors):
    return sum(TYPE_WEIGHTS[e['error_type']] * ABS_WEIGHTS[e['absorption_pattern']] for e in errors)


def validate_errors(data, steps):
    errors = data['implicit_failures']
    if not isinstance(errors, list):
        raise ValueError('Expected an implicit_failures list')
    for error in errors:
        index = error['error_step']
        if type(index) is not int or not 1 <= index <= len(steps):
            raise ValueError('Invalid IF step')
        if error['error_agent'] != steps[index - 1]['agent'] or error['error_type'] not in TYPE_WEIGHTS:
            raise ValueError('Invalid IF agent or error type')
        if any(not isinstance(error[key], str) or not error[key].strip()
               for key in ('error_description', 'error_evidence')):
            raise ValueError('Missing IF description or evidence')
        if error['severity'] not in ('low', 'medium', 'high') or error['confidence'] not in ('low', 'medium', 'high'):
            raise ValueError('Invalid IF severity or confidence')
    return errors


def validate_absorption(data):
    if data['absorption_pattern'] not in ABS_WEIGHTS or data['confidence'] not in ('low', 'medium', 'high'):
        raise ValueError('Invalid absorption label or confidence')
    if not isinstance(data['rationale'], str) or not data['rationale'].strip():
        raise ValueError('Missing absorption rationale')
    return {key: data[key] for key in ('absorption_pattern', 'confidence', 'rationale')}


class Judge:
    def __init__(self):
        self.endpoint = Endpoint('JUDGE')

    def ask(self, system, user, max_tokens=4096, allow_truncated=False):
        text, self.last_tokens = self.endpoint.complete(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            max_tokens=max_tokens, allow_truncated=allow_truncated)
        self.last_truncated = getattr(self.endpoint, 'last_truncated', False)
        return text

    def ask_json(self, system, user, validator):
        for attempt in range(3):
            text = self.ask(system, user)
            try:
                return validator(json.loads(text))
            except (ValueError, KeyError, TypeError):
                if attempt == 2:
                    raise ValueError('Judge returned invalid JSON or annotation schema after 3 attempts') from None
                time.sleep(2 ** attempt)

    def annotate(self, prompt, steps):
        # Only these explicit fields enter the judge, never task answers or rollout metadata.
        trace = format_trace(steps)
        user = fill(DETECT_USER, TASK_PROMPT=prompt, EXECUTION_TRACE=trace)
        errors = self.ask_json(DETECT_SYSTEM, user, lambda data: validate_errors(data, steps))
        for error in errors:
            user = fill(ABSORB_USER, TASK_PROMPT=prompt, ERROR_ANNOTATION=json.dumps(error), EXECUTION_TRACE=trace)
            annotation = self.ask_json(ABSORB_SYSTEM, user, validate_absorption)
            error['absorption_pattern'] = annotation['absorption_pattern']
            error['absorption_confidence'] = annotation['confidence']
            error['absorption_rationale'] = annotation['rationale']
        return errors

    def repair(self, prompt, prefix, step, error, max_tokens=2048):
        # Appendix D, Listings 5-6: no gold answer and no later trace steps.
        user = fill(REPAIR_USER, QUESTION=prompt, PREFIX=format_trace(prefix),
                    STEP_INDEX=error['error_step'], AGENT=step['agent'], BUGGY_MESSAGE=step['content'],
                    ERROR_TYPE=error['error_type'], ERROR_DESCRIPTION=error['error_description'],
                    SEVERITY=error['severity'])
        return self.ask(REPAIR_SYSTEM, user, max_tokens=max_tokens, allow_truncated=True)
