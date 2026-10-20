"""Task I/O, endpoint configuration, and outcome evaluation."""
import argparse
import importlib
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import requests


def read_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not tasks or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Provide nonempty JSONL with unique task ids')
    for task in tasks:
        if not isinstance(task['prompt'], str) or not task['prompt'].strip() or 'answer' not in task:
            raise ValueError('Each task requires id, prompt, and answer')
    return tasks


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_hook(spec):
    module, name = spec.split(':', 1)
    return getattr(importlib.import_module(module), name)


def prediction(text):
    lines = text.rstrip().splitlines()
    if not lines:
        return None
    match = re.fullmatch(r'FINAL:[ \t]*(\S[^\r\n]*)', lines[-1])
    return match[1].strip() if match else None


def grade(task, text):
    """Exact option/text matching, plus exact rational-number equivalence."""
    value = prediction(text)
    if value is None:
        return 0
    gold = str(task['answer']).strip()
    try:
        return int(Fraction(value.replace(',', '')) == Fraction(gold.replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return int(value.casefold() == gold.casefold())


class Endpoint:
    """A user-configured chat-completions endpoint; no embedded service addresses."""
    def __init__(self, prefix):
        missing = [prefix + suffix for suffix in ('_URL', '_MODEL') if not os.environ.get(prefix + suffix)]
        if missing:
            raise ValueError('Set endpoint configuration: ' + ', '.join(missing))
        self.url = os.environ[prefix + '_URL']
        self.model = os.environ[prefix + '_MODEL']
        self.key = os.environ.get(prefix + '_KEY', '')

    def complete(self, messages, temperature=0.0, max_tokens=4096, top_p=0.95, allow_truncated=False):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('Positive integer generation budget required')
        for attempt in range(3):
            try:
                response = requests.post(
                    self.url, headers={'Authorization': 'Bearer ' + self.key},
                    json={'model': self.model, 'messages': messages, 'temperature': temperature,
                          'top_p': top_p, 'max_tokens': max_tokens}, timeout=300)
            except requests.RequestException:
                response = None
            # Do not include remote response bodies, URLs, or headers in exceptions.
            if response is None or response.status_code in (408, 429) or 500 <= response.status_code < 600:
                if attempt == 2:
                    raise RuntimeError('Chat transport/server failure after 3 attempts') from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                raise RuntimeError(f'Chat request failed (HTTP {response.status_code})')
            try:
                data = response.json()
                choice = data['choices'][0]
                content = choice['message']['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Empty model response')
                self.last_truncated = choice.get('finish_reason') == 'length'
                accepted = (None, 'stop', 'length') if allow_truncated else (None, 'stop')
                if choice.get('finish_reason') not in accepted:
                    raise RuntimeError('Incomplete model response; increase the generation budget')
                count = data.get('usage', {}).get('completion_tokens')
                if count is not None and (type(count) is not int or not 0 < count <= max_tokens):
                    raise RuntimeError('Endpoint reported tokens outside the requested generation budget')
                return content, count
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                if attempt == 2:
                    raise RuntimeError('Invalid chat response after 3 attempts') from None
                time.sleep(2 ** attempt)


class RemotePolicy:
    def __init__(self, temperature=0.7):
        self.endpoint = Endpoint('POLICY')
        self.temperature = temperature

    def __call__(self, messages, budget):
        text, count = self.endpoint.complete(messages, self.temperature, budget, allow_truncated=True)
        self.last_truncated = self.endpoint.last_truncated
        if count is None:
            raise RuntimeError('Policy endpoint must report completion_tokens for trace budgets')
        return text, None, count


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--tasks', required=True, help='Task JSONL: id, prompt, answer')
    parser.add_argument('--architecture', choices=['mathchat', 'teamwork', 'self_moa'], default='mathchat')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grader', help='Optional module:function(task, final_text) -> 0 or 1')
    return parser


def evaluate(grader, task, text):
    value = grader(task, text)
    if value not in (0, 1):
        raise ValueError('Grader must return a binary outcome')
    return int(value)
