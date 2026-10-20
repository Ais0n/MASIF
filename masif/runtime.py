"""Small shared-policy MAS runtimes with restorable message and scheduler state."""
import copy
import random
import re
import subprocess
import sys
import tempfile

from .common import prediction

CONTRACT = 'Reason carefully. End your response with one final line: FINAL: <answer>. '
SYSTEM = {
    'mathchat': CONTRACT + '''You are the assistant in a conversation with a math proxy.
You may request computation in a fenced Python block. Use print for results. The proxy
returns stdout or errors. Use only deterministic computation; do not access files,
processes, or networks. Wait for the tool result before giving a final answer.''',
    'teamwork': CONTRACT + 'You are an expert collaborating with three other experts.',
    'self_moa': CONTRACT + 'You are an independent expert proposing a solution.',
}
AGGREGATOR = '''Synthesize the provided independent proposals into a single high-quality
response. Critically evaluate them, since some may be incorrect. Do not simply replicate
the proposals. Produce a coherent answer with careful reasoning. ''' + CONTRACT


def initial_state(prompt, architecture, seed):
    rng = random.Random(seed)
    if architecture == 'mathchat':
        plan = ['assistant'] * 15
    elif architecture == 'teamwork':
        agents = [f'expert_{i}' for i in range(4)]
        plan = agents.copy()
        for _ in range(4):
            order = agents.copy()
            rng.shuffle(order)
            plan.extend(order)
        plan.append(rng.choice(agents))
    elif architecture == 'self_moa':
        plan = [f'proposer_{i}' for i in range(6)] + ['aggregator']
    else:
        raise ValueError('Unknown architecture')
    return dict(prompt=prompt, architecture=architecture, plan=plan, cursor=0,
                histories={agent: [] for agent in set(plan)}, steps=[], proposals=[],
                programs=[], generated_tokens=0, done=False, final='')


def request(state):
    architecture = state['architecture']
    agent = state['plan'][state['cursor']]
    system = SYSTEM[architecture]
    if architecture == 'self_moa' and agent == 'aggregator':
        system = AGGREGATOR + '\nProposals:\n' + '\n\n'.join(state['proposals'])
    chat = [{'role': 'system', 'content': system}, {'role': 'user', 'content': state['prompt']}]
    chat.extend(state['histories'][agent])
    if architecture == 'teamwork':
        cursor = state['cursor']
        instruction = ('Answer independently.' if cursor < 4 else
                       'Give the final team answer.' if cursor == 20 else
                       f'Discuss with your peers. Discussion round {(cursor - 4) // 4 + 1} of 4.')
        chat.append({'role': 'user', 'content': instruction})
    return agent, chat


def execute(program, previous):
    # A separate process with a timeout, NOT a security sandbox. Run in an isolated
    # container when using untrusted policies. Only successful computation state is retained.
    script = ('import contextlib, io\nwith contextlib.redirect_stdout(io.StringIO()):\n'
              '    exec(' + repr('\n\n'.join(previous)) + ')\nexec(' + repr(program) + ')')
    try:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-c', script], cwd=directory,
                                    capture_output=True, text=True, timeout=10)
        return (result.stdout + result.stderr).strip() or '(no output)', result.returncode == 0
    except subprocess.TimeoutExpired:
        return 'Computation timed out.', False


def budget_for(state):
    per_call = 2048 if state['architecture'] == 'mathchat' else 1024
    total_limit = 8192 if state['architecture'] == 'mathchat' else len(state['plan']) * per_call
    used = state['generated_tokens']
    if type(used) is not int or not 0 <= used <= total_limit:
        raise ValueError('Invalid restored trace token count')
    return min(per_call, total_limit - used)


def consume(state, text, tokens, truncated=False):
    # Both fresh generations and injected replacements must pass this guard
    # before any history, tool execution, or final-answer acceptance is changed.
    if state['done'] or state['cursor'] >= len(state['plan']):
        raise ValueError('Cannot inject a step into a completed runtime')
    if type(tokens) is not int or not 0 < tokens <= budget_for(state):
        raise ValueError('Generated step exceeds its call or remaining trace token budget')
    agent = state['plan'][state['cursor']]
    architecture = state['architecture']
    state['steps'].append(dict(agent=agent, content=text, generated_tokens=tokens, truncated=truncated))
    state['histories'][agent].append({'role': 'assistant', 'content': text})
    state['generated_tokens'] += tokens
    state['final'] = text
    cursor = state['cursor']
    if truncated and architecture == 'mathchat':
        state['cursor'] += 1
        state['done'], state['final'] = True, ''
        return
    if architecture == 'mathchat':
        blocks = re.findall(r'```python\s*\n(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            program = '\n\n'.join(blocks)
            output, ok = execute(program, state['programs'])
            if ok:
                state['programs'].append(program)
            state['steps'].append(dict(agent='proxy', content=output))
            state['histories'][agent].append({'role': 'user', 'content': output})
        elif prediction(text) is not None:
            state['done'] = True
        else:
            state['histories'][agent].append({'role': 'user', 'content': 'Continue solving.'})
    elif architecture == 'self_moa':
        if agent != 'aggregator':
            state['proposals'].append(text)
    elif cursor == 3:
        broadcast = '\n\n'.join(f"{s['agent']}: {s['content']}" for s in state['steps'])
        for history in state['histories'].values():
            history.append({'role': 'user', 'content': 'Initial independent answers:\n' + broadcast})
    elif cursor >= 4:
        for other, history in state['histories'].items():
            if other != agent:
                history.append({'role': 'user', 'content': f'{agent}: {text}'})
    state['cursor'] += 1
    if state['cursor'] == len(state['plan']):
        state['done'] = True
        if truncated:
            state['final'] = ''


def run(policy, prompt=None, architecture='mathchat', seed=42, *, state=None,
        replacement=None, replacement_tokens=None, replacement_truncated=False, capture=False):
    state = copy.deepcopy(state) if state is not None else initial_state(prompt, architecture, seed)
    budget_for(state)  # Also reject invalid token counts in already-completed snapshots.
    segments, snapshots = [], {}
    if replacement is not None:
        if replacement_tokens is None:
            raise ValueError('Replacement token count required')
        consume(state, replacement, replacement_tokens, replacement_truncated)
    while not state['done']:
        budget = budget_for(state)
        if budget <= 0:
            state['final'] = ''  # Budget exhaustion is not a valid final answer.
            break
        if capture:
            snapshots[str(len(state['steps']) + 1)] = copy.deepcopy(state)
        _, chat = request(state)
        text, segment, count = policy(chat, budget)
        if segment is not None:
            segments.append(segment)
        consume(state, text, count, getattr(policy, 'last_truncated', False))
    if state['architecture'] == 'mathchat' and (not state['histories']['assistant'] or
                                              state['histories']['assistant'][-1]['role'] != 'assistant'):
        state['final'] = ''  # A pending computation or unfinished answer is a failure.
    return dict(steps=state['steps'], final=state['final'], generated_tokens=state['generated_tokens'],
                snapshots=snapshots, segments=segments)
