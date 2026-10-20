"""Offline audit: simulated chat endpoints, no model credentials needed."""
import json, os, re, sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masif import replay, selection
from masif.runtime import initial_state, run, execute
from masif.train import advantages
for prefix in ('POLICY', 'JUDGE'):
    os.environ[prefix + '_URL'] = 'https://mock.invalid/' + prefix
    os.environ[prefix + '_MODEL'] = 'mock'
calls = []
class Response:
    status_code = 200
    def __init__(self, content): self.content = content
    def json(self):
        return {'choices': [{'message': {'content': self.content}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10}}
def post(url, headers, json, timeout):
    calls.append(json)
    system = json['messages'][0]['content']
    user = json['messages'][1]['content']
    if url.endswith('POLICY'):
        content = 'FINAL: ' + ('4' if 'WRONG' in user else '5')
    elif 'Given a partial execution' in system:
        content = 'FINAL: 5'
    elif 'You are an expert analyzer of how' in system:
        content = __import__('json').dumps({'absorption_pattern': 'trivial_issue', 'confidence': 'high', 'rationale': 'mock'})
    else:
        agent = re.search(r'\[Step 1\] Agent: (\S+)', user).group(1)
        content = __import__('json').dumps({'implicit_failures': [{'error_step': 1, 'error_agent': agent, 'error_type': 'reasoning_error', 'error_description': 'mock error', 'error_evidence': 'mock', 'severity': 'low', 'confidence': 'high'}]})
    return Response(content)
with tempfile.TemporaryDirectory() as tmp, patch('requests.post', post):
    tasks = Path(tmp) / 'tasks.jsonl'
    tasks.write_text('\n'.join(json.dumps({'id': str(i), 'prompt': ('WRONG' if i % 2 else 'RIGHT') + ' What is 2+3?', 'answer': '5'}) for i in range(10)))
    for arch in ('mathchat', 'teamwork', 'self_moa'):
        for module in (selection, replay):
            output = Path(tmp) / f'{arch}-{module.__name__}.json'
            sys.argv = ['audit', '--tasks', str(tasks), '--architecture', arch, '--output', str(output)]
            if module is replay: sys.argv += ['--sample-per-outcome', '1']
            module.main()
            assert output.exists()
            print('PASS CLI', module.__name__, arch)
assert advantages([1]*8) == [0.0]*8
assert advantages([0]*8) == [0.0]*8
values = advantages([1,1,0,0,0,0,0,0], [0,2,None,None,None,None,None,None])
assert abs(values[0] / values[1] - 3) < 1e-12
print('PASS advantage edge cases')
assert execute('print(x+1)', ['x=4']) == ('5', True)
print('PASS computation state restoration')
state = initial_state('2+3?', 'mathchat', 42)
state['generated_tokens'] = 8190
def no_generation(*args):
    raise AssertionError('Should not generate')
try:
    run(no_generation, state=state, replacement='FINAL: 5', replacement_tokens=4096)
except ValueError:
    pass
else:
    raise AssertionError('Original budget defect persists')
result = run(no_generation, state=state, replacement='FINAL: 5', replacement_tokens=2)
assert result['generated_tokens'] == 8192 and result['final'] == 'FINAL: 5'
result = run(no_generation, state=state, replacement='FINAL: 5', replacement_tokens=2, replacement_truncated=True)
assert result['final'] == ''
for arch, count in [('mathchat', 2049), ('teamwork', 1025), ('self_moa', 1025)]:
    state = initial_state('2+3?', arch, 42)
    try:
        run(no_generation, state=state, replacement='FINAL: 5', replacement_tokens=count)
    except ValueError:
        pass
    else:
        raise AssertionError('Oversized replacement accepted')
print('PASS replacement budgets and truncated MathChat repair')
from masif.common import Endpoint, grade
from masif.judge import Judge
from masif.train import evaluate_heldout
class RateLimitedResponse(Response):
    status_code = 429
with patch('requests.post', side_effect=[RateLimitedResponse(''), Response('FINAL: 5')]) as transport, patch('time.sleep'):
    assert Endpoint('POLICY').complete([{'role': 'user', 'content': 'test'}])[0] == 'FINAL: 5'
    assert transport.call_count == 2
print('PASS transport retry')
with patch('requests.post', side_effect=[Response('invalid json'), Response('{"implicit_failures": []}')]) as transport, patch('time.sleep'):
    assert Judge().annotate('test', [{'agent':'assistant','content':'FINAL: 5'}]) == []
    assert transport.call_count == 2
print('PASS judge JSON retry')
class SimplePolicy:
    def __call__(self, messages, budget): return 'FINAL: 5', None, 10
class SimpleJudge:
    def annotate(self, prompt, steps):
        return [{'error_type':'reasoning_error','absorption_pattern':'trivial_issue'}]
report = evaluate_heldout(SimplePolicy(), SimpleJudge(), [{'id':'a','prompt':'2+3','answer':'5'}, {'id':'b','prompt':'2+3','answer':'4'}], grade, 'mathchat', 42)
assert report['accuracy'] == 0.5 and report['if_count_per_task'] == 1
assert abs(report['if_cost_per_task'] - 0.42) < 1e-12
assert abs(report['if_cost_per_1k_tokens'] - 42) < 1e-12
assert report['if_count_per_1k_tokens'] == 100
print('PASS held-out accuracy, IF metrics, and token densities')
