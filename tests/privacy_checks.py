"""Offline checks for the distribution metadata audit."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_artifacts import audit


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    sample = root / 'records.json'
    synthetic_key = 'sk-' + 'Z' * 32
    unsafe = {
        'records': [{'judge': {'model': 'deployment/' + 'model-name',
                               'provider': 'synthetic-provider'},
                     'source_ref': 'private-run/' + 'annotation.json',
                     'meta': {'raw': {'id': 'chatcmpl-' + 'a' * 24,
                                      'created': 1,
                                      'system_fingerprint': None}},
                     'output_dir': 'private-output',
                     'metadata': {'agent': 'private-adapter'}}],
        'credential': synthetic_key,
        'deployment_route': 'gateway/' + 'gemini-example',
        'url': 'https://' + 'operator:example-password' + '@service.invalid/path',
        'signed': 'https://' + 'service.invalid/path?access_token=' + 'Z' * 24,
        'local_url': 'http://' + '127.0.0.1:8080',
    }
    sample.write_text(json.dumps(unsafe))
    report = audit(root)
    required = {'model_route', 'operational_metadata', 'internal_provenance_path',
                'response_identifier', 'response_metadata', 'implementation_metadata',
                'credential_token', 'url_credentials', 'signed_or_credential_url',
                'private_network_address', 'deployment_model_route'}
    assert required <= set(report['counts']), report['counts']
    assert synthetic_key not in json.dumps(report)
    assert 'synthetic-provider' not in json.dumps(report)
    synthetic_request = 'a' * 8 + '-' + '1' * 4 + '-' + '2' * 4 + '-' + '3' * 4 + '-' + 'b' * 12
    encoded = {'error': 'HTTP error: ' + json.dumps({'request_id': synthetic_request}),
               'run_error': 'HTTP error: ' + repr({'provider': 'synthetic-provider'}),
               'content': 'Request ID: ' + 'A1b2C3d4' * 7 + '=='}
    sample.write_text(json.dumps(encoded))
    embedded = audit(root)
    assert {'embedded_runtime_identifier', 'operational_metadata'} <= set(embedded['counts'])
    assert synthetic_request not in json.dumps(embedded)
    assert 'synthetic-provider' not in json.dumps(embedded)
    safe = {'records': [{'judge': {'model': 'model-name'},
                         'source_ref': 'annotation-batch-01',
                         'semantic_success': True,
                         'content': 'Public benchmark question and retrieved evidence.',
                         'error': 'HTTP error: ' + json.dumps({'request_id': '[REDACTED_REQUEST_ID]'})}]}
    sample.write_text(json.dumps(safe))
    assert audit(root)['findings'] == []
print('PASS nested metadata, model routes, credentials, transport identifiers, URL checks, and safe records')
