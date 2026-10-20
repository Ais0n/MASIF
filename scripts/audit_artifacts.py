"""Audit distribution files for credentials and operational metadata.

Findings contain file names, categories, and field locations, never matched
values. Public benchmark questions, retrieved sources, and model names remain
part of the dataset; deployment metadata is excluded.
"""
import argparse
import ast
from collections import Counter
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'credential_token': re.compile(
        r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{25,}'
        r'|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}'
        r'|AIza[A-Za-z0-9_-]{35}|hf_[A-Za-z0-9]{30,}'
        r'|xox[baprs]-[A-Za-z0-9-]{20,})\b'),
    'credential_assignment': re.compile(
        r'''(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret|user[_-]?key)'''
        r'''[\s"':=]{1,12}(?:Bearer\s+)?[A-Za-z0-9_./+\-]{20,}'''),
    'bearer_credential': re.compile(r'(?i)\bBearer\s+[A-Za-z0-9_./+\-]{20,}'),
    'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'),
    'header_credential': re.compile(
        r'''(?i)(?:set-cookie|cookie|x-api-key|x-user-key|x-auth-token|proxy-authorization|subscription-key)'''
        r'''[\s"':=]{1,12}[A-Za-z0-9_./+=\-]{24,}'''),
    'deployment_model_route': re.compile(
        r'\b[a-z]{1,8}/(?:gemini|gpt|claude|qwen|deepseek)[A-Za-z0-9_.-]+'),
    'local_path': re.compile(r'/(?:Users|home|root|private/tmp|tmp|workspace)/[A-Za-z0-9_.@/-]+'),
    'windows_path': re.compile(r'\b[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\s"<>]+'),
    'root_placeholder': re.compile(r'<[A-Z][A-Z0-9_]*ROOT>'),
    'response_identifier': re.compile(r'\b(?:chatcmpl|req|resp|cmpl)[-_][A-Za-z0-9_-]{16,}'),
}
OPERATIONAL_FIELDS = {
    'provider', 'output_dir', 'api_key', 'api_base', 'base_url', 'api_endpoint',
    'endpoint_url', 'access_token', 'user_key', 'authorization', 'password',
    'secret_key', 'proxy_mode',
    'request_id', 'trace_id',
}
SENSITIVE_QUERY_KEYS = {
    'api_key', 'apikey', 'access_token', 'auth_token', 'authorization', 'password',
    'secret', 'session_token', 'signature', 'x-amz-signature', 'x-amz-credential',
}
URL_RE = re.compile(r'https?://[^\s<>"\x27)\\]+')
LABELLED_IDENTIFIER_RE = re.compile(
    r'''(?i)\b(?:(?:request|trace|correlation|transaction|session)[ _-]?id|x-request-id|cf-ray)'''
    r'''[ \t"':=]+(?P<value>[A-Za-z0-9][A-Za-z0-9_+/=:.\-]{7,})''')


def runtime_identifiers(text):
    for match in LABELLED_IDENTIFIER_RE.finditer(text):
        value = match['value']
        if len(value) >= 16 and (
                any(char.isdigit() for char in value)
                or (len(value) >= 24 and any(char.isupper() for char in value)
                    and any(char.islower() for char in value))):
            yield match


def embedded_object(text, error_context=False):
    candidate = text.strip()
    if error_context and not candidate.startswith(('{', '[')):
        start = candidate.find('{')
        if start == -1:
            return None
        candidate = candidate[start:]
    if not candidate.startswith(('{', '[')):
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except (ValueError, RecursionError):
        if not error_context:
            return None
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, RecursionError):
            return None
    return value if isinstance(value, (dict, list)) else None


def distribution_files(root):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or path.name == '.DS_Store' or path.suffix == '.pyc':
            continue
        if any(part.startswith('.') or part.endswith('.egg-info')
               or part in ('__pycache__', 'outputs', 'checkpoints')
               for part in relative.parts[:-1]):
            continue
        yield path


def audit(root=ROOT):
    findings = []
    total = 0

    def finding(file, category, field, count=1):
        findings.append(dict(file=file, category=category, field=field, count=count))

    def walk(value, parts, file, embedded_depth=0):
        if isinstance(value, dict):
            response = 'raw' in parts or value.get('object') in ('chat.completion', 'chat.completion.chunk')
            for key, item in value.items():
                location = '.'.join(parts + [key])
                redacted = isinstance(item, str) and re.fullmatch(r'\[REDACTED_[A-Z_]+\]', item)
                if key.casefold() in OPERATIONAL_FIELDS and not redacted:
                    finding(file, 'operational_metadata', location)
                if key in ('created', 'id', 'system_fingerprint') and response and not redacted:
                    finding(file, 'response_metadata', location)
                if key == 'agent' and parts and parts[-1] == 'metadata':
                    finding(file, 'implementation_metadata', location)
                if key == 'model' and isinstance(item, str) and '/' in item:
                    finding(file, 'model_route', location)
                if key == 'source_ref' and isinstance(item, str):
                    if not re.fullmatch(r'annotation-batch-[0-9]{2}', item):
                        finding(file, 'internal_provenance_path', location)
                walk(item, parts + [key], file, embedded_depth)
        elif isinstance(value, list):
            for item in value:
                walk(item, parts + ['[]'], file, embedded_depth)
        elif isinstance(value, str):
            count = sum(1 for _ in runtime_identifiers(value))
            if count:
                finding(file, 'embedded_runtime_identifier', '.'.join(parts), count)
            if embedded_depth < 8:
                parsed = embedded_object(value, any(p in ('error', 'run_error', 'reason') for p in parts))
                if parsed is not None:
                    walk(parsed, parts + ['$embedded'], file, embedded_depth + 1)

    for path in distribution_files(root):
        total += 1
        name = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            finding(name, 'unexpected_binary', '$')
            continue
        for category, pattern in TEXT_PATTERNS.items():
            count = len(pattern.findall(text))
            if count:
                finding(name, category, '$text', count)
        identifiers = sum(1 for _ in runtime_identifiers(text))
        if identifiers:
            finding(name, 'embedded_runtime_identifier', '$text', identifiers)
        for url in URL_RE.findall(text):
            try:
                parsed = urlsplit(url)
                host = parsed.hostname or ''
                if parsed.username is not None or parsed.password is not None:
                    finding(name, 'url_credentials', '$text')
                if any(key.casefold() in SENSITIVE_QUERY_KEYS for key, _ in parse_qsl(parsed.query)):
                    finding(name, 'signed_or_credential_url', '$text')
                try:
                    address = ipaddress.ip_address(host)
                    if address.is_private or address.is_loopback or address.is_link_local:
                        finding(name, 'private_network_address', '$text')
                except ValueError:
                    if host == 'localhost' or host.endswith(('.local', '.internal')):
                        finding(name, 'private_network_address', '$text')
            except ValueError:
                # Retrieved prose can contain incomplete or punctuation-joined
                # URLs. These are not operational endpoints by themselves.
                continue
        try:
            if path.suffix == '.json':
                walk(json.loads(text), [], name)
            elif path.suffix == '.jsonl':
                for number, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        walk(json.loads(line), [f'line[{number}]'], name)
        except json.JSONDecodeError:
            finding(name, 'invalid_json', '$')
    counts = Counter()
    for item in findings:
        counts[item['category']] += item['count']
    return dict(files_scanned=total, counts=dict(sorted(counts.items())), findings=findings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Save findings without matched values')
    args = parser.parse_args()
    report = audit(args.root)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'findings'}, indent=2))
    raise SystemExit(bool(report['findings']))
