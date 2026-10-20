"""Optional local math PRM adapters. Configure PRM_KIND and PRM_MODEL externally."""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def runtime():
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    kind = os.environ['PRM_KIND']
    model_path = os.environ['PRM_MODEL']
    if kind not in ('qwen_math', 'math_shepherd'):
        raise ValueError('PRM_KIND must be qwen_math or math_shepherd')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=kind == 'qwen_math')
    cls = AutoModel if kind == 'qwen_math' else AutoModelForCausalLM
    model = cls.from_pretrained(model_path, trust_remote_code=kind == 'qwen_math').to(
        os.environ.get('PRM_DEVICE', 'cpu')).eval()
    return torch, tokenizer, model, kind


def score(prompt, steps):
    """Score agent reasoning messages, excluding proxy tool feedback."""
    torch, tokenizer, model, kind = runtime()
    reasoning = [s['content'] for s in steps if s['agent'] != 'proxy']
    if not reasoning:
        raise ValueError('No reasoning steps available')
    if kind == 'qwen_math':
        tag = '<extra_0>'
        chat = [{'role': 'system', 'content': 'Please reason step by step.'},
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': tag.join(reasoning) + tag}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        tag_ids = tokenizer.encode(tag, add_special_tokens=False)
        if len(tag_ids) != 1:
            raise ValueError('PRM step separator must be a single token')
        tag_id = tag_ids[0]
    else:
        tag = 'ки'
        text = prompt + '\n' + '\n'.join(f'Step {i}: {s} {tag}' for i, s in enumerate(reasoning, 1))
        tag_ids = tokenizer.encode(tag, add_special_tokens=False)
        tag_id = tag_ids[-1]
        labels = tokenizer.encode('+ -', add_special_tokens=False)
        if len(labels) != 2 or len(set(labels)) != 2:
            raise ValueError('PRM requires distinct single-token + and - labels')
    ids = tokenizer.encode(text, return_tensors='pt').to(next(model.parameters()).device)
    limit = getattr(model.config, 'max_position_embeddings', None)
    if limit is not None and ids.shape[1] > limit:
        raise ValueError('PRM context exhausted; no silent step truncation')
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=False)
        logits = output.logits if hasattr(output, 'logits') else output[0]
        positions = ids[0] == tag_id
        if kind == 'qwen_math':
            values = logits[0, positions].float().softmax(-1)[:, 1]
        else:
            values = logits[0, positions][:, labels].float().softmax(-1)[:, 0]
    if len(values) != len(reasoning):
        raise ValueError('PRM step markers do not match reasoning steps')
    return values.cpu().tolist()
