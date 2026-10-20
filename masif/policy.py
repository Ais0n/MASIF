"""Local shared causal policy and exact generated-token PPO segments."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def token_logprobs(model, ids, prompt_length):
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    logits = model(input_ids=inputs, use_cache=False).logits[0, prompt_length - 1:-1].float()
    targets = inputs[0, prompt_length:]
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class LocalPolicy:
    def __init__(self, model_path, device, temperature=0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float32 if device == 'cpu' else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        self.model.eval()  # Disable dropout for rollout/update likelihood consistency.
        self.temperature = temperature
        self.record_segments = True

    def __call__(self, messages, budget):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if limit is not None:
            budget = min(budget, limit - len(ids))
        if budget <= 0:
            raise ValueError('Policy context exhausted')
        inputs = torch.tensor([ids], device=next(self.model.parameters()).device)
        kwargs = dict(max_new_tokens=budget, do_sample=self.temperature > 0,
                      pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                      else self.tokenizer.eos_token_id)
        if self.temperature > 0:
            kwargs.update(temperature=self.temperature, top_p=0.95, top_k=0)
        with torch.no_grad():
            output = self.model.generate(input_ids=inputs, **kwargs)[0].tolist()
            old = token_logprobs(self.model, output, len(ids)).cpu().tolist() if self.record_segments else None
        generated = output[len(ids):]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.last_truncated = len(generated) >= budget and generated[-1] not in eos_ids
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        segment = dict(ids=output, prompt_length=len(ids), old_logprobs=old) if self.record_segments else None
        return text, segment, len(generated)
