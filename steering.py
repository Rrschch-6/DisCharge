#!/usr/bin/env python3
from contextlib import contextmanager
from dataclasses import asdict
import os
import torch
from energy import EnergyConfig, TokenEnergy, rollout


class SteeredLM:
    def __init__(self, model, energy, target_layer):
        self.model, self.energy, self.target_layer = model, energy, target_layer
        self.trainable = energy          # what train.py optimizes
        self._edit = None
        block = model.model.layers[target_layer]
        block.register_forward_pre_hook(self._hook, with_kwargs=True)
        self.input_device = model.get_input_embeddings().weight.device

    def _hook(self, module, args, kwargs):
        if self._edit is None:
            return None
        h = args[0] if args else kwargs['hidden_states']
        p = next(self.energy.parameters())
        z = self._edit(h.detach().to(device=p.device, dtype=p.dtype)).to(device=h.device, dtype=h.dtype)
        return ((z,) + args[1:], kwargs) if args else (args, dict(kwargs, hidden_states=z))

    @contextmanager
    def editing(self, edit):
        self._edit = edit
        try:
            yield
        finally:
            self._edit = None

    def edit_last(self, n, steps, alpha, noise_std=0., create_graph=False):
        """Edit function for a full sequence: rollout on the last n positions only.
        alpha: float, or a [1, n, 1] tensor of per-position step sizes."""
        def edit(h):
            c, _ = self.energy.context(h)
            z = rollout(self.energy, h[:, -n:], c[:, -n:], steps, alpha, noise_std, create_graph)
            return torch.cat((h[:, :-n], z), 1)
        return edit

    def span_logps(self, ids, start, end, *, steps=0, alpha=0., noise_std=0., train=False, full=False):
        """Per-token log p(ids[start:end]) under teacher forcing. steps=0 is the frozen LM.

        Predictors are positions start-1 .. end-2 = the last (end-start) inputs.
        full=True also returns the full next-token log-distributions [n, vocab].
        """
        if not 1 <= start < end <= len(ids):
            raise ValueError('Invalid scored span')
        n = end - start
        inputs = torch.tensor([ids[:end - 1]], device=self.input_device)
        edit = self.edit_last(n, steps, alpha, noise_std, create_graph=train) if steps else None
        with self.editing(edit), torch.set_grad_enabled(train):
            logits = self.model(input_ids=inputs, use_cache=False, logits_to_keep=n).logits
        targets = torch.tensor(ids[start:end], device=logits.device)
        logp = logits[0].float().log_softmax(-1)
        token = logp.gather(-1, targets[:, None]).squeeze(-1)
        return (token, logp) if full else token


def attach_generation_steering(slm, steps, alpha):
    """Make slm.model.generate steer.

    HF generate first runs the whole prompt (edit only its last position), then
    one new token per forward (edit it). Batched rows (num_return_sequences)
    are steered independently. Beam search, which reorders the cache, is not
    supported.
    """
    state = {}

    def edit(h):
        c, state['context'] = slm.energy.context(h, state.get('context'))
        k = 1 if 'started' not in state else h.shape[1]
        state['started'] = True
        if not steps:
            return h
        z = rollout(slm.energy, h[:, -k:], c[:, -k:], steps, alpha)
        return torch.cat((h[:, :-k], z), 1)

    original = slm.model.generate

    def generate(*args, **kwargs):
        if kwargs.get('num_beams', 1) > 1:
            raise NotImplementedError('Energy steering supports sampling/greedy, not beam search')
        state.clear()
        with slm.editing(edit):
            return original(*args, **kwargs)

    slm.model.generate = generate
    return slm.model


# ---------------------------------------------------------------- loading
ROUNDTRIP_SAMPLE = 'int main(void) {\n\tif (x  > 1)\n        return 0;\n}\n\ndef f(a, b):\n    return a  +  b\n'


def roundtrips(tokenizer, text=ROUNDTRIP_SAMPLE):
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text


def start_tokens(tokenizer):
    """Special tokens the tokenizer puts before a prompt by default (e.g. [BOS], or [])."""
    plain = tokenizer('x', add_special_tokens=False)['input_ids']
    full = tokenizer('x')['input_ids']
    for i in range(len(full) - len(plain) + 1):
        if full[i:i + len(plain)] == plain:
            return list(full[:i])
    return []


def config_adds_bos(name, tokenizer):
    """The checkpoint's own tokenizer_config.json setting add_bos_token (False if absent)."""
    value = getattr(tokenizer, 'init_kwargs', {}).get('add_bos_token')
    if value is None:
        try:
            import json
            from huggingface_hub import hf_hub_download
            value = json.load(open(hf_hub_download(name, 'tokenizer_config.json'))).get('add_bos_token')
        except Exception:
            value = None
    return bool(value)


def load_tokenizer(name):
    """Tokenizer that reproduces code exactly (whitespace included).

    Some checkpoints (e.g. DeepSeek-Coder-V2) declare a Llama tokenizer class but ship a
    byte-level tokenizer.json; newer transformers rebuild them Llama-style and lose
    whitespace. If AutoTokenizer fails the round trip, load tokenizer.json as-is.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(name)
    if roundtrips(tokenizer):
        return tokenizer
    raw = PreTrainedTokenizerFast.from_pretrained(name)
    for attr in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
        if getattr(raw, attr) is None and getattr(tokenizer, attr) is not None:
            setattr(raw, attr, getattr(tokenizer, attr))
    if (start_tokens(tokenizer) or config_adds_bos(name, tokenizer)) and not start_tokens(raw) \
            and raw.bos_token_id is not None:
        from tokenizers.processors import TemplateProcessing   # keep the original "prompt starts with BOS"
        bos = raw.bos_token
        raw._tokenizer.post_processor = TemplateProcessing(
            single=f'{bos} $A', pair=f'{bos} $A {bos} $B', special_tokens=[(bos, raw.bos_token_id)])
    if roundtrips(raw):
        print(f'{name}: AutoTokenizer ({type(tokenizer).__name__}) does not reproduce code; '
              f'using tokenizer.json directly (prompt start tokens {start_tokens(raw)})', flush=True)
        return raw
    raise RuntimeError(f'{name}: no tokenizer reproduces code exactly (whitespace is lost)')


def load_lm(name, dtype='bfloat16', device_map='auto'):
    from transformers import AutoModelForCausalLM
    if device_map != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available to PyTorch (check `nvidia-smi` and torch.cuda.is_available()); '
                           "refusing to run a 13B model on the CPU. Pass device_map='cpu' to force it.")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), device_map=device_map)
    offloaded = sorted({str(d) for d in (getattr(model, 'hf_device_map', None) or {}).values()} & {'cpu', 'disk'})
    if torch.cuda.is_available() and offloaded:
        raise RuntimeError(f'Model partly placed on {offloaded}: the visible GPUs lack free memory. '
                           'Pick free GPUs (nvidia-smi) instead of running on the CPU.')
    model.eval().requires_grad_(False)
    return model, load_tokenizer(name)


def make_steered(model, target_layer, energy_cfg=None, state=None):
    block = model.model.layers[target_layer]
    device = next(block.parameters()).device
    cfg = energy_cfg or EnergyConfig(input_dim=model.config.hidden_size)
    energy = TokenEnergy(cfg).to(device)
    if state is not None:
        energy.load_state_dict(state)
    return SteeredLM(model, energy, target_layer)


def save_checkpoint(path, slm, *, model_name, model_dtype, infer_steps, infer_alpha, step, args):
    payload = dict(format='ebt_v15', energy=slm.energy.state_dict(), energy_config=asdict(slm.energy.cfg),
                   target_layer=slm.target_layer, infer_steps=infer_steps, infer_alpha=infer_alpha,
                   model_name=model_name, model_dtype=model_dtype, step=step, args=args)
    tmp = str(path) + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_checkpoint(path):
    p = torch.load(path, map_location='cpu', weights_only=False)
    if p.get('format') != 'ebt_v15':
        raise ValueError(f'Unknown checkpoint format: {path}')
    return p


def load_policy(path, device_map='auto'):
    ckpt = read_checkpoint(path)
    model, tokenizer = load_lm(ckpt['model_name'], ckpt['model_dtype'], device_map)
    slm = make_steered(model, ckpt['target_layer'], EnergyConfig(**ckpt['energy_config']), ckpt['energy'])
    slm.energy.eval().requires_grad_(False)
    return slm, tokenizer, ckpt
