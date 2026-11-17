#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys


def install_compat_shims():
    """Let CodeGuard+'s generation code import under newer transformers.

    Both stubs cover code paths that never run here: the bundled Salesforce
    CodeGen model (only for Salesforce/codegen-* checkpoints) and phrasal
    constraints (constrained beam search, rejected by our steering anyway).
    """
    import types
    import torch
    import transformers
    hf = types.ModuleType('inference.hf')

    class CodeGenForCausalLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('Salesforce CodeGen is not supported by the EBT adapter')

    hf.CodeGenForCausalLM = CodeGenForCausalLM
    sys.modules['inference.hf'] = hf
    try:
        transformers.PhrasalConstraint
    except (AttributeError, ImportError):
        class PhrasalConstraint:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError('Constrained beam search is not supported by the EBT adapter')
        transformers.PhrasalConstraint = PhrasalConstraint


def shard_cwes(cwes, data_dir, k, n):
    """Deterministic split of CWE groups into n shards with balanced scenario counts."""
    if not 0 <= k < n:
        raise SystemExit('--shard must be K/N with 0 <= K < N')
    loads, shards = [0] * n, [[] for _ in range(n)]
    sizes = {c: len(os.listdir(os.path.join(data_dir, c))) for c in cwes}
    for cwe in sorted(cwes, key=lambda c: (-sizes[c], c)):
        i = loads.index(min(loads))
        shards[i].append(cwe); loads[i] += sizes[cwe]
    return [c for c in cwes if c in shards[k]]


def main():
    p = argparse.ArgumentParser(add_help=False)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument('--checkpoint', help='DisCharge checkpoint (best.pt from train.py)')
    which.add_argument('--model-name', help='Baseline: the unsteered Hugging Face model, no checkpoint')
    p.add_argument('--model-dtype', default='bfloat16', choices=('bfloat16', 'float32'), help='With --model-name')
    p.add_argument('--steps', type=int, help='Default: checkpoint inference steps; 0 = unsteered LM')
    p.add_argument('--alpha', type=float, help='Default: checkpoint inference step size')
    p.add_argument('--alpha-scale', type=float, default=1.)
    p.add_argument('--shard', help='K/N: generate only shard K of N (CWE groups balanced by scenario count)')
    p.add_argument('--cwes', help='Comma-separated CWE groups to generate (e.g. cwe-078,cwe-089); resumes partial runs')
    ours, rest = p.parse_known_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from steering import attach_generation_steering, load_lm, load_policy, read_checkpoint
    ckpt = (read_checkpoint(ours.checkpoint) if ours.checkpoint else
            dict(format='baseline', model_name=ours.model_name, model_dtype=ours.model_dtype))
    steps = ckpt.get('infer_steps', 0) if ours.steps is None else ours.steps
    alpha = (ckpt.get('infer_alpha', 0.) if ours.alpha is None else ours.alpha) * ours.alpha_scale

    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, 'inference', 'evaler.py')):
        raise SystemExit('Run this from the CodeGuard+ directory')
    sys.path.insert(0, cwd)
    install_compat_shims()
    spec = importlib.util.spec_from_file_location('codeguard_generate', os.path.join(cwd, 'generate.py'))
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)

    def load_model(self):
        if ckpt['format'] == 'baseline':
            model, tokenizer = load_lm(ckpt['model_name'], ckpt['model_dtype'])
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token_id = tokenizer.bos_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            self.tokenizer, self.model = tokenizer, model
            self.input_device = model.get_input_embeddings().weight.device
            self.args.logger.info(f"baseline (unsteered): {ckpt['model_name']} ({ckpt['model_dtype']})")
            return
        slm, tokenizer, _ = load_policy(ours.checkpoint)
        if tokenizer.eos_token_id is None:            # same fixes as CodeGuard+'s load_model
            tokenizer.eos_token_id = tokenizer.bos_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer, self.input_device = tokenizer, slm.input_device
        self.model = attach_generation_steering(slm, steps, alpha)
        self.args.logger.info(f"EBT {ckpt['format']} step {ckpt['step']}: layer {ckpt['target_layer']}, "
                              f'steps={steps}, alpha={alpha:.6f}, checkpoint={ours.checkpoint}')

    generate.LMEvaler.load_model = load_model
    if ours.shard:
        k, n = map(int, ours.shard.split('/'))
        eval_type = rest[rest.index('--eval_type') + 1] if '--eval_type' in rest else 'base'
        mine = shard_cwes(generate.CWES_DICT[eval_type], os.path.join(cwd, 'data', eval_type), k, n)
        generate.CWES_DICT = {**generate.CWES_DICT, eval_type: mine}
        print(f'shard {k}/{n}: {len(mine)} CWEs: {" ".join(mine)}', flush=True)
    if ours.cwes:
        eval_type = rest[rest.index('--eval_type') + 1] if '--eval_type' in rest else 'base'
        mine = [c for c in ours.cwes.split(',') if c]
        unknown = sorted(set(mine) - set(generate.CWES_DICT[eval_type]))
        if unknown:
            raise SystemExit(f'unknown CWE groups for {eval_type}: {unknown}')
        generate.CWES_DICT = {**generate.CWES_DICT, eval_type: mine}
        print(f'CWE list: {len(mine)} CWEs: {" ".join(mine)}', flush=True)
    sys.argv = [sys.argv[0], '--model_type', 'lm', '--model_dir', ckpt['model_name']] + rest
    generate.main()


if __name__ == '__main__':
    main()
