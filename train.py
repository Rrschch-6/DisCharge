#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from energy import EnergyConfig
from steering import load_lm, make_steered, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_argument
    a('--dataset', required=True, help='Saved dataset.pt with examples[train|valid]')
    a('--output-dir', required=True)
    a('--model-name', default='codellama/CodeLlama-13b-hf')
    a('--model-dtype', default='bfloat16', choices=('bfloat16', 'float32'))
    a('--target-layer', type=int, default=30, help='Edit the INPUT of this zero-based decoder block')
    a('--device-map', default='auto', help='HF device_map; use CUDA_VISIBLE_DEVICES to pick GPUs')
    # energy
    a('--width', type=int, default=128); a('--blocks', type=int, default=4)
    a('--context-width', type=int, default=128); a('--context-layers', type=int, default=2)
    a('--context-heads', type=int, default=4); a('--context-ff-mult', type=int, default=4)
    a('--context-dropout', type=float, default=.1); a('--max-length', type=int, default=2048)
    # trajectories
    a('--train-steps', type=int, nargs=2, default=(1, 4), metavar=('MIN', 'MAX'))
    a('--train-alpha', type=float, nargs=2, default=(.1, 1.), metavar=('MIN', 'MAX'))
    a('--noise-std', type=float, default=.005)
    a('--infer-steps', type=int, default=2)
    a('--infer-alpha', type=float, default=math.sqrt(.1))
    # objective
    a('--objective', default='dpo', choices=('dpo', 'simpo'))
    a('--dpo-beta', type=float, default=2.5, help='On per-token mean log-ratios')
    a('--simpo-beta', type=float, default=2.5); a('--simpo-gamma', type=float, default=1.25)
    a('--kl-lambda', type=float, default=.5, help='Reverse KL to the frozen LM on safe-branch positions')
    a('--preserve-lambda', type=float, default=.5); a('--preserve-delta', type=float, default=0.)
    a('--unsafe-floor', type=float, help='Cap suppression: penalize unsafe dropping more than this '
                                         'many nats/token below the frozen LM (default: off)')
    a('--floor-lambda', type=float, default=1.)
    # checkpoint selection: best.pt = highest valid margin among evals within both limits
    a('--select-max-kl', type=float, default=.05)
    a('--select-max-safe-degradation', type=float, default=.05)
    # optimization
    a('--lr', type=float, default=1e-4); a('--weight-decay', type=float, default=.1)
    a('--grad-clip', type=float, default=1.); a('--epochs', type=int, default=10)
    a('--steps', type=int, help='Optimizer updates (default: epochs * train pairs)')
    a('--seed', type=int, default=42)
    # bookkeeping
    a('--eval-every', type=int, default=128); a('--save-every', type=int, default=4096)
    a('--log-every', type=int, default=32)
    a('--reference-from', help='Reuse reference.pt from an earlier run with the same model and dataset')
    a('--wandb-project')
    a('--config', nargs='+', default=[], metavar='JSON',
      help='JSON files of argument defaults, applied in order (e.g. configs/discharge.json configs/models/'
           'codellama-13b.json); keys are argument names; command-line flags override them')
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument('--config', nargs='+', default=[])
    pre, _ = cfg_parser.parse_known_args(argv)
    defaults = {}
    for path in pre.config:
        for key, value in json.load(open(path)).items():
            if key.startswith('_'):                     # "_comment" and similar keys are documentation
                continue
            dest = key.lstrip('-').replace('-', '_')
            action = next((x for x in p._actions if x.dest == dest), None)
            if action is None:
                p.error(f'{path}: unknown argument {key!r}')
            action.required = False                     # supplied by the config
            defaults[dest] = value
    p.set_defaults(**defaults)
    args = p.parse_args(argv)
    if not (1 <= args.train_steps[0] <= args.train_steps[1] and 0 < args.train_alpha[0] <= args.train_alpha[1]):
        p.error('Invalid --train-steps or --train-alpha range')
    return args


def objective(safe, unsafe, ref_safe, ref_unsafe, kl, args):
    """Scalars in, (total loss, components dict) out. kl: mean reverse KL on safe-branch positions."""
    if args.objective == 'dpo':
        pref = -F.logsigmoid(args.dpo_beta * ((safe - ref_safe) - (unsafe - ref_unsafe)))
    else:
        pref = F.softplus(args.simpo_gamma - args.simpo_beta * (safe - unsafe))
    preserve = F.relu(ref_safe - safe - args.preserve_delta)
    floor = F.relu(ref_unsafe - unsafe - args.unsafe_floor) if args.unsafe_floor is not None else safe * 0
    total = pref + args.kl_lambda * kl + args.preserve_lambda * preserve + args.floor_lambda * floor
    return total, dict(pref=pref, kl=kl, preserve=preserve, floor=floor)


def reverse_kl(steered_logp, frozen_logp):
    """Mean over positions of KL(steered || frozen); inputs are [n, vocab] log-distributions."""
    return (steered_logp.exp() * (steered_logp - frozen_logp)).sum(-1).mean()


def score_pair(slm, ex, *, steps, alpha_for, noise_std, train):
    """Steered mean span logp of both branches and the safe branch's reverse KL to the frozen LM."""
    ids, start, end = branch(ex, 'safe')
    with torch.no_grad():
        _, frozen = slm.span_logps(ids, start, end, full=True)
    tok, steered = slm.span_logps(ids, start, end, steps=steps, alpha=alpha_for(end - start),
                                  noise_std=noise_std, train=train, full=True)
    ids, start, end = branch(ex, 'unsafe')
    unsafe = slm.span_logps(ids, start, end, steps=steps, alpha=alpha_for(end - start),
                            noise_std=noise_std, train=train).mean()
    return tok.mean(), unsafe, reverse_kl(steered, frozen)


def selection_ok(v, args):
    return v['kl'] <= args.select_max_kl and v['safe_degradation'] <= args.select_max_safe_degradation


def branch(ex, name):
    return ex[name + '_input_ids'], ex[name + '_preference_start'], ex[name + '_preference_end']


@torch.no_grad()
def reference_scores(slm, examples):
    """Frozen-LM mean span logp per pair, computed once (the LM never changes)."""
    out, started = {}, time.time()
    for split, rows in examples.items():
        scores = []
        for i, ex in enumerate(rows, 1):
            scores.append([float(slm.span_logps(*branch(ex, b)).mean()) for b in ('safe', 'unsafe')])
            if i % 250 == 0 or i == len(rows):
                print(f'reference {split}: {i}/{len(rows)} pairs ({time.time() - started:.0f}s)', flush=True)
        out[split] = torch.tensor(scores, dtype=torch.float64)
    return out


@torch.no_grad()
def validate(slm, rows, refs, args):
    slm.trainable.eval()
    sums = {}
    for ex, (ref_safe, ref_unsafe) in zip(rows, refs.tolist()):
        safe, unsafe, kl = score_pair(slm, ex, steps=args.infer_steps, alpha_for=lambda n: args.infer_alpha,
                                      noise_std=0., train=False)
        loss, parts = objective(safe, unsafe, ref_safe, ref_unsafe, kl, args)
        margin = float(safe - unsafe)
        row = dict(loss=loss, **parts, margin=margin, accuracy=margin > 0, base_margin=ref_safe - ref_unsafe,
                   reward_margin=margin - (ref_safe - ref_unsafe), safe_logp=safe, unsafe_logp=unsafe,
                   ref_safe_logp=ref_safe, ref_unsafe_logp=ref_unsafe, safe_degradation=ref_safe - float(safe),
                   unsafe_degradation=ref_unsafe - float(unsafe))
        for k, v in row.items():
            sums[k] = sums.get(k, 0.) + float(v)
    slm.trainable.train()
    return {k: v / len(rows) for k, v in sums.items()}


def make_optimizer(module, args):
    params = [p for p in module.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    rest = [p for p in params if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=args.weight_decay),
                              dict(params=rest, weight_decay=0.)], lr=args.lr)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=2))

    examples = torch.load(args.dataset, map_location='cpu', weights_only=False)['examples']
    examples = {k: examples[k] for k in ('train', 'valid')}
    model, tokenizer = load_lm(args.model_name, args.model_dtype, args.device_map)
    vocab = model.config.vocab_size
    for rows in examples.values():
        for ex in rows:
            for b in ('safe', 'unsafe'):
                ids = ex[b + '_input_ids']
                if max(ids) >= vocab or len(ids) > args.max_length:
                    raise ValueError('Dataset token IDs or lengths do not fit this model/--max-length')
    cfg = EnergyConfig(input_dim=model.config.hidden_size, width=args.width, blocks=args.blocks,
                       max_length=args.max_length, context_width=args.context_width,
                       context_layers=args.context_layers, context_heads=args.context_heads,
                       context_ff_mult=args.context_ff_mult, context_dropout=args.context_dropout)
    slm = make_steered(model, args.target_layer, cfg)
    trainable = [p for p in slm.trainable.parameters() if p.requires_grad]
    print(f'trainable parameters: {sum(p.numel() for p in trainable):,}', flush=True)
    device = trainable[0].device

    meta = dict(model_name=args.model_name, model_dtype=args.model_dtype,
                sizes={k: len(v) for k, v in examples.items()})
    if args.reference_from:
        saved = torch.load(args.reference_from, weights_only=False)
        if saved['meta'] != meta:
            raise ValueError('reference.pt was computed for a different model or dataset')
        refs = saved['scores']
    else:
        refs = reference_scores(slm, examples)
    torch.save(dict(meta=meta, scores=refs), out / 'reference.pt')

    optimizer = make_optimizer(slm.trainable, args)
    total = args.steps or args.epochs * len(examples['train'])
    ckpt = lambda name, step: save_checkpoint(out / name, slm, model_name=args.model_name,
        model_dtype=args.model_dtype, infer_steps=args.infer_steps, infer_alpha=args.infer_alpha,
        step=step, args=vars(args))
    wandb = None
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, dir=str(out), config=vars(args), name=out.name)
    log = (out / 'train.jsonl').open('a')

    def record(step, metrics):
        log.write(json.dumps(dict(step=step, **metrics)) + '\n'); log.flush()
        if wandb:
            wandb.log(metrics, step=step)

    def run_validation(step, best):
        """best.pt = highest valid margin among evals within the KL and safe-degradation limits."""
        v = validate(slm, examples['valid'], refs['valid'], args)
        ok = selection_ok(v, args)
        record(step, {**{'valid/' + k: x for k, x in v.items()}, 'valid/selectable': float(ok)})
        print(f'[valid {step}] ' + ' '.join(f'{k}={x:.4f}' for k, x in v.items()) + f' selectable={ok}', flush=True)
        if ok and v['margin'] > best:
            ckpt('best.pt', step)
            (out / 'best.json').write_text(json.dumps(dict(step=step, **v), indent=2))
            return v['margin']
        return best

    slm.trainable.train()
    ckpt('initial.pt', 0)
    best = run_validation(0, -math.inf)
    order, window, started = [], [], time.time()
    for step in range(1, total + 1):
        if not order:
            order = list(range(len(examples['train']))); random.shuffle(order)
        i = order.pop()
        ex, (ref_safe, ref_unsafe) = examples['train'][i], refs['train'][i].tolist()
        steps = random.randint(*args.train_steps)
        alpha_for = lambda n: torch.empty(1, n, 1, device=device).uniform_(*args.train_alpha)
        scores = score_pair(slm, ex, steps=steps, alpha_for=alpha_for, noise_std=args.noise_std, train=True)
        loss, parts = objective(*scores[:2], ref_safe, ref_unsafe, scores[2], args)
        safe, unsafe = (float(s.detach()) for s in scores[:2])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        margin = safe - unsafe
        window.append(dict(loss=float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()},
                           margin=margin, accuracy=float(margin > 0), safe_logp=safe, unsafe_logp=unsafe,
                           ref_safe_logp=ref_safe, ref_unsafe_logp=ref_unsafe, base_margin=ref_safe - ref_unsafe,
                           reward_margin=margin - (ref_safe - ref_unsafe), safe_degradation=ref_safe - safe,
                           unsafe_degradation=ref_unsafe - unsafe, grad_norm=float(grad_norm), steps=steps))
        if step % args.log_every == 0 or step == total:
            mean = {k: sum(r[k] for r in window) / len(window) for k in window[0]}
            record(step, {'train/' + k: x for k, x in mean.items()})
            print(f'[train {step}/{total} {time.time() - started:.0f}s] '
                  + ' '.join(f'{k}={x:.4f}' for k, x in mean.items()), flush=True)
            window = []
        if step % args.eval_every == 0 or step == total:
            best = run_validation(step, best)
        if step % args.save_every == 0:
            ckpt(f'checkpoint_step_{step:06d}.pt', step)
    ckpt('final.pt', total)
    log.close()
    if wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
