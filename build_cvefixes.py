#!/usr/bin/env python3
import argparse
from collections import Counter
import csv
from difflib import SequenceMatcher
import io
import json
import random
import re
import sys
import tokenize
import torch
from steering import start_tokens

C_FAMILY = {'C', 'C++', 'Java', 'PHP', 'JavaScript', 'Go'}
SPLITS = (('train', 'train.csv'), ('valid', 'validation.csv'), ('test', 'test.csv'))


# ------------------------------------------------------------comments 
def _strip_c_family(code, lang):
    out, i, n = [], 0, len(code)
    while i < n:
        ch, nxt = code[i], code[i + 1] if i + 1 < n else ''
        if ch == '/' and nxt == '/' or lang == 'PHP' and ch == '#' and nxt != '[':
            while i < n and code[i] != '\n':
                i += 1
        elif ch == '/' and nxt == '*':
            end = code.find('*/', i + 2)
            end = n if end < 0 else end + 2
            out.append('\n' * code.count('\n', i, end))   # keep line numbering
            i = end
        elif ch in '"\'`':
            j = i + 1
            while j < n and code[j] != ch:
                j += 2 if code[j] == '\\' and ch != '`' else 1
            out.append(code[i:j + 1]); i = j + 1
        else:
            out.append(ch); i += 1
    return ''.join(out)


def _strip_python(code):
    lines = code.splitlines(keepends=True)
    cuts = [(t.start, t.end) for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type == tokenize.COMMENT]
    for (row, col), (_, end) in reversed(cuts):
        line = lines[row - 1]
        lines[row - 1] = line[:col] + line[end:]
    return ''.join(lines)


def strip_comments(code, lang):
    """Remove comments, preserving line count; drop lines that held only a comment."""
    if lang == 'Python':
        stripped = _strip_python(code)
    elif lang in C_FAMILY:
        stripped = _strip_c_family(code, lang)
    else:
        raise ValueError(f'unsupported_language_{lang}')
    original, kept = code.splitlines(), stripped.splitlines()
    if len(original) != len(kept):
        raise ValueError('comment_strip_changed_line_count')
    return '\n'.join(s.rstrip() for o, s in zip(original, kept) if s.strip() or not o.strip()) + '\n'


def normalized(code):
    return re.sub(r'\s+', '', code)


# ------------------------------------------------------------ pairs
def edit_envelope(safe, unsafe, context_lines=3):
    """Character span per branch covering every changed line plus shared context lines."""
    lines = [safe.splitlines(keepends=True), unsafe.splitlines(keepends=True)]
    changes = [op for op in SequenceMatcher(None, *lines, autojunk=False).get_opcodes() if op[0] != 'equal']
    if not changes:
        raise ValueError('identical_code')
    spans = []
    for b in (0, 1):
        start = max(0, min(op[1 + 2 * b] for op in changes) - context_lines)
        end = min(len(lines[b]), max(op[2 + 2 * b] for op in changes) + context_lines)
        if end <= start:
            raise ValueError('empty_edit_envelope')
        spans.append((sum(map(len, lines[b][:start])), sum(map(len, lines[b][:end]))))
    return spans


def token_span_by_decoding(tokenizer, ids, c0, c1, code):
    """Token range [first, end) covering characters [c0, c1), found by decoding growing prefixes.

    Fallback for tokenizers whose offset mapping is unusable for some inputs. Requires that
    decoding reproduces the code exactly; binary search keeps it to O(log n) decodes.
    """
    decode = lambda k: tokenizer.decode(ids[:k], clean_up_tokenization_spaces=False)
    if decode(len(ids)) != code:
        return None

    def first_k(target):   # smallest k with len(decode(ids[:k])) >= target
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(decode(mid)) >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo
    first = max(first_k(c0 + 1) - 1, 0)   # token containing character c0
    end = first_k(c1)                       # tokens up to and including the one ending at/after c1
    return (first, end) if end > first else None


def make_example(row, tokenizer, args):
    lang = row['lang']
    safe, unsafe = row['after'], row['before']
    if not args.keep_comments:
        safe, unsafe = strip_comments(safe, lang), strip_comments(unsafe, lang)
    if normalized(safe) == normalized(unsafe):
        raise ValueError('no_code_change' if args.keep_comments else 'no_code_change_after_comment_removal')
    branches = []
    for code, (c0, c1) in zip((safe, unsafe), edit_envelope(safe, unsafe)):
        enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc['input_ids'])
        inside = [i for i, (s, e) in enumerate(enc['offset_mapping']) if e > c0 and s < c1 and e > s]
        if inside:
            end = inside[-1] + 1
        else:   # unusable offsets (seen with DeepSeek-Coder-V2's tokenizer): locate the span by decoding
            span = token_span_by_decoding(tokenizer, ids, c0, c1, code)
            if span is None:
                raise ValueError('empty_token_span')
            end = span[1]
        branches.append((ids, end))
    shared = 0
    for x, y in zip(branches[0][0], branches[1][0]):
        if x != y:
            break
        shared += 1
    if any(shared >= end for _, end in branches):
        raise ValueError('empty_span_after_divergence')   # pure deletion/insertion at envelope end
    if max(end - shared for _, end in branches) > args.max_span_tokens:
        raise ValueError('span_exceeds_max_span_tokens')
    prefix = start_tokens(tokenizer)   # exactly what the tokenizer puts before a prompt at generation time
    # One common left crop, only inside the shared prefix, keeping >= context_tokens of context.
    crop = max(0, max(end for _, end in branches) + len(prefix) - args.max_length)
    if crop > max(0, shared - args.context_tokens):
        raise ValueError('exceeds_max_length')
    ex = dict(pair_id=row['pair_id'], lang=lang, cwes=row['cwes'], vulnerability_type=row['vulnerability_type'],
              repo=row['repo'], sha=row['sha'], n_changed_in_commit=int(row['n_changed_in_commit'] or 0),
              left_crop_tokens=crop)
    for name, (ids, end) in zip(('safe', 'unsafe'), branches):
        start = shared - crop + len(prefix)
        if start < 1:
            raise ValueError('no_predictor_context')
        ex.update({name + '_input_ids': prefix + ids[crop:end], name + '_preference_start': start,
                   name + '_preference_end': end - crop + len(prefix)})
    return ex


def build_split(path, tokenizer, args, seen):
    csv.field_size_limit(sys.maxsize)
    examples, drops, read = [], Counter(), 0
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            read += 1
            if row['lang'] not in args.langs:
                drops['language_not_selected'] += 1; continue
            if not args.keep_unknown_cwe and row['cwe_known'] != 'True':
                drops['unknown_cwe'] += 1; continue
            if args.single_file_only and row['n_changed_in_commit'] != '1':
                drops['multi_file_commit'] += 1; continue
            try:
                ex = make_example(row, tokenizer, args)
            except (ValueError, SyntaxError, tokenize.TokenError) as error:
                reason = str(error) if isinstance(error, ValueError) else 'python_tokenize_error'
                drops[reason] += 1; continue
            key = (normalized(tokenizer.decode(ex['safe_input_ids'])), normalized(tokenizer.decode(ex['unsafe_input_ids'])))
            if key in seen:
                drops['duplicate'] += 1; continue
            seen.add(key)
            examples.append(ex)
    return examples, dict(read=read, kept=len(examples), drops=dict(drops.most_common()))


def summary(rows):
    span = sorted(ex['safe_preference_end'] - ex['safe_preference_start'] for ex in rows)
    return dict(pairs=len(rows), languages=dict(Counter(ex['lang'] for ex in rows).most_common()),
                vulnerability_types=dict(Counter(ex['vulnerability_type'] for ex in rows).most_common()),
                top_cwes=dict(Counter(ex['cwes'].split(';')[0] for ex in rows).most_common(15)),
                safe_span_tokens_median=span[len(span) // 2] if span else 0,
                input_tokens_median=sorted(len(ex['safe_input_ids']) for ex in rows)[len(rows) // 2] if rows else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-dir', required=True, help='Directory with train.csv, validation.csv, test.csv')
    p.add_argument('--output', required=True, help='dataset.pt to write')
    p.add_argument('--model-name', default='codellama/CodeLlama-13b-hf', help='Tokenizer to use')
    p.add_argument('--langs', default='C,C++,Python', help='Comma-separated; CodeGuard+ covers C, C++, Python')
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--context-tokens', type=int, default=64)
    p.add_argument('--max-span-tokens', type=int, default=512, help='Drop large rewrites')
    p.add_argument('--valid-pairs', type=int, default=256, help='Random validation subset (0 = all)')
    p.add_argument('--keep-comments', action='store_true')
    p.add_argument('--keep-unknown-cwe', action='store_true')
    p.add_argument('--single-file-only', action='store_true', help='Keep only isolated one-file fixes')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args(argv)
    args.langs = set(args.langs.split(','))
    from steering import load_tokenizer
    tokenizer = load_tokenizer(args.model_name)
    print(f'prompt start tokens: {start_tokens(tokenizer)}', flush=True)
    return build(args, tokenizer)


def build(args, tokenizer):
    examples, audit, seen = {}, {}, set()
    for split, filename in SPLITS:
        examples[split], audit[split] = build_split(f'{args.csv_dir}/{filename}', tokenizer, args, seen)
    full_valid = len(examples['valid'])
    if args.valid_pairs and full_valid > args.valid_pairs:
        examples['valid'] = random.Random(args.seed).sample(examples['valid'], args.valid_pairs)
    random.Random(args.seed).shuffle(examples['train'])
    config = {k: sorted(v) if isinstance(v, set) else v for k, v in vars(args).items()}
    meta = dict(source='CVEfixes v1.0.8', tokenizer=args.model_name, config=config,
                build=audit, valid_subsampled_from=full_valid, summary={k: summary(v) for k, v in examples.items()})
    torch.save(dict(examples=examples, audit=meta), args.output)
    print(json.dumps({k: meta[k] for k in ('build', 'summary', 'valid_subsampled_from')}, indent=2))
    return examples, meta


if __name__ == '__main__':
    main()
