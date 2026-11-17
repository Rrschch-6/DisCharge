#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict

CATEGORIES = {   # paper's grouping; Path traversal = CWE-22 through CWE-40
    'Memory safety': {119, 120, 125, 787, 476, 416, 190},
    'Injection': {79, 89, 94, 77, 78, 88},
    'Input validation': {20},
    'Resource handling': {399, 400, 401, 772},
    'Access control and authentication': {264, 284, 287, 269},
}
LANGUAGES = ('C', 'C++', 'Python', 'Java', 'PHP', 'JavaScript', 'Go')
REQUIRED = {'fixes': {'cve_id', 'hash', 'repo_url'},
            'file_change': {'file_change_id', 'hash', 'programming_language'},
            'method_change': {'file_change_id', 'name', 'signature', 'code', 'before_change'},
            'cwe_classification': {'cve_id', 'cwe_id'}}
csv.field_size_limit(sys.maxsize)


def category_of(cwe):
    for name, ids in CATEGORIES.items():
        if cwe in ids:
            return name
    return 'Path traversal' if 22 <= cwe <= 40 else 'Other'


def vulnerability_type(cwes):
    """Category of a pair from all its CWE labels: one category, 'Mixed' if several, 'Unknown' if none."""
    numeric = [int(c.split('-')[1]) for c in cwes if re.fullmatch(r'CWE-\d+', c)]
    cats = {category_of(c) for c in numeric}
    return 'Unknown' if not cats else (cats.pop() if len(cats) == 1 else 'Mixed')


def check_schema(con):
    for table, cols in REQUIRED.items():
        have = {r[1] for r in con.execute(f'PRAGMA table_info({table})')}
        if not have:
            raise SystemExit(f'table {table} not found: is this the CVEfixes SQLite database?')
        if cols - have:
            raise SystemExit(f'table {table} lacks columns {sorted(cols - have)}')


def norm(code):
    return re.sub(r'\s+', ' ', code or '').strip()


def extract(con, languages, max_lines):
    commit_cwes, commit_repo = defaultdict(list), {}
    for cve, sha, repo in con.execute('SELECT cve_id, hash, repo_url FROM fixes'):
        commit_repo.setdefault(sha, repo)
        for (cwe,) in con.execute('SELECT cwe_id FROM cwe_classification WHERE cve_id = ?', (cve,)):
            if cwe not in commit_cwes[sha]:
                commit_cwes[sha].append(cwe)
    files = {fid: (sha, lang) for fid, sha, lang in con.execute(
        'SELECT file_change_id, hash, programming_language FROM file_change')
        if lang in languages and sha in commit_repo}
    methods = defaultdict(lambda: {'True': [], 'False': []})
    for fid, name, sig, code, before in con.execute(
            'SELECT file_change_id, name, signature, code, before_change FROM method_change'):
        if fid in files and str(before) in ('True', 'False') and code:
            methods[fid][str(before)].append((name, sig, code))
    pairs, seen = [], set()
    for fid, sides in methods.items():
        sha, lang = files[fid]
        after_by_sig = {(n, s): c for n, s, c in sides['False']}
        after_by_name = Counter(n for n, _, _ in sides['False'])
        for name, sig, before in sides['True']:
            after = after_by_sig.get((name, sig))
            if after is None and after_by_name[name] == 1:
                after = next(c for n, _, c in sides['False'] if n == name)
            if after is None or norm(before) == norm(after):
                continue
            if max(before.count('\n'), after.count('\n')) + 1 > max_lines:
                continue
            key = (hashlib.sha1(before.encode()).hexdigest(), hashlib.sha1(after.encode()).hexdigest())
            if key in seen:
                continue
            seen.add(key)
            cwes = commit_cwes.get(sha, [])
            pairs.append(dict(pair_id=hashlib.sha1(f'{sha}|{fid}|{name}|{sig}'.encode()).hexdigest()[:16], lang=lang,
                              cwes=';'.join(cwes), cwe_known=str(any(re.fullmatch(r'CWE-\d+', c) for c in cwes)),
                              vulnerability_type=vulnerability_type(cwes), repo=commit_repo[sha], sha=sha,
                              before=before, after=after))
    per_commit = Counter(p['sha'] for p in pairs)
    for p in pairs:
        p['n_changed_in_commit'] = per_commit[p['sha']]
    return pairs


def split_by_repo(pairs, fractions, seed):
    groups = defaultdict(list)
    for p in pairs:
        groups[p['repo']].append(p)
    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda r: -len(groups[r]))          # stable: equal sizes keep the seeded order
    names, total = list(fractions), len(pairs)
    target = {n: fractions[n] * total for n in names}
    out = {n: [] for n in names}
    for repo in order:
        n = max(names, key=lambda k: (target[k] - len(out[k])) / target[k])
        out[n] += groups[repo]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', required=True, help='CVEfixes SQLite database (CVEfixes.db)')
    p.add_argument('--out-dir', default='data/cvefixes')
    p.add_argument('--languages', default=','.join(LANGUAGES), help='Comma-separated programming languages to keep')
    p.add_argument('--max-lines', type=int, default=1000, help='Drop functions longer than this (either version)')
    p.add_argument('--fractions', type=float, nargs=3, default=(.70, .15, .15), metavar=('TRAIN', 'VALID', 'TEST'))
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if abs(sum(args.fractions) - 1) > 1e-6:
        p.error('--fractions must sum to 1')

    con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    check_schema(con)
    pairs = extract(con, set(args.languages.split(',')), args.max_lines)
    if not pairs:
        raise SystemExit('no function pairs found; check --languages and the database')
    splits = split_by_repo(pairs, dict(zip(('train', 'validation', 'test'), args.fractions)), args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ['pair_id', 'lang', 'cwes', 'cwe_known', 'vulnerability_type', 'repo', 'sha', 'n_changed_in_commit',
            'before', 'after']
    stats = dict(pairs=len(pairs), repositories=len({p['repo'] for p in pairs}),
                 primary_cwes=len({p['cwes'].split(';')[0] for p in pairs if p['cwes']}), splits={})
    for name, rows in splits.items():
        with open(os.path.join(args.out_dir, f'{name}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, cols)
            w.writeheader(); w.writerows(rows)
        stats['splits'][name] = dict(pairs=len(rows), languages=dict(Counter(r['lang'] for r in rows)),
                                     categories=dict(Counter(r['vulnerability_type'] for r in rows)))
    json.dump(stats, open(os.path.join(args.out_dir, 'stats.json'), 'w'), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != 'splits'}))
    for name, s in stats['splits'].items():
        print(f"{name:<11} {s['pairs']:>6} pairs, {len(s['languages'])} languages, {len(s['categories'])} categories")


if __name__ == '__main__':
    main()
