#!/usr/bin/env python3
import argparse
import contextlib
import csv
import fcntl
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

METRICS = ('pass@1', 'sec_rate', 'sec-pass@1')
# Unit tests that write fixed /tmp paths: unreliable on a shared machine, excluded from 'clean'.
TMP_SCENARIOS = {'cwe-643/1-py', 'cwe-022/2-py', 'cwe-022/4-py', 'cwe-170/0-c',
                 'cwe-611/2-py', 'cwe-611/3-py', 'cwe-611/4-py', 'cwe-611/5-py'}


def scenario_dirs(name):
    """{'cwe-xxx/scenario': dir} for one experiment (single seed directory)."""
    seeds = sorted(glob.glob(f'experiments/{name}/*/base'))
    if len(seeds) != 1:
        raise SystemExit(f'experiments/{name}: expected one seed directory, found {seeds or "none"}')
    out = {}
    for stat in glob.glob(f'{seeds[0]}/*/*/stat.json'):
        d = os.path.dirname(stat)
        out[f'{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}'] = d
    return seeds[0].split('/')[2], out


def programs(stat):
    return {k: v for k, v in stat.items() if isinstance(v, dict) and 'num' in v}


def missing(dirs, key):
    return sorted(s for s, d in dirs.items()
                  if any(key not in v for v in programs(json.load(open(f'{d}/stat.json'))).values()))


@contextlib.contextmanager
def output_to(log):
    """Send this process's stdout/stderr (and its subprocesses') to the log file."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log, 'a') as f:
        os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def has_query(scenario):
    """False for scenarios CodeGuard+ checks with SonarQube only (check_ql is null)."""
    return json.load(open(f'data/base/{scenario}/info.json')).get('check_ql') is not None


def run_codeql(name, seed, scenarios, log):
    """CodeGuard+'s codeql_eval.eval_single per scenario; returns scenarios whose CodeQL run failed."""
    sys.path.insert(0, os.getcwd())
    import codeql_eval
    failed = []
    print(f'  CodeQL on {len(scenarios)} scenarios', flush=True)
    for s in scenarios:
        cwe, scenario = s.split('/')
        stale = os.path.join(f'experiments/{name}', seed, 'base', cwe, scenario, 'codeql_db')
        shutil.rmtree(stale, ignore_errors=True)     # half-built database from an interrupted or clashing run
        try:
            with output_to(log):
                codeql_eval.eval_single(None if seed == 'none' else int(seed), f'experiments/{name}', cwe, scenario,
                                        category='base')
        except Exception as e:
            failed.append(s)
            with open(log, 'a') as f:
                f.write(f'CodeQL failed for {s}: {e!r}\n')
    return failed


def jail(private_dir, proot):
    """Command prefix that runs a command with /tmp bound to private_dir, or None if impossible."""
    if shutil.which('unshare') and subprocess.run(['unshare', '-rm', 'true'], capture_output=True).returncode == 0:
        return ['unshare', '-rm', 'sh', '-c', 'mount --bind "$0" /tmp && exec "$@"', private_dir]
    if proot and os.access(os.path.expanduser(proot), os.X_OK):
        return [os.path.expanduser(proot), '-b', f'{private_dir}:/tmp']
    return None


def rerun_tmp_tests(name, seed, dirs, args, log):
    """Rerun the fixed-/tmp unit tests with a private /tmp; returns the scenarios rerun."""
    done = []
    for s in sorted(TMP_SCENARIOS & set(dirs)):
        path = f'{dirs[s]}/stat.json'
        stat = json.load(open(path))
        if not programs(stat):
            continue
        for v in programs(stat).values():
            v.pop('functional', None)
        json.dump(stat, open(path, 'w'), indent=4)
        private = tempfile.mkdtemp(prefix='cg_tmp_', dir=os.path.expanduser('~'))
        try:
            prefix = jail(private, args.proot)
            if prefix is None:
                raise SystemExit('--private-tmp: `unshare -rm` is not permitted and no usable --proot binary was '
                                 'given; see README (CodeGuard+ benchmark) for installing proot')
            cwe, scenario = s.split('/')
            run(prefix + [sys.executable, f'unit_test/{cwe}/{scenario}/functional.py', '--path',
                          os.path.abspath(os.path.join(os.path.dirname(path), 'deduplicated'))], log, check=False)
            done.append(s)
        finally:
            shutil.rmtree(private, ignore_errors=True)
    print(f'  reran with a private /tmp: {" ".join(done) or "none"}', flush=True)
    return done


def run(cmd, log, check=True):
    print('  $ ' + ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode and check:
            raise SystemExit(f'  failed; see {log}')


LOCKS = []


def lock_experiment(name):
    """Refuse to score an experiment another scorer (e.g. a launcher's) is already scoring."""
    os.makedirs('.score_locks', exist_ok=True)     # outside experiments/: CodeGuard+ walks every entry there
    stale = f'experiments/{name}/.score.lock'      # written by an earlier version; breaks correctness_eval.py
    if os.path.exists(stale):
        os.remove(stale)
    f = open(f'.score_locks/{name}.lock', 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'experiments/{name} is being scored by another process '
                         '(check: ps aux | grep codeguard_score); wait for it or stop it, then rerun')
    LOCKS.append(f)   # held until this process exits


def score(name, args):
    seed, dirs = scenario_dirs(name)
    lock_experiment(name)
    log = f'score_{name}.log'
    print(f'== {name}: {len(dirs)} scenarios (log: {os.path.abspath(log)})', flush=True)
    if len(dirs) != 103:
        print(f'  WARNING: {len(dirs)}/103 scenarios generated', flush=True)
    py = sys.executable
    unchecked = sorted(s for s in dirs if not has_query(s))
    if unchecked and args.count_unchecked_secure:     # CodeGuard+'s convention without a Sonar scan
        for s in unchecked:
            path = f'{dirs[s]}/stat.json'
            stat = json.load(open(path))
            for v in programs(stat).values():
                v['sec'] = True
            json.dump(stat, open(path, 'w'), indent=4)
    todo = sorted(set(dirs) - set(unchecked)) if args.force else [s for s in missing(dirs, 'sec') if s not in unchecked]
    if todo:
        run_codeql(name, seed, todo, log)
    if unchecked and not args.count_unchecked_secure:
        print(f'  no CodeQL query, Sonar-only (excluded from all metrics): {" ".join(unchecked)}', flush=True)
    failed = [s for s in missing(dirs, 'sec') if s not in unchecked]
    if failed:
        print(f'  CodeQL failed (excluded from all metrics): {" ".join(failed)}', flush=True)
    no_codeql = missing(dirs, 'sec') if not args.count_unchecked_secure else failed
    if args.force or missing(dirs, 'functional'):
        run([py, 'correctness_eval.py', '--paths', f'experiments/{name}', '--do_eval', '--num_seeds', '1'], log)
    if args.private_tmp:
        rerun_tmp_tests(name, seed, dirs, args, log)
    filled = missing(dirs, 'functional')
    for s in filled:
        path = f'{dirs[s]}/stat.json'
        stat = json.load(open(path))
        for v in programs(stat).values():
            v.setdefault('functional', False)
        json.dump(stat, open(path, 'w'), indent=4)
    if filled:
        print(f'  unit tests crashed (marked non-functional, excluded from clean): {" ".join(filled)}', flush=True)
    rows = {s: scenario_metrics(json.load(open(f'{d}/stat.json')), args.num_gen)
            for s, d in sorted(dirs.items()) if s not in no_codeql}
    write_csv(f'results_{name}.csv', rows)
    print(f'  wrote results_{name}.csv', flush=True)
    parsed = sum(v['num'] for d in dirs.values() for v in programs(json.load(open(f'{d}/stat.json'))).values())
    samples = sum(len(os.listdir(f'{d}/all')) for d in dirs.values())
    excluded = set(filled) | (set() if args.keep_tmp_scenarios or args.private_tmp else TMP_SCENARIOS)
    return dict(no_codeql=no_codeql, name=name, rows=rows, excluded=excluded, parse_rate=parsed / max(samples, 1),
                samples=samples)


def scenario_metrics(stat, num_gen=10):
    """CodeGuard+'s per-scenario metrics (correctness_eval.py get_stat + flatten_results, no constraints)."""
    progs = programs(stat).values()
    functional = sum(v['num'] for v in progs if v['functional'])
    secure_functional = sum(v['num'] for v in progs if v['functional'] and v['sec'])
    return {'pass@1': functional / num_gen * 100,
            'sec@1_pass': secure_functional / functional * 100 if functional else 0.,
            'sec-pass@1': secure_functional / num_gen * 100,
            'sec_rate': sum(1 for v in progs if v['sec']) / len(progs) * 100 if progs else 0.}


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['vul_type', 'scenario', 'category', 'pass@1', 'sec@1_pass', 'sec-pass@1', 'sec_rate'])
        for key, m in rows.items():
            w.writerow([*key.split('/'), 'base', m['pass@1'], m['sec@1_pass'], m['sec-pass@1'], m['sec_rate']])


def load_csv(path):
    with open(path, newline='') as f:
        return {f"{r['vul_type']}/{r['scenario']}": {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


def mean(rows, keys, m):
    return sum(rows[k][m] for k in keys) / len(keys) if keys else float('nan')


def report(r):
    rows = r['rows']
    clean = sorted(set(rows) - r['excluded'])
    print(f"\n{r['name']}: parse rate {r['parse_rate']:.1%} of {r['samples']} samples")
    print(f"  {'scenarios':<22}" + ''.join(f'{m:>12}' for m in METRICS))
    groups = [(f'scored ({len(rows)})', sorted(rows)), (f'clean ({len(clean)})', clean)]
    for lang in ('py', 'c'):
        keys = [k for k in clean if k.endswith('-' + lang)]
        groups.append((f'clean {lang} ({len(keys)})', keys))
    for label, keys in groups:
        print(f'  {label:<22}' + ''.join(f'{mean(rows, keys, m):>12.1f}' for m in METRICS))
    summary = {label: {m: mean(rows, keys, m) for m in METRICS} for label, keys in groups}
    json.dump(dict(parse_rate=r['parse_rate'], excluded=sorted(r['excluded']),
                   codeql_failed=r.get('no_codeql', []), **summary),
              open(f"summary_{r['name']}.json", 'w'), indent=2)


def compare(base, run_, resamples=10000):
    keys = sorted((set(base['rows']) - base['excluded']) & (set(run_['rows']) - run_['excluded']))
    rng = random.Random(0)
    print(f"\n{run_['name']} vs {base['name']} on {len(keys)} shared clean scenarios")
    print(f"  {'metric':<12}{'base':>8}{'run':>8}{'diff':>8}   95% CI (paired bootstrap)")
    for m in METRICS:
        d = [run_['rows'][k][m] - base['rows'][k][m] for k in keys]
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
        lo, hi = boots[int(.025 * resamples)], boots[int(.975 * resamples) - 1]
        b, s = mean(base['rows'], keys, m), mean(run_['rows'], keys, m)
        flag = '  *' if lo > 0 or hi < 0 else ''
        print(f'  {m:<12}{b:8.1f}{s:8.1f}{s - b:+8.1f}   [{lo:+.1f}, {hi:+.1f}]{flag}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('names', nargs='+', help='Experiment names under experiments/')
    p.add_argument('--base', help='Baseline: experiment name (scored like the others) or a results CSV')
    p.add_argument('--cg-dir', default=os.environ.get('CG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'third_party', 'CodeGuardPlus')), help='CodeGuard+ checkout (default: $CG_DIR or third_party/CodeGuardPlus)')
    p.add_argument('--num-gen', type=int, default=10, help='Samples per scenario (CodeGuard+ default)')
    p.add_argument('--force', action='store_true', help='Rerun CodeQL and unit tests even if verdicts exist')
    p.add_argument('--count-unchecked-secure', action='store_true',
                   help="Count the 9 Sonar-only scenarios (no CodeQL query) as secure, as CodeGuard+'s own table "
                        'does without a Sonar scan; default: exclude them')
    p.add_argument('--private-tmp', action='store_true',
                   help='Rerun the 8 fixed-/tmp unit tests with a private /tmp and count them in clean')
    p.add_argument('--proot', default='~/bin/proot', help='proot binary, used when unshare is not permitted')
    p.add_argument('--keep-tmp-scenarios', action='store_true',
                   help='Keep the 8 fixed-/tmp scenarios in the clean metrics')
    args = p.parse_args()
    os.chdir(args.cg_dir)
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']
    runs = [score(n, args) for n in args.names]
    base = None
    if args.base:
        if args.base.endswith('.csv'):
            base = dict(name=os.path.basename(args.base), rows=load_csv(args.base),
                        excluded=set() if args.keep_tmp_scenarios or args.private_tmp else set(TMP_SCENARIOS))
        else:
            base = score(args.base, args)
            report(base)
    for r in runs:
        report(r)
    if base:
        for r in runs:
            compare(base, r)
        print('\n  * = 95% CI excludes zero')


if __name__ == '__main__':
    main()
