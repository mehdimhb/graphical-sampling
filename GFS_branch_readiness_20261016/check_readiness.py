"""Read-only Git assessment and clean committed-tree package tests.

No branch switch, merge, push, original source modification or research run.
The existing notebook test runs its reduced validation settings in /tmp.
"""
import collections
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    report_path = HERE/'assessment.json'
    if report_path.exists():
        raise FileExistsError('Preserving completed assessment; use a new evidence directory')
    head = git('rev-parse', 'HEAD')
    main = git('rev-parse', 'origin/main')
    refs = []
    for row in git('for-each-ref', '--sort=-committerdate',
                   '--format=%(refname)|%(objectname)|%(committerdate:iso-strict)',
                   'refs/heads', 'refs/remotes/origin').splitlines():
        ref, sha, date = row.split('|')
        if ref == 'refs/remotes/origin/HEAD':
            continue
        ahead, missing = map(int, git('rev-list', '--left-right', '--count', f'{head}...{sha}').split())
        refs.append(dict(ref=ref, sha=sha, committer_date=date,
                         current_unique_commits=ahead, branch_unique_commits=missing,
                         wholly_contained_in_current=missing == 0))
    statuses = git('diff', '--name-status', f'{main}...{head}').splitlines()
    snapshot = Path(tempfile.mkdtemp(prefix='gfs-main-readiness-'))/'snapshot'
    snapshot.mkdir()
    # Export committed inputs only, including the otherwise locally deleted notebook.
    export = subprocess.Popen(['git', 'archive', head, 'graphical_sampling', 'tests',
        'prescribed_sip', 'simulations/jupyters/17_graphical_maxent_die.ipynb',
        'pyproject.toml'], cwd=ROOT, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=export.stdout, mode='r|') as archive:
        archive.extractall(snapshot, filter='data')
    assert export.wait() == 0
    env = os.environ.copy()
    env.update({key:'1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                   'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})
    env['MPLCONFIGDIR'] = str(snapshot.parent/'matplotlib')
    env['PYTHONPATH'] = str(snapshot)
    python = str(ROOT/'.venv/bin/python')
    imported = subprocess.check_output([python, '-c',
        'import graphical_sampling; print(graphical_sampling.__file__)'],
        cwd=snapshot, env=env, text=True).strip()
    assert Path(imported).is_relative_to(snapshot), imported
    assessment = dict(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        current_branch=git('branch', '--show-current'), current_SHA=head,
        remote_current_SHA=git('rev-parse', 'origin/codex/gfs-numerical-corrections-20261016'),
        main_SHA=main, main_current_divergence=git('rev-list', '--left-right', '--count', f'{main}...{head}'),
        ahead_commits=git('log', '--format=%h %s', f'{main}..{head}').splitlines(),
        modified_paths=len(statuses), statuses=dict(collections.Counter(r.split('\t')[0] for r in statuses)),
        path_groups=dict(collections.Counter(r.split('\t')[-1].split('/')[0] for r in statuses)),
        references=refs, snapshot=str(snapshot), imported_package=imported,
        test_command=[python, '-m', 'pytest', 'tests', '-q'],
        test_status='running', working_tree_status=git('status', '--porcelain=v1'),
        note='Committed snapshot excludes local untracked tests/studies. Snapshot research suites are not run. Package tests include the existing reduced MaxEnt notebook contract; no original Monte Carlo or DI480 evaluation is run.')
    report_path.write_text(json.dumps(assessment, indent=2)+'\n')
    with (HERE/'clean_committed_tests.log').open('w') as output:
        result = subprocess.run(assessment['test_command'], cwd=snapshot, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    assessment.update(test_status='complete', test_exit_code=result.returncode,
                      remote_main_reference_unchanged=git('rev-parse', 'origin/main') == main,
                      local_main_SHA=git('rev-parse', 'main'))
    report_path.write_text(json.dumps(assessment, indent=2)+'\n')
    print(json.dumps({key:assessment[key] for key in ('current_SHA','main_SHA',
          'main_current_divergence','modified_paths','snapshot','test_exit_code')}, indent=2))
    print('\n'.join((HERE/'clean_committed_tests.log').read_text().splitlines()[-12:]))


if __name__ == '__main__':
    main()
