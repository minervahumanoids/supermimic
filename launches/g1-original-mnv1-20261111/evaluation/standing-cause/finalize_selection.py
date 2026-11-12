"""Record a validated selection without resuming or rewriting the stopped run."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.request

import wandb

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
API = 'https://orion1.taile1c92b.ts.net/api/availability'
RUN = 'minervahumanoids/Supermimic/ga216ipt'


def flags():
    with urllib.request.urlopen(API, timeout=20) as response:
        return json.load(response)['flags']


def main():
    state = subprocess.check_output(['systemctl', '--user', 'show',
        'supermimic-g1-original-train-20261111.service', '-p', 'ActiveState',
        '-p', 'MainPID', '-p', 'Restart'], text=True)
    assert all(x in state for x in ('ActiveState=inactive', 'MainPID=0', 'Restart=no'))
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    assert not gpu, 'Preserve the reservation while compute processes remain.'
    sources = {}
    for name, expected in (('supermimic', 'abc124c1bd5976341034668af3d12c087f2f1f55'),
                           ('mjlab_research', 'dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd')):
        repo = WORK / name
        head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        changes = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
        assert head == expected and not changes
        sources[name] = {'commit': head, 'tracked_changes': changes}
    selection = json.loads((ROOT / 'checkpoint-selection.json').read_text())
    assert selection['passed'] and selection['selected_iteration'] == 25000
    summary = json.loads((ROOT / 'review-25000/evaluation-summary.json').read_text())
    assert summary['actor_sha256'] == selection['actor_sha256']
    import hashlib
    actual_sha = hashlib.sha256((ROOT / 'checkpoints/v25/actor.pt').read_bytes()).hexdigest()
    assert actual_sha == selection['actor_sha256']
    evidence = {'checkpoint_artifact': summary['checkpoint_artifact'], 'iteration': 25000,
        'actor_sha256': actual_sha, 'backend': summary['backend'],
        'full_fixed_cohort': '30 BONES clips and three PICO sessions',
        'reason': 'Checkpoint 403000 has a standing regression. Select the unchanged earlier artifact after matched evaluation.',
        'previous_evaluation_preserved': 'model-ga216ipt:v403 at iteration 403000',
        'source_verified_unchanged': sources, 'architecture_modified': False,
        'pull_request': 'https://github.com/minervahumanoids/supermimic/pull/1',
        'gate_file': 'launches/g1-original-mnv1-20261111/evaluation/standing-cause/checkpoint-selection.json',
        'training_started': False,
        'limitations': 'Higher aggregate palm-orientation error than checkpoint 403000. The precise training cause of the regression remains unproven.'}
    run = wandb.Api(timeout=60).run(RUN)
    assert run.state == 'killed'
    launch_commit = run.config['launch_commit']
    previous_evaluation = run.config['stopped_evaluation']
    run.config['standing_checkpoint_selection'] = evidence
    run.update()
    run.summary['selected_checkpoint_evaluation'] = summary
    run.summary.update()
    verified = wandb.Api(timeout=60).run(RUN)
    assert verified.state == 'killed'
    assert verified.config['launch_commit'] == launch_commit
    assert verified.config['stopped_evaluation'] == previous_evaluation
    assert verified.config['standing_checkpoint_selection'] == evidence
    assert verified.summary['selected_checkpoint_evaluation.iteration'] == 25000
    before = flags().get('mnv1')
    expected = json.loads((ROOT / 'orion-reservation.json').read_text())
    if before is not None:
        assert before['version'] == expected['version']
        assert before['owner'] == 'Mattia' and before['kind'] == 'reservation'
        assert before['note'] == 'PR1 G1 cause: standing'
        request = urllib.request.Request(API, data=json.dumps({'machine': 'mnv1',
            'operation': 'clear', 'expectedVersion': before['version']}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    after = flags().get('mnv1')
    assert after is None
    receipt = {'verified_utc': datetime.now(timezone.utc).isoformat(),
        'service_state': state, 'gpu_compute_processes': gpu,
        'sources': sources, 'selected_checkpoint': evidence,
        'wandb_url': verified.url, 'wandb_state': verified.state,
        'launch_commit_preserved': launch_commit, 'original_evaluation_preserved': True,
        'orion_before': before, 'orion_after': after,
        'new_training_started': False, 'unrelated_jobs_modified': False}
    (ROOT / 'completion-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
