"""Preserve the stopped official G1 checkpoint and verify its server provenance."""

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

import wandb

ROOT = Path('/home/mattia/workspace/supermimic-official-g1-mnv1-20261111')
OUT = ROOT / 'stopped-by-user-20261112'
UNIT = 'supermimic-g1-original-train-20261111.service'


def hashes(path):
    return {'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
            'md5': base64.b64encode(hashlib.file_digest(path.open('rb'), 'md5').digest()).decode(),
            'bytes': path.stat().st_size}


state = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
    '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Restart', '-p', 'ExecMainStatus'], text=True)
assert 'ActiveState=inactive' in state and 'MainPID=0' in state and 'Restart=no' in state
OUT.mkdir(exist_ok=True)
api = wandb.Api(timeout=60)
run = api.run('minervahumanoids/Supermimic/ga216ipt')
assert run.state == 'killed'
artifact = api.artifact('minervahumanoids/Supermimic/model-ga216ipt:v403', type='model')
assert artifact.logged_by().id == run.id
destination = OUT / 'checkpoint-v403'
destination.mkdir(exist_ok=True)
files = {}
for entry in artifact.files():
    assert entry.name in ('actor.pt', 'world_model.pt', 'critic.pt')
    path = destination / entry.name
    if not path.exists() or path.stat().st_size != entry.size:
        artifact.get_path(entry.name).download(root=str(destination))
    value = hashes(path)
    assert value['md5'] == entry.digest and value['bytes'] == entry.size
    assert zipfile.is_zipfile(path)
    files[entry.name] = value
assert set(files) == {'actor.pt', 'world_model.pt', 'critic.pt'}
last = int(run.summary['Train/iter'])
assert 403001 <= last < 404001
warnings = subprocess.run(['rg', '-c', 'Checkpoint artifact log/prune failed', str(ROOT/'train.log')],
                          text=True, capture_output=True)
assert warnings.returncode == 1 and not warnings.stdout.strip()
local = ROOT/'supermimic/logs/SuperMimic-GenCoord-G1/2026-11-11_21-31-37'
raw = OUT/'local-checkpoint'
raw.mkdir(exist_ok=True)
for name, value in files.items():
    shutil.copy2(local/name, raw/name)
    assert hashes(raw/name) == value
for name in ('launch-config.json', 'telemetry-ready.json', 'runtime-freeze.txt'):
    shutil.copy2(ROOT/name, OUT/name)
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                              '--format=csv,noheader'], text=True).strip()
receipt = {'user_requested_stop':True, 'host':'mnv1', 'unit':UNIT, 'service_state':state,
    'gpu_processes_after':gpu, 'run_id':run.id, 'wandb_url':run.url, 'wandb_state':run.state,
    'checkpoint_artifact':artifact.qualified_name, 'checkpoint_version':artifact.version,
    'checkpoint_digest':artifact.digest, 'checkpoint_iteration':403000,
    'checkpoint_iteration_basis':'Version v403, one launch, native save interval 1000, no checkpoint upload errors, final iteration bound',
    'last_collector_iteration':last, 'checkpoint_files':files,
    'checkpoint_directory':str(destination), 'optimizer_state_available':False,
    'source_commit':run.config['source_commit'], 'launch_commit':run.config['launch_commit'],
    'pull_request':run.config['pull_request'], 'unrelated_jobs_modified':False,
    'verified_utc':datetime.now(timezone.utc).isoformat()}
(OUT/'stop-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
run.summary['user_requested_stop'] = True
run.summary['stopped_collector_iteration'] = last
run.summary['evaluation_checkpoint_iteration'] = 403000
run.summary.update()
api_url = 'https://orion1.taile1c92b.ts.net/api/availability'
with urllib.request.urlopen(api_url,timeout=20) as response:
    flags=json.load(response)['flags']
flag=flags.get('mnv1')
if flag:
    assert flag['owner']=='Mattia' and flag['kind']=='reservation' and flag['wandb_url']==run.url
    payload={'machine':'mnv1','operation':'save','owner':'Mattia','kind':'reservation',
             'duration':240,'note':'PR1 G1 eval: checkpoint 403000','wandb_url':run.url,
             'expectedVersion':flag['version']}
    request=urllib.request.Request(api_url,data=json.dumps(payload).encode(),
                                  headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request,timeout=20) as response:
        json.load(response)
print(json.dumps(receipt),flush=True)
