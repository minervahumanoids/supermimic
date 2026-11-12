"""Bind the G1 recordings to the approved comparison viewer contract."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


contract=json.loads((ROOT/'plant-contract.json').read_text())
manifest={'schema':'frozen-policy-review-v1','robot':'Unitree G1',
    'model_path':str(ROOT/'plant/scene.xml'),'model_sha256':contract['model_sha256'],
    'available_encoder_routes':['robot'],'encoder_route_labels':{'robot':'Motion encoder'},
    'recording_route_labels':{'robot':'motion encoder'},'camera_vertical_view_offset':0.4,
    'palm_sites':contract['palm_sites'],'dataset_label':'BONES-SEED G1','asset_label':'official G1',
    'policies':{'G1':{'actor_sha256':'07e064bd4b6500b97a4a6b4af0060a801588abb8f0abdceb38138e2c40c0144b',
        'source_commit':'abc124c1bd5976341034668af3d12c087f2f1f55','update':403000,'routes':['robot'],
        'wandb_url':'https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/runs/ga216ipt',
        'checkpoint_artifact':'minervahumanoids/Supermimic/model-ga216ipt:v403'}},
    'display':{
        'description':'### Official G1 · iteration 403,000\n\nStandalone CPU MuJoCo. '
            'The original human input, G1 reference, and recorded policy share the saved 50 Hz source clock.\n\n'
            'The review retains all 30 approved BONES clips and all three PICO sessions.',
        'provenance':'Training stopped at collector iteration 403,091. The evaluated checkpoint is iteration 403,000.\n\n'
            'The motion encoder uses the official current-plus-future reference window. '
            'The velocity encoder has no training exposure. A SMPL encoder is unavailable.\n\n'
            'BONES references use the original G1 release. The reviewed rise clip is excluded from training. '
            'PICO references use the pinned MDP GMR backend on the original SMPL poses. '
            'PICO references include one constant floor correction per session. '
            'The original human input remains unchanged.\n\n'
            'Policy A uses Roger and cannot execute on G1. A matched G1 baseline is unavailable.\n\n'
            'Palm errors use the official G1 palm sites relative to the pelvis. '
            'Root drift uses instantaneous world XY distance from the G1 reference. '
            'Curves end at recorded boundaries.',
        'lanes':{'g1':{'policy':'G1','label':'Official G1 · 403,000','short':'G1 403k',
                        'route':'robot','color':'#356aa0','dash':''}}},
    'recordings':{}}
for cohort in ('bones','pico'):
    report_path=ROOT/'recordings'/cohort/'report.json'
    report=json.loads(report_path.read_text())
    manifest['recordings'][cohort]={}
    for trial in report['trials']:
        path=report_path.parent/trial['file']
        record=dict(trial,file=str(path.relative_to(ROOT)),sha256=sha(path),
                    report_file=str(report_path.relative_to(ROOT)),report_sha256=sha(report_path))
        manifest['recordings'][cohort][trial['clip']]={'G1':{'robot':record}}
(ROOT/'review-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'manifest':str(ROOT/'review-manifest.json'),'bones':30,'pico':3}))
