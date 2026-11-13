"""Bind verified G1 recordings to the canonical Viser comparison profile."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT.parent / 'ali-baseline-prototype-20261112'
sys.path.insert(0, str(CANONICAL))
from tracking_metrics import Landmarks, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    results = json.loads((ROOT / 'comparison-results.json').read_text())
    assert results['mujoco_version'] == '3.15.0' and results['torch_version'] == '2.15.0+cpu'
    contract = json.loads((ROOT / 'plant-contract.json').read_text())
    model_path = ROOT.parent / 'g1-stop-review-20261112/plant/scene.xml'
    assert sha(model_path) == results['model_sha256']
    selection = json.loads((ROOT / 'inputs/bones/selection.json').read_text())
    clip = selection['clips'][0]
    clip['original_human'] = str(ROOT / 'walk-human.npz')
    clip['human'] = clip['original_human']
    clip['label'] = 'Walking · clip_6659630b5d665786'
    assert sha(clip['original_human']) == clip['original_human_sha256']
    with np.load(clip['original_human'], allow_pickle=False) as human:
        times = human['source_time'] if 'source_time' in human else human['times']
        np.testing.assert_allclose(times, np.arange(214) / 50, rtol=0, atol=1e-6)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    landmarks = dict(wrists=['robot/left_wrist_yaw_link', 'robot/right_wrist_yaw_link'], feet=None)
    measure = Landmarks(model, contract['model_joint_names'], landmarks)
    reference = dict(np.load(ROOT / 'reference.npz', allow_pickle=False))
    binding = dict(clip=clip['clip'], frames=214, reference=str(ROOT / 'reference.npz'), reference_sha256=sha(ROOT / 'reference.npz'), policies={})
    policies, summary_policies = {}, {}
    for lane, source in results['policies'].items():
        report_path = ROOT / lane / 'recordings/bones/report.json'
        assert sha(report_path) == source['report_sha256']
        report = json.loads(report_path.read_text())
        trial = report['trials'][0]
        record_path = report_path.parent / trial['file']
        record = dict(np.load(record_path, allow_pickle=False))
        assert len(record['q']) == 214
        values = measure.measure(reference, record)
        metrics_path = ROOT / f'metrics-{lane}.npz'
        np.savez_compressed(metrics_path, **values)
        score = summarize(values, trial, 214)
        policy = dict(label=source['label'], report=str(report_path), report_sha256=sha(report_path), actor_sha256=source['actor_sha256'], update=source['iteration'], run_id=source['run_id'])
        policies[lane] = policy
        binding['policies'][lane] = dict(recording=str(record_path), recording_sha256=sha(record_path), metrics=str(metrics_path), metrics_sha256=sha(metrics_path), metadata=trial)
        summary_policies[lane] = dict(label=source['label'], completed=int(trial['complete']), guard_stops=int(trial['stop'] is not None), recorded_frames=trial['frames'], requested_frames=214, common_prefix_macro_mean=score['mean'], standing_root_xy_mean_cm=None)
    summary = dict(clips=1, cohort='Requested G1 walking clip', policies=summary_policies)
    summary_path = ROOT / 'matched-summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    profile = dict(robot='G1', description='### G1 walking comparison\n\nSingle-clip checkpoint 15,000 against full-dataset checkpoint 25,000. Both use standalone MuJoCo 3.15, the same G1 plant, and all 214 source frames.',
        cohort_label='Requested G1 walking clip · 1', encoder_description='**Encoder:** Native G1 motion encoder for both policies. The official current-plus-future reference window remains unchanged. The original SOMA input stays synchronized.',
        plant=dict(model_path=str(model_path), model_sha256=sha(model_path), joint_names=contract['model_joint_names'], pelvis_body='robot/pelvis'),
        selection=selection, catalog=dict(clips=copy.deepcopy(selection['clips'])), bindings=dict(policies=policies, bindings=[binding]), registry=dict(policies=copy.deepcopy(policies)),
        styles={'g1-single15':['G1 single clip · 15,000', '#356aa0', ''], 'g1-full25':['G1 full dataset · 25,000', '#bd3c2f', '']},
        landmarks=landmarks, summary_path=str(summary_path), summary_sha256=sha(summary_path))
    path = ROOT / 'viewer-manifest.json'
    path.write_text(json.dumps(profile, indent=2) + '\n')
    print(json.dumps({'manifest': str(path), 'results': summary_policies}), flush=True)


if __name__ == '__main__':
    main()
