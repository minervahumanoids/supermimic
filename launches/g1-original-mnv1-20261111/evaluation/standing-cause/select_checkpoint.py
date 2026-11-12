"""Select an unchanged artifact after the full cohort and standing checks.

This external gate controls checkpoint selection. It does not start or stop
training, change a training budget, or change the official learner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g1-stop-review-20261112'
SELECTED_SHA = 'a0ce1eb1614c6380dc304a0781dc0dc16c9f79e9b05b33d1b0e3ccd06a3c013c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(original_review=ORIGINAL):
    checks = []
    def check(name, actual, passed):
        checks.append({'name': name, 'actual': actual, 'passed': bool(passed)})
    review = ROOT / 'review-25000'
    summary = json.loads((review / 'evaluation-summary.json').read_text())
    check('original artifact', summary['actor_sha256'], summary['actor_sha256'] == SELECTED_SHA)
    check('zero learning steps', summary['optimizer_steps'], summary['optimizer_steps'] == 0)
    for cohort, count in (('bones', 30), ('pico', 3)):
        report = json.loads((review / 'recordings' / cohort / 'report.json').read_text())
        original = json.loads((original_review / 'recordings' / cohort / 'report.json').read_text())
        check(cohort + ' simulator', report['backend'], report['backend'] == 'standalone CPU MuJoCo')
        check(cohort + ' policy', report['policy_sha256'], report['policy_sha256'] == SELECTED_SHA)
        for field in ('model_sha256', 'plant_contract_sha256', 'input_selection_sha256', 'authoritative_selection_sha256'):
            check(cohort + ' ' + field, report[field], report[field] == original[field])
        actual = [(t['clip'], t['comparison_number'], t['display_number'], t['requested_frames']) for t in report['trials']]
        expected = [(t['clip'], t['comparison_number'], t['display_number'], t['requested_frames']) for t in original['trials']]
        check(cohort + ' full fixed cohort', len(actual), actual == expected and len(actual) == count)
        check(cohort + ' full intervals', sum(t['frames'] for t in report['trials']),
            all(t['complete'] and t['frames'] == t['requested_frames'] and t['stop'] is None for t in report['trials']))
    contract = json.loads((original_review / 'plant-contract.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(original_review / 'plant/scene.xml'))
    data = mujoco.MjData(model)
    ankles = [model.body('robot/' + side + '_ankle_roll_link').id for side in ('left', 'right')]
    reports = json.loads((ROOT / 'checkpoint-screen.json').read_text())['reports']
    standing = {}
    thresholds = {'root_xy_max_cm': 10., 'joint_velocity_rms_rad_s': .05,
        'action_change_rms': .02, 'ankle_xy_speed_rms_after_2s_cm_s': 1.}
    for label in ('v25_screen_nominal', 'v25_screen_friction_1'):
        report = next(r for r in reports if r['label'] == label)
        check(label + ' full interval', report['frames'], report['complete'] and report['frames'] == 1001)
        check(label + ' policy', report['actor_sha256'], report['actor_sha256'] == SELECTED_SHA)
        for field in ('root_xy_max_cm', 'joint_velocity_rms_rad_s', 'action_change_rms'):
            check(label + ' ' + field, report[field], report[field] <= thresholds[field])
        archive = np.load(ROOT / 'cases' / (label + '.npz'))
        points = []
        for root, quat, joints in zip(archive['root'], archive['quat'], archive['q']):
            data.qpos[:3] = root
            data.qpos[3:7] = quat
            data.qpos[contract['qpos_addresses']] = joints
            mujoco.mj_forward(model, data)
            points.append(data.xpos[ankles, :2].copy())
        delta = np.diff(np.asarray(points)[100:], axis=0)
        speed = np.linalg.norm(delta, axis=-1) * 5000
        rms = np.sqrt(np.mean(speed**2, axis=0))
        check(label + ' ankle XY speed', rms.tolist(), np.all(rms <= thresholds['ankle_xy_speed_rms_after_2s_cm_s']))
        standing[label] = {**report, 'ankle_xy_speed_rms_after_2s_cm_s': rms.tolist(),
            'ankle_xy_path_after_2s_cm': (np.linalg.norm(delta, axis=-1).sum(axis=0)*100).tolist(),
            'recording_sha256': sha(ROOT / 'cases' / (label + '.npz'))}
    native = json.loads((ROOT / 'native-v25.json').read_text())
    check('native diagnostic policy', native['actor_sha256'], native['actor_sha256'] == SELECTED_SHA)
    check('native standing root error', native['root_xy_max_cm'], native['root_xy_max_cm'] <= thresholds['root_xy_max_cm'])
    negative = json.loads((ROOT / 'probe-summary.json').read_text())['reports']
    negative = next(r for r in negative if r['label'] == 'v403_cpu')
    rejected = {field: negative[field] > thresholds[field] for field in
        ('root_xy_max_cm', 'joint_velocity_rms_rad_s', 'action_change_rms')}
    check('regressed final checkpoint rejected', rejected, all(rejected.values()))
    passed = all(c['passed'] for c in checks)
    result = {'passed': passed, 'scope': 'Frozen checkpoint promotion only. No training-stop or budget changes.',
        'selected_artifact': summary['checkpoint_artifact'] if passed else None,
        'selected_iteration': 25000 if passed else None, 'actor_sha256': SELECTED_SHA,
        'thresholds': thresholds, 'checks': checks, 'standing': standing,
        'foot_definition': 'Ankle-frame XY movement after two seconds on the saved 50 Hz clock. This is not a contact-slip estimate.',
        'report_sha256': {cohort: sha(review / 'recordings' / cohort / 'report.json') for cohort in ('bones', 'pico')},
        'latest_artifact_preserved': 'minervahumanoids/Supermimic/model-ga216ipt:v403',
        'official_source_modified': False, 'architecture_modified': False, 'optimizer_steps': 0,
        'new_training_started': False,
        'limitations': ['The standing gate does not establish gait quality on every motion.',
            'The selected checkpoint has higher aggregate palm-orientation error than checkpoint 403000.',
            'The tests identify a learned-policy regression. They do not establish which training choice caused it.']}
    (ROOT / 'checkpoint-selection.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': passed, 'checks': len(checks), 'selected_artifact': result['selected_artifact'], 'standing': standing}))
    if not passed:
        raise SystemExit('Checkpoint promotion rejected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-review', type=Path, default=ORIGINAL)
    main(parser.parse_args().original_review)
