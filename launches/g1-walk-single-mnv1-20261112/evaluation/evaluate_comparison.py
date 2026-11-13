"""Evaluate the two frozen G1 actors on the exact requested native motion."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor

ROOT = Path(__file__).resolve().parent
NEW = ROOT.parent
OLD = Path('/home/mattia/workspace/supermimic-official-g1-mnv1-20261111')
CLIP = 'clip_6659630b5d665786'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    torch.set_num_threads(4)
    assert mujoco.__version__ == '3.15.0'
    assert torch.__version__.split('+')[0] == '2.15.0'
    parent = json.loads((ROOT / 'parent-selection.json').read_text())
    clip = next(row for row in parent['clips'] if row['clip'] == CLIP)
    clip['original_human'] = str(ROOT / 'walk-human.npz')
    assert sha(ROOT / 'walk-human.npz') == clip['original_human_sha256']
    clip['training_membership'] = 'Training member of both G1 selections. The 15k run trains on this clip only.'
    inputs = ROOT / 'inputs/bones'
    (inputs / 'robot/clips').mkdir(parents=True, exist_ok=False)
    motion_file = inputs / f'robot/clips/{CLIP}.npz'
    shutil.copyfile(NEW / f'data/{CLIP}.npz', motion_file)
    assert sha(motion_file) == '6334b0fcc56c86427ad079572c50bac9f9214b3b5179fff2f66ff24610fc70b2'
    write(inputs / 'selection.json', dict(authoritative_selection_sha256=sha(ROOT / 'parent-selection.json'), clips=[clip], user_requested_subset=True))
    write(inputs / 'robot/index.json', dict(clips=[dict(clip=CLIP, path=f'clips/{CLIP}.npz', sha256=sha(motion_file))]))
    source_plant = OLD / 'evaluation-403000/plant'
    (ROOT / 'plant').symlink_to(source_plant, target_is_directory=True)
    shutil.copyfile(OLD / 'evaluation-403000/plant-contract.json', ROOT / 'plant-contract.json')
    contract = json.loads((ROOT / 'plant-contract.json').read_text())
    assert sha(ROOT / 'plant/scene.xml') == contract['model_sha256']
    with np.load(motion_file, allow_pickle=False) as data:
        assert data['joint_pos'].shape == (214, 29)
        np.savez_compressed(ROOT / 'reference.npz', joint_pos=data['joint_pos'], root_pos=data['root_pos_w'], root_quat_wxyz=data['root_quat_w'])
    policies = {
        'g1-single15': dict(label='G1 single clip · 15,000', run_id='45u76a15', update=15000, checkpoint=NEW / 'stopped-at-15000-20261112', config=NEW / 'launch-config.json', expected_sha='7db8a31f0075a4eef5fe5b5594cbf777389a60b0706d9bc899fb1006ced6d83f'),
        'g1-full25': dict(label='G1 full dataset · 25,000', run_id='ga216ipt', update=25000, checkpoint=OLD / 'standing-cause-20261112/checkpoints/v25', config=OLD / 'launch-config.json', expected_sha='a0ce1eb1614c6380dc304a0781dc0dc16c9f79e9b05b33d1b0e3ccd06a3c013c'),
    }
    results = {}
    for lane, policy in policies.items():
        checkpoint = policy['checkpoint']
        assert sha(checkpoint / 'actor.pt') == policy['expected_sha']
        wm = torch.load(checkpoint / 'world_model.pt', map_location='cpu', weights_only=True)['dynamics']
        for key, values in (('action_scale', contract['action_scale']), ('action_offset', contract['default_joint_pos'])):
            torch.testing.assert_close(wm[key].reshape(-1), torch.tensor(values), atol=1e-6, rtol=0)
        cfg = load_runner_cfg(json.loads(policy['config'].read_text()))
        representation = create_representation('gencoord', 71, 29, .02, contract['decimation'], torch.tensor(contract['default_joint_pos']), actor_obs_mode=cfg.actor_obs)
        actor = build_actor(cfg, SimpleNamespace(representation=representation, num_goals=670, num_actions=29), 'cpu')
        actor.load_state_dict(torch.load(checkpoint / 'actor.pt', map_location='cpu', weights_only=True), strict=True)
        actor.eval().requires_grad_(False)
        output = ROOT / lane
        output.mkdir()
        (output / 'inputs').symlink_to(ROOT / 'inputs', target_is_directory=True)
        (output / 'plant').symlink_to(ROOT / 'plant', target_is_directory=True)
        shutil.copyfile(ROOT / 'plant-contract.json', output / 'plant-contract.json')
        spec = importlib.util.spec_from_file_location('approved_g1_' + lane.replace('-', '_'), ROOT / 'approved_evaluate_g1.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = output
        module.ACTOR_SHA = policy['expected_sha']
        report = module.evaluate('bones', actor, representation, contract)
        report.update(iteration=policy['update'], run_id=policy['run_id'], mujoco_version=mujoco.__version__, torch_version=torch.__version__, checkpoint_artifact=f'minervahumanoids/Supermimic/model-{policy["run_id"]}:v{policy["update"]//1000}', same_requested_clip=True, evaluator_source_sha256=sha(ROOT / 'approved_evaluate_g1.py'), metadata_adapter_only=True, policy_modified=False, wandb_initialized=False)
        write(output / 'recordings/bones/report.json', report)
        results[lane] = dict(label=policy['label'], iteration=policy['update'], run_id=policy['run_id'], actor_sha256=policy['expected_sha'], report=str(output / 'recordings/bones/report.json'), report_sha256=sha(output / 'recordings/bones/report.json'), trial=report['trials'][0])
    write(ROOT / 'comparison-results.json', dict(backend='standalone CPU MuJoCo', mujoco_version=mujoco.__version__, torch_version=torch.__version__, clip=CLIP, frames=214, sample_rate_hz=50, source_motion_sha256=sha(motion_file), model_sha256=contract['model_sha256'], physical_contract_sha256=sha(ROOT / 'plant-contract.json'), optimizer_steps=0, policies=results))
    print(json.dumps(results, default=str), flush=True)


if __name__ == '__main__':
    main()
