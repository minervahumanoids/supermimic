"""Test a frozen artifact in native physics for diagnosis only."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import probe
from mjlab.envs import ManagerBasedRlEnv
from supermimic.representations import create_representation, extract_actor_goal
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.scripts.train import TrainConfig
from supermimic.tasks.task_spec import build_actor_obs_td

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def main(version):
    torch.set_num_threads(4)
    torch.manual_seed(69)
    np.random.seed(69)
    path = probe.checkpoint(version)
    runner = load_runner_cfg(json.loads((WORK / 'stopped-by-user-20261112/launch-config.json').read_text()))
    cfg = TrainConfig.from_task('SuperMimic-GenCoord-G1').env
    cfg.scene.num_envs = 1
    cfg.seed = 69
    cfg.episode_length_s = 1e9
    cfg.events = {}
    cfg.terminations = {}
    cfg.auto_reset = False
    cfg.observations['actor'].enable_corruption = False
    command_cfg = cfg.commands['motion']
    command_cfg.motion_file = str(WORK / 'evaluation-403000/inputs/bones/robot/00.npz')
    command_cfg.pose_range = {}
    command_cfg.velocity_range = {}
    command_cfg.sampling_mode = 'start'
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    command = env.command_manager.get_term('motion')
    robot = env.scene['robot']
    assert np.allclose(env.scene.env_origins.cpu().numpy(), 0)
    rep = create_representation(runner.representation, 71, 29, env.step_dt,
        cfg.decimation, robot.data.default_joint_pos[0], runner.actor_obs)
    actor = build_actor(runner, SimpleNamespace(representation=rep,
        num_goals=670, num_actions=29), 'cuda:0')
    actor.load_state_dict(torch.load(path, map_location='cuda:0', weights_only=True), strict=True)
    actor.eval().requires_grad_(False)
    env.reset()
    command.reset_to_frame(torch.tensor([0], device=env.device), 0)
    env.sim.forward()
    command.update_relative_body_poses()
    env.observation_manager.compute(update_history=True)
    roots, qd, actions = [], [], []
    for frame in range(1001):
        obs = env.get_observations()
        state = obs['full_state']
        roots.append(state[0, :3].cpu().numpy().copy())
        qd.append(robot.data.joint_vel[0].cpu().numpy().copy())
        if frame == 1000:
            break
        with torch.inference_mode():
            action = actor(build_actor_obs_td({'motion': extract_actor_goal(obs['goal'])},
                rep.represent(state)[:, None]))
        actions.append(action[0].cpu().numpy().copy())
        env.step(action)
    roots, qd, actions = map(np.asarray, (roots, qd, actions))
    reference = np.load(command_cfg.motion_file)['root_pos_w']
    error = np.linalg.norm(roots[:, :2] - reference[:, :2], axis=1)*100
    result = {'diagnostic_only': True, 'backend': 'native MuJoCo-Warp',
        'iteration': version*1000, 'actor_sha256': probe.sha(path),
        'root_xy_mean_cm': float(error.mean()), 'root_xy_final_cm': float(error[-1]),
        'root_xy_max_cm': float(error.max()),
        'joint_velocity_rms_rad_s': float(np.sqrt(np.mean(qd**2))),
        'action_change_rms': float(np.sqrt(np.mean(np.diff(actions, axis=0)**2))),
        'optimizer_steps': 0, 'official_source_modified': False, 'architecture_modified': False}
    probe.write(ROOT / f'native-v{version}.json', result)
    np.savez_compressed(ROOT / 'cases' / f'v{version}_native.npz', root=roots,
        joint_vel=qd, actions=actions)
    print(json.dumps(result), flush=True)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True)
    main(parser.parse_args().version)
