"""Compare frozen imagined transitions with physical transitions. No learning."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import probe
from mjlab.utils.forward_kinematics import ForwardKinematics
from supermimic.models import Dynamics
from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.scripts.train import TrainConfig
from supermimic.tasks.task_spec import build_actor_obs_td

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REVIEW = WORK / 'evaluation-403000'


def main():
    torch.set_num_threads(4)
    probe.configure_torch_backends(allow_tf32=False)
    cfg = load_runner_cfg(json.loads((WORK / 'stopped-by-user-20261112/launch-config.json').read_text()))
    contract = json.loads((REVIEW / 'plant-contract.json').read_text())
    task = TrainConfig.from_task('SuperMimic-GenCoord-G1').env
    model = mujoco.MjModel.from_xml_path(str(REVIEW / 'plant/scene.xml'))
    probe.MujocoCfg(**contract['physics']).apply(model)
    qa, va, ai = (np.array(contract[key]) for key in ('qpos_addresses', 'dof_addresses', 'actuator_ids'))
    robot_ids = [i for i in range(model.nbody) if model.body(i).name.startswith('robot/')]
    body_ids = [model.body('robot/' + name).id for name in task.commands['motion'].body_names]
    fk = ForwardKinematics(model, torch.tensor(robot_ids), torch.tensor(qa),
        torch.tensor([robot_ids.index(i) for i in body_ids]))
    rep = create_representation('gencoord', 71, 29, .02, 4,
        torch.tensor(contract['default_joint_pos']), cfg.actor_obs)
    rep.fk = fk
    limits = model.jnt_range[[model.joint(name).id for name in contract['model_joint_names']]]
    midpoint, half = limits.mean(axis=1), np.diff(limits, axis=1).ravel()*.45
    rep._soft_joint_pos_limits = torch.tensor(np.stack([midpoint-half, midpoint+half], axis=-1), dtype=torch.float32)
    original = dict(np.load(REVIEW / 'inputs/bones/robot/00.npz'))
    motion = SimpleNamespace(**{key: torch.tensor(original[key]) for key in
        ('root_pos_w', 'root_quat_w', 'joint_pos', 'root_lin_vel_w', 'root_ang_vel_w', 'joint_vel')},
        motion_ends=torch.tensor([1001]))
    targets = fk(motion.root_pos_w, motion.root_quat_w, motion.joint_pos,
        motion.root_lin_vel_w, motion.root_ang_vel_w, motion.joint_vel)
    root_id = model.body('robot/pelvis').id
    def state(data):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root_id, vel, 0)
        return torch.tensor(np.r_[data.xpos[root_id], data.xquat[root_id], data.qpos[qa],
            vel[3:], vel[:3], data.qvel[va]], dtype=torch.float32)[None]
    def reward(full, action, previous, frame):
        r = rep.represent(full)
        return rep.compute_wm_reward_from_reference(r, full[:, :7], rep.joint_state(r),
            action, previous, *[x[frame:frame+1] for x in targets])
    actors, dynamics, contexts = {}, {}, {}
    for version in (25, 403):
        path = probe.checkpoint(version)
        folder = path.parent
        if not (folder / 'world_model.pt').exists():
            probe.wandb.Api(timeout=60).artifact(f'minervahumanoids/Supermimic/model-ga216ipt:v{version}').get_path('world_model.pt').download(root=str(folder))
        bundle = torch.load(folder / 'world_model.pt', weights_only=True, map_location='cpu')
        actor = build_actor(cfg, SimpleNamespace(representation=rep, num_goals=670, num_actions=29), 'cpu')
        actor.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
        actors[version] = actor.eval().requires_grad_(False)
        dim = len(bundle['sim_context']['mean'])
        dynamics[version] = Dynamics(rep, 29, dim, bundle['dynamics']['action_scale'],
            bundle['dynamics']['action_offset'], cfg.dynamics_hidden_dims)
        dynamics[version].load_state_dict(bundle['dynamics'], strict=True)
        dynamics[version].eval().requires_grad_(False)
        com_name = task.events['base_com'].params['asset_cfg'].body_names[0]
        raw = torch.tensor(np.r_[.6, model.body_ipos[model.body('robot/' + com_name).id], np.zeros(29)], dtype=torch.float32)
        contexts[version] = ((raw-bundle['sim_context']['mean'])/bundle['sim_context']['scale'])[None]
    reports = []
    with torch.inference_mode():
        for version, actor in actors.items():
            for start in (0, 100, 500):
                data = mujoco.MjData(model)
                data.qpos[:3] = original['root_pos_w'][start]
                data.qpos[3:7] = original['root_quat_w'][start]
                data.qpos[qa] = np.clip(original['joint_pos'][start], midpoint-half, midpoint+half)
                data.qvel[:3] = original['root_lin_vel_w'][start]
                data.qvel[3:6] = Rotation.from_quat(data.qpos[3:7], scalar_first=True).inv().apply(original['root_ang_vel_w'][start])
                data.qvel[va] = original['joint_vel'][start]
                mujoco.mj_forward(model, data)
                initial = state(data)
                imagined = initial.clone()
                last_real = torch.zeros((1, 29))
                last_imagined = last_real.clone()
                returns = {'actual': 0., 'imagined': 0., 'teacher_forced': 0.}
                root_errors, joint_velocity_errors = [], []
                for step in range(32):
                    frame = start+step
                    full = state(data)
                    _, goal = rep.compute_wm_goal(motion, torch.tensor([frame]), full[:, :7])
                    action = actor(build_actor_obs_td({'motion': goal}, rep.represent(full)[:, None]))
                    aux, represented = dynamics[version](rep.represent(full), action, full[:, :7], contexts[version])
                    predicted = rep.reconstruct_full_state(aux, represented)
                    data.ctrl[ai] = np.array(contract['default_joint_pos'])+np.array(contract['action_scale'])*action[0].numpy()
                    for _ in range(4): mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    following = state(data)
                    root_errors.append(float(torch.linalg.vector_norm(predicted[:, :3]-following[:, :3])*100))
                    joint_velocity_errors.append(float(torch.mean((predicted[:, -29:]-following[:, -29:])**2)))
                    returns['actual'] += float(reward(following, action, last_real, frame))
                    returns['teacher_forced'] += float(reward(predicted, action, last_real, frame))
                    last_real = action
                    _, goal = rep.compute_wm_goal(motion, torch.tensor([frame]), imagined[:, :7])
                    action = actor(build_actor_obs_td({'motion': goal}, rep.represent(imagined)[:, None]))
                    aux, represented = dynamics[version](rep.represent(imagined), action, imagined[:, :7], contexts[version])
                    imagined = rep.reconstruct_full_state(aux, represented)
                    returns['imagined'] += float(reward(imagined, action, last_imagined, frame))
                    last_imagined = action
                result = {'iteration': version*1000, 'start_frame': start, 'horizon': 32,
                    'undiscounted_tracking_return': returns,
                    'one_step_root_position_error_mean_cm': float(np.mean(root_errors)),
                    'one_step_joint_velocity_rmse_rad_s': float(np.sqrt(np.mean(joint_velocity_errors))),
                    'imagined_root_xy_error_cm': float(torch.linalg.vector_norm(imagined[:, :2]-motion.root_pos_w[frame+1:frame+2, :2])*100)}
                reports.append(result)
                print(json.dumps(result), flush=True)
    probe.write(ROOT / 'frozen-dynamics-audit.json', {'diagnostic_only': True,
        'reports': reports, 'optimizer_steps': 0, 'architecture_modified': False,
        'official_source_modified': False, 'reward_definition': 'Unchanged official tracking reward, dt-scaled, summed over 32 steps. No critic bootstrap or training objective claim.',
        'interpretation_limit': 'A frozen model mismatch does not establish which optimizer or training-data choice caused regression.'})


if __name__ == '__main__':
    main()
