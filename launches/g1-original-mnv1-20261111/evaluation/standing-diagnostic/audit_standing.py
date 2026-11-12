"""Diagnose standing with native observations and standalone MuJoCo.

The native MuJoCo-Warp path is a diagnostic only. The approved review continues
to use standalone MuJoCo recordings. This script creates no learner or W&B run.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import MujocoCfg
from supermimic.representations import create_representation, extract_actor_goal
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.scripts.train import TrainConfig
from supermimic.tasks.task_spec import build_actor_obs_td

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OUT = ROOT / "standing-diagnostic"
ACTOR = WORK / "stopped-by-user-20261112/checkpoint-v403/actor.pt"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cpu_state(model, data, contract):
    root = model.body("robot/pelvis").id
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root, velocity, 0)
    return torch.tensor(np.r_[data.xpos[root], data.xquat[root],
        data.qpos[contract["qpos_addresses"]], velocity[3:], velocity[:3],
        data.qvel[contract["dof_addresses"]]], dtype=torch.float32)[None]


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(69)
    np.random.seed(69)
    actor_sha = hashlib.sha256(ACTOR.read_bytes()).hexdigest()
    assert actor_sha == "07e064bd4b6500b97a4a6b4af0060a801588abb8f0abdceb38138e2c40c0144b"
    contract = json.loads((ROOT / "plant-contract.json").read_text())
    launch = json.loads((WORK / "stopped-by-user-20261112/launch-config.json").read_text())
    runner = load_runner_cfg(launch)
    cfg = TrainConfig.from_task("SuperMimic-GenCoord-G1").env
    cfg.scene.num_envs = 1
    cfg.seed = 69
    cfg.episode_length_s = 1e9
    cfg.events = {}
    cfg.terminations = {}
    cfg.auto_reset = False
    cfg.observations["actor"].enable_corruption = False
    command_cfg = cfg.commands["motion"]
    command_cfg.motion_file = str(ROOT / "inputs/bones/robot/00.npz")
    command_cfg.pose_range = {}
    command_cfg.velocity_range = {}
    command_cfg.sampling_mode = "start"
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    command = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    assert np.allclose(env.scene.env_origins.cpu().numpy(), 0)
    representation = create_representation(runner.representation, 71, 29,
        env.step_dt, cfg.decimation, robot.data.default_joint_pos[0], runner.actor_obs)
    actor = build_actor(runner, SimpleNamespace(representation=representation,
        num_goals=670, num_actions=29), "cuda:0")
    actor.load_state_dict(torch.load(ACTOR, map_location="cuda:0", weights_only=True), strict=True)
    actor.eval().requires_grad_(False)
    cpu_rep = create_representation(runner.representation, 71, 29, env.step_dt,
        cfg.decimation, robot.data.default_joint_pos[0].cpu(), runner.actor_obs)
    cpu_actor = build_actor(runner, SimpleNamespace(representation=cpu_rep,
        num_goals=670, num_actions=29), "cpu")
    cpu_actor.load_state_dict(torch.load(ACTOR, map_location="cpu", weights_only=True), strict=True)
    cpu_actor.eval().requires_grad_(False)
    cpu_motion = SimpleNamespace(**{key: getattr(command.motion, key).cpu()
        for key in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
                    "joint_pos", "joint_vel", "motion_ends")})
    model = mujoco.MjModel.from_xml_path(str(ROOT / "plant/scene.xml"))
    MujocoCfg(**contract["physics"]).apply(model)
    differences = {}
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos",
        "body_quat", "jnt_pos", "jnt_axis", "jnt_range", "dof_armature", "dof_damping",
        "dof_frictionloss", "geom_size", "geom_pos", "geom_quat", "geom_friction",
        "geom_condim", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
        "geom_margin", "geom_gap", "actuator_gainprm", "actuator_biasprm", "actuator_trnid",
        "actuator_forcerange", "actuator_ctrlrange", "actuator_gear"):
        a, b = np.asarray(getattr(model, name)), np.asarray(getattr(env.sim.mj_model, name))
        differences[name] = {"same_shape": a.shape == b.shape,
            "max_abs": float(np.max(np.abs(a-b))) if a.shape == b.shape and a.size else None}
    write("model-parity.json", {"fields": differences,
        "native_joint_names": robot.joint_names,
        "action_target_names": env.action_manager.get_term("joint_pos").target_names,
        "native_physics": asdict(cfg.sim.mujoco), "actor_sha256": actor_sha})

    env.reset()
    ids = torch.tensor([0], device=env.device)
    command.reset_to_frame(ids, 0)
    env.sim.forward()
    command.update_relative_body_poses()
    env.observation_manager.compute(update_history=True)
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(model, data)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    parity = []
    for frame in range(100):
        # Compare identical generalized states. Each native transition supplies
        # the next state, so closed-loop trajectory divergence cannot hide errors.
        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        obs = env.get_observations()
        state = obs["full_state"]
        goal = extract_actor_goal(obs["goal"])
        cs = cpu_state(model, data, contract)
        _, cg = cpu_rep.compute_wm_goal(cpu_motion, command.time_steps.cpu(), cs[:, :7])
        with torch.inference_mode():
            action = actor(build_actor_obs_td({"motion": goal}, representation.represent(state)[:, None]))
            cpu_action = cpu_actor(build_actor_obs_td({"motion": cg}, cpu_rep.represent(cs)[:, None]))
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        controls = env.sim.data.ctrl[0].cpu().numpy()
        expected = np.zeros(model.nu)
        expected[contract["actuator_ids"]] = (np.array(contract["default_joint_pos"]) +
            np.array(contract["action_scale"]) * cpu_action[0].numpy())
        parity.append({"frame": frame,
            "state_max_abs": float((state.cpu()-cs).abs().max()),
            "goal_max_abs": float((goal.cpu()-cg).abs().max()),
            "action_max_abs": float((action.cpu()-cpu_action).abs().max()),
            "control_max_abs": float(np.max(np.abs(controls-expected)))})
        env.step(action)
    write("input-parity.json", {"frames": parity,
        "maxima": {key: max(row[key] for row in parity) for key in parity[0] if key != "frame"}})

    # Independent clean trajectories use the same initial state and checkpoint.
    command.reset_to_frame(ids, 0)
    env.sim.data.qpos[0] = torch.tensor(initial_qpos, device=env.device, dtype=torch.float32)
    env.sim.data.qvel[0] = torch.tensor(initial_qvel, device=env.device, dtype=torch.float32)
    env.sim.forward()
    command.update_relative_body_poses()
    env.observation_manager.compute(update_history=True)
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    records = {key: [] for key in ("native_root", "native_q", "cpu_root", "cpu_q", "reference_root")}
    for frame in range(1001):
        obs = env.get_observations()
        state = obs["full_state"]
        cs = cpu_state(model, data, contract)
        records["native_root"].append(state[0, :3].cpu().numpy().copy())
        records["native_q"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        records["cpu_root"].append(cs[0, :3].numpy().copy())
        records["cpu_q"].append(data.qpos[contract["qpos_addresses"]].copy())
        records["reference_root"].append(cpu_motion.root_pos_w[frame].numpy().copy())
        if frame == 1000:
            break
        goal = extract_actor_goal(obs["goal"])
        _, cg = cpu_rep.compute_wm_goal(cpu_motion, torch.tensor([frame]), cs[:, :7])
        with torch.inference_mode():
            action = actor(build_actor_obs_td({"motion": goal}, representation.represent(state)[:, None]))
            cpu_action = cpu_actor(build_actor_obs_td({"motion": cg}, cpu_rep.represent(cs)[:, None]))
        data.ctrl[contract["actuator_ids"]] = (np.array(contract["default_joint_pos"]) +
            np.array(contract["action_scale"]) * cpu_action[0].numpy())
        for _ in range(contract["decimation"]):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        env.step(action)
    arrays = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(OUT / "trajectories.npz", **arrays)
    summary = {"native_backend": "MuJoCo-Warp diagnostic only",
        "review_backend": "standalone CPU MuJoCo", "frames": 1001, "duration_s": 20,
        "actor_sha256": actor_sha, "policy_modified": False, "optimizer_steps": 0,
        "input_parity_maxima": {key: max(row[key] for row in parity) for key in parity[0] if key != "frame"},
        "trajectories": {}}
    for name in ("native", "cpu"):
        drift = np.linalg.norm(arrays[name+"_root"][:, :2]-arrays["reference_root"][:, :2], axis=1)*100
        summary["trajectories"][name] = {"mean_root_xy_cm": float(drift.mean()),
            "final_root_xy_cm": float(drift[-1]), "max_root_xy_cm": float(drift.max())}
    base_state = torch.cat([cpu_motion.root_pos_w[:1], cpu_motion.root_quat_w[:1],
        cpu_motion.joint_pos[:1], cpu_motion.root_lin_vel_w[:1],
        cpu_motion.root_ang_vel_w[:1], cpu_motion.joint_vel[:1]], dim=-1)
    shifted = base_state.clone()
    shifted[:, 0] += 1.0
    _, goal0 = cpu_rep.compute_wm_goal(cpu_motion, torch.tensor([0]), base_state[:, :7])
    _, goal1 = cpu_rep.compute_wm_goal(cpu_motion, torch.tensor([0]), shifted[:, :7])
    with torch.inference_mode():
        a0 = cpu_actor(build_actor_obs_td({"motion": goal0}, cpu_rep.represent(base_state)[:, None]))
        a1 = cpu_actor(build_actor_obs_td({"motion": goal1}, cpu_rep.represent(shifted)[:, None]))
    summary["translation_observability"] = {"root_x_shift_m": 1.0,
        "represented_state_max_abs": float((cpu_rep.represent(base_state)-cpu_rep.represent(shifted)).abs().max()),
        "actor_goal_max_abs": float((goal0-goal1).abs().max()),
        "action_max_abs": float((a0-a1).abs().max())}
    write("summary.json", summary)
    print(json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    main()
