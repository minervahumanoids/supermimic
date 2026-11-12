"""Causal standing probes with frozen actors and standalone MuJoCo.

All altered inputs are diagnostic copies. Source data, official modules,
architecture, checkpoints, and the approved review remain unchanged.
"""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import wandb

from mjlab.sim import MujocoCfg
from mjlab.utils.torch import configure_torch_backends
from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.tasks.task_spec import build_actor_obs_td

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REVIEW = WORK / "evaluation-403000"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkpoint(version):
    if version == 403:
        return WORK / "stopped-by-user-20261112/checkpoint-v403/actor.pt"
    folder = ROOT / "checkpoints" / f"v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api(timeout=60).artifact(f"minervahumanoids/Supermimic/model-ga216ipt:v{version}")
    assert artifact.logged_by().id == "ga216ipt"
    entry = artifact.get_path("actor.pt")
    path = folder / "actor.pt"
    if not path.exists():
        entry.download(root=str(folder))
    assert base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode() == entry.digest
    write(folder / "receipt.json", {"artifact": artifact.qualified_name,
        "digest": artifact.digest, "actor_sha256": sha(path), "iteration": version*1000})
    return path


def rollout(label, path, device, precision, mode, original, contract, config, foot_friction=None, inference_batch=1):
    configure_torch_backends(allow_tf32=precision == "tf32")
    arrays = {key: value.copy() for key, value in original.items()}
    keys = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
    if mode == "held_pose":
        for key in keys:
            arrays[key][:] = arrays[key][0]
        for key in ("joint_vel", "root_lin_vel_w", "root_ang_vel_w"):
            arrays[key][:] = 0
    elif mode == "zero_goal_joint_velocity":
        arrays["joint_vel"][:] = 0
    n = len(arrays["joint_pos"])
    motion = SimpleNamespace(**{key: torch.tensor(arrays[key], device=device) for key in keys},
        motion_ends=torch.tensor([n], device=device))
    rep = create_representation("gencoord", 71, 29, 0.02, 4,
        torch.tensor(contract["default_joint_pos"], dtype=torch.float32, device=device), config.actor_obs)
    actor = None
    if path is not None:
        actor = build_actor(config, SimpleNamespace(representation=rep, num_goals=670, num_actions=29), device)
        actor.load_state_dict(torch.load(path, weights_only=True, map_location=device), strict=True)
        actor.eval().requires_grad_(False)
    model = mujoco.MjModel.from_xml_path(str(REVIEW / "plant/scene.xml"))
    MujocoCfg(**contract["physics"]).apply(model)
    if foot_friction is not None:
        feet = [i for i in range(model.ngeom) if "foot" in model.geom(i).name and "collision" in model.geom(i).name]
        model.geom_friction[feet, 0] = foot_friction
    data = mujoco.MjData(model)
    qa, va, ai = (np.array(contract[key]) for key in ("qpos_addresses", "dof_addresses", "actuator_ids"))
    root = model.body("robot/pelvis").id
    limits = model.jnt_range[[model.joint(name).id for name in contract["model_joint_names"]]]
    midpoint, half = limits.mean(axis=1), np.diff(limits, axis=1).ravel()*.45
    data.qpos[:3] = original["root_pos_w"][0]
    data.qpos[3:7] = original["root_quat_w"][0]
    data.qpos[qa] = np.clip(original["joint_pos"][0], midpoint-half, midpoint+half)
    data.qvel[:3] = original["root_lin_vel_w"][0]
    data.qvel[3:6] = Rotation.from_quat(data.qpos[3:7], scalar_first=True).inv().apply(original["root_ang_vel_w"][0])
    data.qvel[va] = original["joint_vel"][0]
    mujoco.mj_forward(model, data)
    records = {key: [] for key in ("root", "quat", "q", "joint_vel", "root_vel", "actions", "ctrl")}
    stop = None
    for frame in range(n):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root, velocity, 0)
        for key, value in (("root", data.xpos[root]), ("quat", data.xquat[root]),
            ("q", data.qpos[qa]), ("joint_vel", data.qvel[va]), ("root_vel", velocity[3:])):
            records[key].append(value.copy())
        if frame == n-1:
            break
        if actor is not None:
            state = torch.tensor(np.r_[data.xpos[root], data.xquat[root], data.qpos[qa],
                velocity[3:], velocity[:3], data.qvel[va]], dtype=torch.float32, device=device)[None]
            _, goal = rep.compute_wm_goal(motion, torch.tensor([frame], device=device), state[:, :7])
            with torch.inference_mode():
                history = rep.represent(state)[:, None].expand(inference_batch, -1, -1)
                action = actor(build_actor_obs_td({"motion": goal.expand(inference_batch, -1)}, history))[0].cpu().numpy()
            control = np.array(contract["default_joint_pos"])+np.array(contract["action_scale"])*action
        else:
            control = original["joint_pos"][frame if mode == "reference_pd" else 0]
            action = (control-np.array(contract["default_joint_pos"]))/np.array(contract["action_scale"])
        records["actions"].append(action.copy())
        records["ctrl"].append(control.copy())
        data.ctrl[ai] = control
        for _ in range(4):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or abs(data.xpos[root, 2]-original["root_pos_w"][frame, 2]) > .25:
            stop = {"frame": frame+1, "reason": "nonfinite state or native height gate"}
            if np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all():
                terminal_velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root, terminal_velocity, 0)
                for key, value in (("root", data.xpos[root]), ("quat", data.xquat[root]),
                    ("q", data.qpos[qa]), ("joint_vel", data.qvel[va]), ("root_vel", terminal_velocity[3:])):
                    records[key].append(value.copy())
            break
    record = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(ROOT / "cases" / (label+".npz"), **record)
    count = len(record["root"])
    error = np.linalg.norm(record["root"][:, :2]-original["root_pos_w"][:count, :2], axis=1)*100
    result = {"label": label, "backend": "standalone CPU MuJoCo", "inference_device": device,
        "precision": precision, "mode": mode, "frames": count, "complete": count == n and stop is None,
        "foot_friction_override": foot_friction, "inference_batch": inference_batch,
        "stop": stop, "actor_sha256": sha(path) if path else None,
        "root_xy_mean_cm": float(error.mean()), "root_xy_final_cm": float(error[-1]),
        "root_xy_max_cm": float(error.max()),
        "root_velocity_xy_rms_m_s": float(np.sqrt(np.mean(record["root_vel"][:, :2]**2))),
        "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(record["joint_vel"]**2))),
        "action_change_rms": float(np.sqrt(np.mean(np.diff(record["actions"], axis=0)**2))) if len(record["actions"])>1 else None,
        "optimizer_steps": 0, "source_data_modified": False}
    write(ROOT / "cases" / (label+".json"), result)
    print(json.dumps(result), flush=True)
    return result


def main():
    (ROOT / "cases").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    config = load_runner_cfg(json.loads((WORK / "stopped-by-user-20261112/launch-config.json").read_text()))
    contract = json.loads((REVIEW / "plant-contract.json").read_text())
    original = dict(np.load(REVIEW / "inputs/bones/robot/00.npz", allow_pickle=False))
    reports, failures = [], []
    latest = checkpoint(403)
    cases = [("v403_cpu", latest, "cpu", "ieee", "motion"),
        ("v403_cuda_ieee", latest, "cuda:0", "ieee", "motion"),
        ("v403_cuda_tf32", latest, "cuda:0", "tf32", "motion"),
        ("v403_held_pose", latest, "cpu", "ieee", "held_pose"),
        ("v403_zero_goal_joint_velocity", latest, "cpu", "ieee", "zero_goal_joint_velocity"),
        ("reference_pd", None, "cpu", "ieee", "reference_pd"),
        ("held_reference_pd", None, "cpu", "ieee", "held_reference_pd")]
    for args in cases:
        reports.append(rollout(*args, original, contract, config))
    for version in (0, 100, 200, 300):
        try:
            path = checkpoint(version)
            reports.append(rollout(f"v{version}_cpu", path, "cpu", "ieee", "motion", original, contract, config))
        except Exception as error:
            failures.append({"version": version, "error_type": type(error).__name__, "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
    write(ROOT / "probe-summary.json", {"diagnostic_only": True, "reports": reports,
        "failures": failures, "new_training_started": False, "official_source_modified": False})


if __name__ == "__main__":
    main()
