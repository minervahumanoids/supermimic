"""Measure local control-period amplification with unchanged frozen actors."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import probe
from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.tasks.task_spec import build_actor_obs_td

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REVIEW = WORK / "evaluation-403000"


def main():
    torch.set_num_threads(4)
    config = load_runner_cfg(json.loads((WORK / "stopped-by-user-20261112/launch-config.json").read_text()))
    contract = json.loads((REVIEW / "plant-contract.json").read_text())
    archive = dict(np.load(REVIEW / "inputs/bones/robot/00.npz"))
    motion = SimpleNamespace(**{key: torch.tensor(archive[key]) for key in
        ("root_pos_w", "root_quat_w", "joint_pos", "root_lin_vel_w", "root_ang_vel_w", "joint_vel")},
        motion_ends=torch.tensor([1001]))
    model = mujoco.MjModel.from_xml_path(str(REVIEW / "plant/scene.xml"))
    probe.MujocoCfg(**contract["physics"]).apply(model)
    qa, va, ai = (np.array(contract[key]) for key in ("qpos_addresses", "dof_addresses", "actuator_ids"))
    root = model.body("robot/pelvis").id
    rep = create_representation("gencoord", 71, 29, .02, 4,
        torch.tensor(contract["default_joint_pos"]), config.actor_obs)
    reports = []
    for version in (100, 403):
        path = (ROOT / "checkpoints/v100/actor.pt" if version == 100 else
            WORK / "stopped-by-user-20261112/checkpoint-v403/actor.pt")
        actor = build_actor(config, SimpleNamespace(representation=rep, num_goals=670, num_actions=29), "cpu")
        actor.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        actor.eval().requires_grad_(False)
        for frame in (0, 500):
            data = mujoco.MjData(model)
            data.qpos[:3] = archive["root_pos_w"][frame]
            data.qpos[3:7] = archive["root_quat_w"][frame]
            data.qpos[qa] = archive["joint_pos"][frame]
            data.qvel[:3] = archive["root_lin_vel_w"][frame]
            data.qvel[3:6] = Rotation.from_quat(data.qpos[3:7], scalar_first=True).inv().apply(archive["root_ang_vel_w"][frame])
            data.qvel[va] = archive["joint_vel"][frame]
            mujoco.mj_forward(model, data)
            initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
            def control(qpos, qvel):
                sample = mujoco.MjData(model)
                sample.qpos[:] = qpos
                sample.qvel[:] = qvel
                mujoco.mj_forward(model, sample)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, sample, mujoco.mjtObj.mjOBJ_XBODY, root, velocity, 0)
                full = torch.tensor(np.r_[sample.xpos[root], sample.xquat[root], sample.qpos[qa],
                    velocity[3:], velocity[:3], sample.qvel[va]], dtype=torch.float32)[None]
                _, goal = rep.compute_wm_goal(motion, torch.tensor([frame]), full[:, :7])
                with torch.inference_mode():
                    action = actor(build_actor_obs_td({"motion": goal}, rep.represent(full)[:, None]))[0].numpy()
                result = np.zeros(model.nu)
                result[ai] = np.array(contract["default_joint_pos"])+np.array(contract["action_scale"])*action
                return result
            data.ctrl[:] = control(initial_qpos, initial_qvel)
            size = 2*model.nv+model.na
            transition, inputs = np.eye(size), np.zeros((size, model.nu))
            for _ in range(4):
                a, b = np.zeros((size, size)), np.zeros((size, model.nu))
                mujoco.mjd_transitionFD(model, data, 1e-6, True, a, b, None, None)
                transition, inputs = a@transition, a@inputs+b
                mujoco.mj_step(model, data)
            for epsilon in (1e-4, 3e-4):
                jacobian = np.zeros((model.nu, size))
                for column in range(size):
                    sides = []
                    for sign in (-1, 1):
                        qpos, qvel = initial_qpos.copy(), initial_qvel.copy()
                        if column < model.nv:
                            tangent = np.zeros(model.nv)
                            tangent[column] = sign
                            mujoco.mj_integratePos(model, qpos, tangent, epsilon)
                        else:
                            qvel[column-model.nv] += sign*epsilon
                        sides.append(control(qpos, qvel))
                    jacobian[:, column] = (sides[1]-sides[0])/(2*epsilon)
                values = np.linalg.eigvals(transition+inputs@jacobian)
                selected = sorted(values, key=abs, reverse=True)[:8]
                report = {"iteration": version*1000, "source_frame": frame,
                    "state_difference_epsilon": epsilon,
                    "control_dt": .02,
                    "largest_modes": [{"magnitude": float(abs(value)),
                        "frequency_hz": float(abs(np.angle(value))/(2*np.pi*.02)),
                        "real": float(value.real), "imaginary": float(value.imag)} for value in selected]}
                reports.append(report)
                print(json.dumps(report), flush=True)
    probe.write(ROOT / "control-linearization.json", {"diagnostic_only": True,
        "reports": reports, "interpretation_limit": "Local derivatives at reference states do not establish global stability or exact equilibrium.",
        "official_source_modified": False, "architecture_modified": False, "optimizer_steps": 0})


if __name__ == "__main__":
    main()
