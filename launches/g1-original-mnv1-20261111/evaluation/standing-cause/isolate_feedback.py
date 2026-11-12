"""Separate normalization drift from learned velocity feedback without training."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
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
    original = dict(np.load(REVIEW / "inputs/bones/robot/00.npz", allow_pickle=False))
    paths = {100: ROOT / "checkpoints/v100/actor.pt",
             403: WORK / "stopped-by-user-20261112/checkpoint-v403/actor.pt"}
    states = {version: torch.load(path, map_location="cpu", weights_only=True) for version, path in paths.items()}
    variants = [("late_early_obs_norm", 403, 100, ("obs_normalizer.",)),
        ("late_early_goal_norm", 403, 100, ("goal_normalizers.",)),
        ("late_early_all_norm", 403, 100, ("obs_normalizer.", "goal_normalizers.")),
        ("early_late_all_norm", 100, 403, ("obs_normalizer.", "goal_normalizers."))]
    output = ROOT / "diagnostic-derivatives"
    output.mkdir(exist_ok=True)
    results = []
    for label, weight_version, normalization_version, prefixes in variants:
        state = {key: (states[normalization_version][key] if key.startswith(prefixes) else value).clone()
                 for key, value in states[weight_version].items()}
        path = output / (label+".pt")
        torch.save(state, path)
        result = probe.rollout(label, path, "cpu", "ieee", "motion", original, contract, config)
        result["diagnostic_only"] = True
        result["weights_from_iteration"] = weight_version*1000
        result["normalization_from_iteration"] = normalization_version*1000
        result["replaced_prefixes"] = prefixes
        results.append(result)
    feedback = []
    model = mujoco.MjModel.from_xml_path(str(REVIEW / "plant/scene.xml"))
    ids = np.array(contract["actuator_ids"])
    kp, kd = model.actuator_gainprm[ids, 0], -model.actuator_biasprm[ids, 2]
    rep = create_representation("gencoord", 71, 29, .02, 4,
        torch.tensor(contract["default_joint_pos"]), config.actor_obs)
    motion = SimpleNamespace(**{key: torch.tensor(original[key]) for key in
        ("root_pos_w", "root_quat_w", "joint_pos", "root_lin_vel_w", "root_ang_vel_w", "joint_vel")},
        motion_ends=torch.tensor([1001]))
    for version, state in states.items():
        actor = build_actor(config, SimpleNamespace(representation=rep, num_goals=670, num_actions=29), "cpu")
        actor.load_state_dict(state, strict=True)
        actor.eval().requires_grad_(False)
        for frame in (0, 250, 500, 750):
            full = torch.cat([getattr(motion, key)[frame] for key in
                ("root_pos_w", "root_quat_w", "joint_pos", "root_lin_vel_w", "root_ang_vel_w", "joint_vel")])[None]
            represented = rep.represent(full)
            _, goal = rep.compute_wm_goal(motion, torch.tensor([frame]), full[:, :7])
            def function(x):
                return actor(build_actor_obs_td({"motion": goal}, x[:, None]))[0]
            jacobian = torch.autograd.functional.jacobian(function, represented)[..., 0, -29:].numpy()
            with torch.no_grad():
                raw = function(represented).numpy()
            target = np.array(contract["default_joint_pos"])+np.array(contract["action_scale"])*raw
            j_target = np.array(contract["action_scale"])[:, None]*jacobian
            clip = model.actuator_ctrllimited[ids].astype(bool)
            in_range = (~clip) | ((target > model.actuator_ctrlrange[ids, 0]) & (target < model.actuator_ctrlrange[ids, 1]))
            j_target *= in_range[:, None]
            damping = np.diag(kd)-np.diag(kp)@j_target
            symmetric = (damping+damping.T)/2
            eigenvalues, vectors = np.linalg.eigh(symmetric)
            row = {"iteration": version*1000, "source_frame": frame,
                "position_target_velocity_jacobian_frobenius_s": float(np.linalg.norm(j_target)),
                "min_symmetric_damping_Nm_s_rad": float(eigenvalues[0]),
                "negative_damping_modes": int(np.sum(eigenvalues < 0)),
                "minimum_mode_joint_weights": vectors[:, 0].tolist()}
            feedback.append(row)
            print(json.dumps(row), flush=True)
    norms = {}
    for prefix in ("obs_normalizer", "goal_normalizers.motion"):
        mean0, mean1 = (states[v][prefix+"._mean"].numpy() for v in (100, 403))
        std0, std1 = (states[v][prefix+"._std"].numpy() for v in (100, 403))
        norms[prefix] = {"mean_change_in_early_std_rms": float(np.sqrt(np.mean(((mean1-mean0)/(std0+.01))**2))),
            "std_ratio_min": float(np.min((std1+.01)/(std0+.01))),
            "std_ratio_max": float(np.max((std1+.01)/(std0+.01)))}
    probe.write(ROOT / "feedback-isolation.json", {"diagnostic_only": True,
        "normalization_probes": results, "feedback_jacobians": feedback, "normalization_changes": norms,
        "interpretation_limit": "A negative symmetric damping mode is a local feedback measure, not a full closed-loop stability proof.",
        "official_source_modified": False, "architecture_modified": False, "optimizer_steps": 0})


if __name__ == "__main__":
    main()
