"""Run the unchanged approved G1 evaluation function on a preserved milestone."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb

from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
ORIGINAL = WORK / "evaluation-403000"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(version=100):
    torch.set_num_threads(4)
    output = ROOT / f"review-{version*1000}"
    checkpoint = ROOT / f"checkpoints/v{version}"
    artifact = wandb.Api(timeout=60).artifact(f"minervahumanoids/Supermimic/model-ga216ipt:v{version}")
    assert artifact.logged_by().id == "ga216ipt"
    for name in ("world_model.pt", "critic.pt"):
        if not (checkpoint / name).exists():
            artifact.get_path(name).download(root=str(checkpoint))
    receipt = json.loads((checkpoint / "receipt.json").read_text())
    assert sha(checkpoint / "actor.pt") == receipt["actor_sha256"]
    contract = json.loads((ORIGINAL / "plant-contract.json").read_text())
    wm = torch.load(checkpoint / "world_model.pt", map_location="cpu", weights_only=True)["dynamics"]
    for key, values in (("action_scale", contract["action_scale"]),
                        ("action_offset", contract["default_joint_pos"])):
        torch.testing.assert_close(wm[key].reshape(-1), torch.tensor(values), rtol=0, atol=1e-6)
    config = load_runner_cfg(json.loads((WORK / "stopped-by-user-20261112/launch-config.json").read_text()))
    representation = create_representation("gencoord", 71, 29, .02, 4,
        torch.tensor(contract["default_joint_pos"]), config.actor_obs)
    actor = build_actor(config, SimpleNamespace(representation=representation,
        num_goals=670, num_actions=29), "cpu")
    actor.load_state_dict(torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True), strict=True)
    actor.eval().requires_grad_(False)
    output.mkdir(exist_ok=True)
    for name in ("inputs", "plant"):
        link = output / name
        if not link.exists():
            link.symlink_to(ORIGINAL / name, target_is_directory=True)
    for name in ("plant-contract.json", "references-ready.json"):
        (output / name).write_bytes((ORIGINAL / name).read_bytes())
    source = ORIGINAL / "evaluate_g1.py"
    spec = importlib.util.spec_from_file_location("approved_g1_evaluator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = output
    module.ACTOR_SHA = receipt["actor_sha256"]
    reports = {}
    for cohort in ("bones", "pico"):
        report = module.evaluate(cohort, actor, representation, contract)
        report["iteration"] = version * 1000
        report["checkpoint_artifact"] = artifact.qualified_name
        report["metadata_adapter"] = {"fixed_source_iteration": 403000,
            "actual_iteration": version*1000,
            "reason": "The unchanged evaluator function has a fixed iteration label. Only checkpoint metadata changes.",
            "evaluator_sha256": sha(source)}
        module.write(output / "recordings" / cohort / "report.json", report)
        reports[cohort] = report
    import numpy as np
    summary = {cohort: {"clips": len(report["trials"]),
        "complete": sum(trial["complete"] for trial in report["trials"]),
        "requested_frames": sum(trial["requested_frames"] for trial in report["trials"]),
        "recorded_frames": sum(trial["frames"] for trial in report["trials"]),
        **{key: float(np.mean([trial[field] for trial in report["trials"]])) for key, field in (
            ("mean_clip_root_xy_cm", "root_xy_mean_cm"),
            ("mean_clip_palm_position_cm", "palm_position_mean_cm"),
            ("mean_clip_palm_orientation_deg", "palm_orientation_mean_deg"))}}
        for cohort, report in reports.items()}
    module.write(output / "evaluation-summary.json", {"iteration": version*1000,
        "backend": "standalone CPU MuJoCo", "cohorts": summary,
        "policy_modified": False, "optimizer_steps": 0, "evaluator_modified": False,
        "checkpoint_artifact": artifact.qualified_name, "actor_sha256": receipt["actor_sha256"]})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=100)
    main(parser.parse_args().version)
