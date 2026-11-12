"""Launch the untouched official G1 learner with external training bookkeeping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import wandb
from orion import associate
from standard_logger import StandardLogger, create_workspace, metric_definitions

SOURCE = "abc124c1bd5976341034668af3d12c087f2f1f55"
DEPENDENCY = "dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd"
TASK = "SuperMimic-GenCoord-G1"


def digest(path):
  h = hashlib.sha256()
  with path.open("rb") as stream:
    for data in iter(lambda: stream.read(8 << 20), b""):
      h.update(data)
  return h.hexdigest()


def verify_source(repo, expected):
  head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
  ).strip()
  if head != expected:
    raise RuntimeError(f"Unexpected upstream source: {head}")
  changed = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True
  )
  if changed:
    raise RuntimeError("The upstream checkout must remain unchanged")


def verify_data(path):
  from mjlab.scripts.csv_to_npz import JOINT_NAMES

  if path.is_dir():
    provenance = json.loads((path / "provenance.json").read_text())
    if provenance["rate_hz"] != 50 or provenance["mjlab_commit"] != DEPENDENCY:
      raise ValueError("Dataset conversion does not match the official G1 runtime")
    names = json.loads((path / "joint_names.json").read_text())
    ends = np.load(path / "motion_ends.npy", allow_pickle=False)
    arrays = {
      name: np.load(path / (name + ".npy"), mmap_mode="r", allow_pickle=False)
      for name in (
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
      )
    }
    frames = int(ends[-1])
    if ends.ndim != 1 or np.any(np.diff(np.r_[0, ends]) < 3):
      raise ValueError("Invalid G1 clip boundaries")
    for name, array in arrays.items():
      width = 29 if name.startswith("joint_") else 4 if name == "root_quat_w" else 3
      if array.shape != (frames, width):
        raise ValueError(f"Invalid G1 field shape: {name}")
      for start in range(0, frames, 100000):
        if not np.isfinite(array[start : start + 100000]).all():
          raise ValueError(f"Nonfinite G1 data: {name}")
    fields = {
      str(p.relative_to(path)): digest(p) for p in sorted(path.iterdir()) if p.is_file()
    }
    if provenance["clips"] != len(ends) or provenance["frames"] != frames:
      raise ValueError("Dataset selection and packed boundaries differ")
    result = {
      "clips": len(ends),
      "frames": frames,
      "field_sha256": fields,
      "provenance": provenance,
    }
  else:
    with np.load(path, allow_pickle=False) as clip:
      names = clip["joint_names"].tolist()
      if clip["joint_pos"].shape[1] != 29 or float(clip["fps"].reshape(-1)[0]) != 50:
        raise ValueError("G1 needs 29 joints at 50 Hz")
      result = {"clips": 1, "frames": len(clip["joint_pos"]), "sha256": digest(path)}
  if names != list(JOINT_NAMES):
    raise ValueError("Dataset joints do not match the official G1 asset")
  result.update(robot="Unitree G1", path=str(path), rate_hz=50)
  return result


class TrainingLogger(StandardLogger):
  """Adapt telemetry without modifying the official learner or its reward."""

  def log(self, *args, **kwargs):
    losses = dict(kwargs.pop("loss_dict"))
    factor = self.runner.cfg.policy.horizon * self.env.step_dt
    if "Policy/reward" in losses:
      losses["Policy/reward"] = losses["Policy/reward"] * factor
      losses["Policy/loss"] = -losses["Policy/reward"] + losses.get("Policy/bc_loss", 0)
    gradients = [
      p.grad for p in self.runner._wm.dynamics.parameters() if p.grad is not None
    ]
    if gradients:
      losses["Loss/wm_grad_norm"] = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in gradients])
      )
    super().log(*args, loss_dict=losses, **kwargs)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workspace", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--mode", choices=("check", "smoke", "train"), default="check")
  parser.add_argument("--pr-url")
  parser.add_argument("--launch-commit")
  args = parser.parse_args()
  workspace = args.workspace.resolve()
  repo = workspace / "supermimic"
  verify_source(repo, SOURCE)
  verify_source(workspace / "mjlab_research", DEPENDENCY)
  dataset = verify_data(args.motion_file.resolve())
  from supermimic.runners.supermimic_runner import SuperMimicRunner
  from supermimic.scripts.train import TrainConfig
  from supermimic.scripts.train_utils import create_env, setup_device

  cfg = TrainConfig.from_task(TASK)
  original = asdict(cfg.runner)
  cfg.env.seed = 69
  cfg.env.scene.num_envs = cfg.num_envs
  cfg.env.commands["motion"].motion_file = str(args.motion_file.resolve())
  cfg.env.commands["motion"].sampling_mode = "external"
  cfg.registry_name = None
  cfg.runner.wandb_project = "Supermimic"
  date = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d")
  cfg.runner.run_name = f"g1-SM-original-{date}-seed69"
  effective = asdict(cfg.runner)
  if {k: v for k, v in original.items() if k not in ("wandb_project", "run_name")} != {
    k: v for k, v in effective.items() if k not in ("wandb_project", "run_name")
  }:
    raise RuntimeError("The original training parameters changed")
  record = {
    "source_commit": SOURCE,
    "mjlab_commit": DEPENDENCY,
    "robot": "G1",
    "host": socket.gethostname(),
    "dataset": dataset,
    "runner_cfg": effective,
    "env_cfg": asdict(cfg.env),
    "training_category": "SM",
    "pull_request": args.pr_url,
    "launch_commit": args.launch_commit,
    "checkpoint_provenance": "Fresh initialization. No checkpoint loaded.",
    "wall_time_limit": None,
    "unchanged_upstream_parameters": True,
  }
  (workspace / "launch-config.json").write_text(
    json.dumps(record, indent=2, default=str) + "\n"
  )
  print(
    json.dumps(
      {
        "mode": args.mode,
        "data": dataset,
        "iterations": cfg.runner.max_iterations,
        "num_envs": cfg.num_envs,
        "original_parameters": True,
      }
    ),
    flush=True,
  )
  if args.mode == "check":
    return
  if not torch.cuda.is_available():
    raise RuntimeError("G1 training requires CUDA")
  if args.mode == "train":
    if not args.pr_url or not args.launch_commit:
      parser.error("Training requires its own PR URL and launch commit")
    if os.environ.get("WANDB_MODE", "online") != "online":
      raise RuntimeError("Official training requires online telemetry")
    os.environ.update(
      WANDB_BASE_URL="https://api.wandb.ai", WANDB_ENTITY="minervahumanoids"
    )
    run = wandb.init(
      entity="minervahumanoids",
      project="Supermimic",
      name=cfg.runner.run_name,
      mode="online",
      config=record,
      settings=wandb.Settings(reinit="return_previous"),
    )
    run.define_metric("Train/iter")
    run.define_metric("*", step_metric="Train/iter")
    run.log({"Train/iter": 0, "Train/steps": 0})
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
      remote = wandb.Api().run(f"minervahumanoids/Supermimic/{run.id}")
      if list(remote.scan_history(keys=["Train/iter", "Train/steps"], page_size=10)):
        break
      time.sleep(2)
    else:
      run.finish(exit_code=1)
      raise RuntimeError("W&B did not confirm pre-training metrics")
  os.chdir(repo)
  device = setup_device("cuda:0")
  env = create_env(cfg.env, cfg.num_envs, device)
  runner = SuperMimicRunner(
    env=env, cfg=cfg.runner, task_id=TASK, registry_name=None, device=device
  )
  logger = TrainingLogger(
    runner, workspace / "smoke-metrics.jsonl" if args.mode == "smoke" else None
  )
  runner.logger = logger
  if args.mode == "smoke":
    if wandb.run is not None:
      raise RuntimeError("Smoke must never initialize W&B")
    runner.env.reset()
    runner._seed_sim_history()
    for step in range(32):
      runner.step()
      if not torch.isfinite(runner.env.get_observations()["full_state"]).all():
        raise RuntimeError("Nonfinite G1 collection")
      if step % 8 == 0:
        print(json.dumps({"smoke_collection_step": step + 1}), flush=True)
    for update in range(5):
      batches = [
        runner.replay_buffer.sample(cfg.runner.wm.horizon, cfg.runner.wm.batch_size)
        for _ in range(cfg.runner.wm.num_batches)
      ]
      losses = runner.world_model_learner.update(batches)
      losses.update(runner.policy_learner.update())
      if not all(torch.isfinite(v).all() for v in losses.values()):
        raise RuntimeError("Nonfinite native G1 learner values")
      print(json.dumps({"smoke_learner_update": update + 1}), flush=True)
    receipt = {
      "passed": True,
      "num_envs": cfg.num_envs,
      "wm_batch": cfg.runner.wm.batch_size,
      "policy_batch": cfg.runner.policy.batch_size,
      "original_parameters": True,
      "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
      "wandb_initialized": False,
      "dataset": dataset,
      "source_commit": SOURCE,
    }
    (workspace / "smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    env.close()
    return
  definitions = metric_definitions(env.unwrapped.reward_manager.active_terms)
  definitions["Train/wm_grad_norm"].update(
    reduction="L2 gradient norm after clipping from the latest WM update",
    sampling_window="latest WM update",
  )
  run.config.update({"metric_definitions": definitions}, allow_val_change=True)
  url = create_workspace(
    run, env.unwrapped.reward_manager.active_terms, routes=tuple(logger.routes)
  )
  associate(run, args.pr_url, workspace)
  (workspace / "telemetry-ready.json").write_text(
    json.dumps(
      {
        "run_id": run.id,
        "wandb_url": run.url,
        "workspace_url": url,
        "pr_url": args.pr_url,
        "server_confirmed": True,
      },
      indent=2,
    )
    + "\n"
  )
  logger.start_clock()
  runner.run()


if __name__ == "__main__":
  main()
