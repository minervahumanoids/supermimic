"""Completed-episode telemetry for the original G1 SuperMimic training."""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from pathlib import Path

import torch
import wandb


class StandardLogger:
  """Keep returns, lengths, and signed components on the same completed episodes."""

  def __init__(self, runner, local_path: Path | None = None):
    self.runner = runner
    self.env = runner.env.unwrapped
    self.local_path = local_path
    self.names = self.env.reward_manager.active_terms
    self.returns = torch.zeros(self.env.num_envs, device=runner.device)
    self.lengths = torch.zeros_like(self.returns)
    self.components = torch.zeros(
      self.env.num_envs, len(self.names), device=runner.device
    )
    self.episodes = deque(maxlen=100)
    self.routes = {
      spec.name: (lo, hi)
      for spec, (lo, hi) in zip(
        runner.policy_learner.actor.task_specs,
        runner.policy_learner.actor.slot_ranges(self.env.num_envs),
        strict=True,
      )
      if hi > lo
    }
    self.route_episodes = {name: deque(maxlen=100) for name in self.routes}
    self.steps = self.iterations = self.interval_steps = 0
    self.interval_reward = 0.0
    self.tracking = {name: [] for name in self.routes}
    self.start_clock()

  def start_clock(self):
    self.started = self.last_log = time.perf_counter()

  def process_env_step(self, rewards, terminated, extras):
    manager = self.env.termination_manager
    timeouts = manager.time_outs.bool()
    done = terminated.bool() | timeouts
    failures = terminated.bool() & ~timeouts
    self.returns += rewards
    self.lengths += 1
    self.components += self.env.reward_manager.term_rewards
    ids = done.nonzero().flatten()
    if ids.numel():
      values = (
        torch.cat(
          [
            self.returns[ids, None],
            self.lengths[ids, None],
            failures[ids, None].float(),
            timeouts[ids, None].float(),
            self.components[ids],
          ],
          dim=1,
        )
        .cpu()
        .tolist()
      )
      for index, value in zip(ids.cpu().tolist(), values, strict=True):
        self.episodes.append(value)
        for name, (lo, hi) in self.routes.items():
          if lo <= index < hi:
            self.route_episodes[name].append(value)
      self.returns[ids] = self.lengths[ids] = 0
      self.components[ids] = 0
    command = self.runner.motion_cmd
    root_error = torch.linalg.vector_norm(
      command.robot_anchor_pos_w[:, :2] - command.anchor_pos_w[:, :2],
      dim=-1,
    )
    for name, (lo, hi) in self.routes.items():
      self.tracking[name].append(root_error[lo:hi].mean())
    self.steps += self.env.num_envs
    self.interval_steps += self.env.num_envs
    self.iterations += 1
    self.interval_reward += rewards.sum().item()

  def log(
    self,
    it,
    total_it,
    collect_time,
    wm_learn_time,
    policy_learn_time,
    loss_dict,
    num_collected_steps,
    **kwargs,
  ):
    now = time.perf_counter()
    row = {
      "Train/iter": self.iterations,
      "Train/steps": self.steps,
      "Train/time_s": now - self.started,
      "Reward/step": self.interval_reward / self.interval_steps,
      "Perf/fps": self.interval_steps / (now - self.last_log),
      "Perf/collect_s": collect_time,
      "Perf/learn_s": wm_learn_time + policy_learn_time,
      "Perf/wm_update_s": wm_learn_time,
      "Perf/policy_update_s": policy_learn_time,
    }
    mapping = {
      "Loss/state": "Loss/wm",
      "Loss/wm_grad_norm": "Train/wm_grad_norm",
      "Policy/loss": "Loss/policy",
      "Policy/value_function": "Loss/value",
      "Policy/grad_norm": "Train/policy_grad_norm",
      "Policy/reward": "Train/imag_return",
      "Policy/bc_loss": "Loss/latent_bc",
    }
    for source, destination in mapping.items():
      if source in loss_dict:
        row[destination] = float(loss_dict[source])
    if "Loss/root_pos" in loss_dict:
      row["Metrics/wm_root_m"] = float(loss_dict["Loss/root_pos"]) ** 0.5
    if self.episodes:
      columns = list(zip(*self.episodes, strict=True))
      row.update(
        {
          "Reward/total": statistics.mean(columns[0]),
          "Episode/len_s": statistics.mean(columns[1]) * self.env.step_dt,
          "Episode/fail_frac": statistics.mean(columns[2]),
          "Episode/timeout_frac": statistics.mean(columns[3]),
        }
      )
      row.update(
        {
          f"Reward/terms/{name}": statistics.mean(columns[i + 4])
          for i, name in enumerate(self.names)
        }
      )
    for name, (lo, hi) in self.routes.items():
      row[f"Train/{name}/steps"] = self.iterations * (hi - lo)
      if self.route_episodes[name]:
        columns = list(zip(*self.route_episodes[name], strict=True))
        row[f"Reward/{name}/total"] = statistics.mean(columns[0])
        row[f"Episode/{name}/len_s"] = statistics.mean(columns[1]) * self.env.step_dt
        row[f"Episode/{name}/fail_frac"] = statistics.mean(columns[2])
      if self.tracking[name]:
        row[f"Metrics/{name}/root_xy_m"] = (
          torch.stack(self.tracking[name]).mean().item()
        )
      self.tracking[name].clear()
    if not all(torch.isfinite(torch.tensor(value)) for value in row.values()):
      raise RuntimeError("Nonfinite training telemetry")
    if self.local_path is None:
      wandb.log(row)
    else:
      with self.local_path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    self.interval_reward = 0.0
    self.interval_steps = 0
    self.last_log = now


def metric_definitions(reward_names):
  definitions = {
    "Train/iter": {
      "unit": "iteration",
      "reduction": "count",
      "population": "collector iterations",
    },
    "Train/steps": {
      "unit": "transition",
      "reduction": "count",
      "population": "actual simulator transitions",
    },
    "Reward/total": {"unit": "reward", "reduction": "mean completed episodic sum"},
    "Reward/step": {
      "unit": "reward/transition",
      "reduction": "sum applied rewards / actual transitions",
    },
    "Episode/len_s": {"unit": "s", "reduction": "mean completed episode duration"},
    "Episode/fail_frac": {
      "unit": "fraction",
      "reduction": "failures / completed episodes",
      "definition": "termination without a simultaneous timeout",
    },
    "Episode/timeout_frac": {
      "unit": "fraction",
      "reduction": "timeouts / completed episodes",
      "definition": "environment time_outs, including clip completion",
    },
    "Perf/fps": {
      "unit": "transition/s",
      "reduction": "actual transitions / full wall time between log calls",
    },
    "Metrics/wm_root_m": {
      "unit": "m",
      "frame": "world",
      "reduction": "root position RMS over sampled WM prediction windows",
    },
  }
  for name in reward_names:
    definitions[f"Reward/terms/{name}"] = {
      "unit": "reward",
      "reduction": "mean signed completed episodic sum",
    }
  for key, unit, reduction in (
    (
      "Loss/wm",
      "dimensionless",
      "mean normalized state prediction loss over WM windows",
    ),
    (
      "Loss/policy",
      "reward",
      "mean negative imagined return plus weighted latent loss",
    ),
    (
      "Loss/value",
      "reward squared",
      "mean squared GAE target error over critic minibatches",
    ),
    ("Loss/latent_bc", "dimensionless", "mean weighted latent imitation loss"),
    (
      "Train/wm_grad_norm",
      "gradient norm",
      "mean L2 norm before clipping over WM updates",
    ),
    (
      "Train/policy_grad_norm",
      "gradient norm",
      "mean L2 norm before clipping over policy updates",
    ),
    (
      "Train/imag_return",
      "reward",
      "mean discounted imagined return with critic bootstrap",
    ),
    ("Train/time_s", "s", "full elapsed wall time since training admission"),
    ("Perf/collect_s", "s", "collection wall time over the logging interval"),
    ("Perf/learn_s", "s", "combined learner wall time over the logging interval"),
    ("Perf/wm_update_s", "s", "WM update wall time over the logging interval"),
    ("Perf/policy_update_s", "s", "policy update wall time over the logging interval"),
  ):
    definitions[key] = {"unit": unit, "reduction": reduction}
  for route in ("motion", "velocity"):
    for key, unit, reduction in (
      (
        f"Train/{route}/steps",
        "transition",
        "count of actual transitions for the encoder route",
      ),
      (
        f"Reward/{route}/total",
        "reward",
        "mean completed episodic sum for the encoder route",
      ),
      (
        f"Episode/{route}/len_s",
        "s",
        "mean completed episode duration for the encoder route",
      ),
      (
        f"Episode/{route}/fail_frac",
        "fraction",
        "failures / completed episodes for the encoder route",
      ),
      (
        f"Metrics/{route}/root_xy_m",
        "m",
        "mean instantaneous XY root error over route environments and collector steps",
      ),
    ):
      definitions[key] = {
        "unit": unit,
        "reduction": reduction,
        "population": f"{route} training diagnostics",
      }
      if key.startswith("Metrics/"):
        definitions[key]["frame"] = "original world XY"
      if key.startswith(("Reward/", "Episode/")):
        definitions[key]["sampling_window"] = (
          "last 100 completed episodes for this route"
        )
  for key, definition in definitions.items():
    definition.setdefault("frame", "not applicable")
    definition.setdefault(
      "sampling_window",
      "last 100 completed episodes"
      if key.startswith(("Episode/", "Reward/total", "Reward/terms/"))
      else "logging interval",
    )
    definition.setdefault("population", "training diagnostics")
    definition.setdefault("denominator", definition["reduction"])
  return definitions


def create_workspace(run, reward_names, routes=("motion",)):
  import wandb_workspaces.reports.v2 as wr
  import wandb_workspaces.workspaces as ws

  keys = {
    "Reward": ["Reward/total", "Reward/step"]
    + [f"Reward/terms/{name}" for name in reward_names],
    "Loss": ["Loss/wm", "Loss/policy", "Loss/value"],
    "Train": ["Train/steps", "Train/wm_grad_norm", "Train/policy_grad_norm"],
    "Metrics": [
      *[f"Metrics/{route}/root_xy_m" for route in routes],
      "Metrics/wm_root_m",
    ],
    "Episode": ["Episode/len_s", "Episode/fail_frac", "Episode/timeout_frac"],
    "Perf": [
      "Perf/fps",
      "Perf/collect_s",
      "Perf/learn_s",
      "Perf/wm_update_s",
      "Perf/policy_update_s",
    ],
  }
  sections = []
  for name, metrics in keys.items():
    panels = [
      wr.LinePlot(
        title=metric,
        x="Train/iter",
        y=[metric],
        title_x="Training iterations",
        layout=wr.Layout(x=(i % 3) * 8, y=(i // 3) * 6, w=8, h=6),
      )
      for i, metric in enumerate(metrics)
    ]
    sections.append(
      ws.Section(
        name=name,
        panels=panels,
        is_open=True,
        layout_settings=ws.SectionLayoutSettings(columns=3, rows=1),
      )
    )
  workspace = ws.Workspace(
    entity=run.entity,
    project=run.project,
    name=run.name,
    sections=sections,
    settings=ws.WorkspaceSettings(x_axis="Train/iter", group_by_prefix="first"),
  ).save()
  restored = ws.Workspace.from_url(workspace.url)
  if restored.settings.x_axis != "Train/iter":
    raise RuntimeError("W&B workspace axis verification failed")
  if any(
    section.layout_settings.rows != 1 or section.layout_settings.columns != 3
    for section in restored.sections
  ):
    raise RuntimeError("W&B workspace layout verification failed")
  return workspace.url
