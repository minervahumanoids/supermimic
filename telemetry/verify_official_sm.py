"""Check completed episodes, pooled root samples, and actual timing denominators."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from official_sm_contract import normalize
from official_sm_logger import StandardLogger, metric_definitions


def main():
    manager = NS(time_outs=torch.zeros(2, dtype=torch.bool))
    reward_manager = NS(active_terms=["motion_global_root_pos", "action_rate_l2"])
    env = NS(num_envs=2, step_dt=0.02, termination_manager=manager,
             reward_manager=reward_manager)
    actor = NS(task_specs=[NS(name="motion")], slot_ranges=lambda n: [(0, n)])
    command = NS(anchor_pos_w=torch.zeros(2, 3), robot_anchor_pos_w=torch.zeros(2, 3))
    runner = NS(env=NS(unwrapped=env), device="cpu", motion_cmd=command,
                policy_learner=NS(actor=actor))
    rewards = [(1., 2.), (2., 3.), (1., 4.), (1., 5.)]
    root_errors = [(1., 2.), (100., 4.), (3., 100.), (9., 10.)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.jsonl"
        with patch("official_sm_logger.time.perf_counter", side_effect=[100., 110., 112.]), \
             patch("official_sm_logger.wandb.init", side_effect=AssertionError("Smoke initialized W&B")), \
             patch("official_sm_logger.wandb.log", side_effect=AssertionError("Smoke published W&B")):
            logger = StandardLogger(runner, path)
            for index, values in enumerate(rewards):
                reward = torch.tensor(values)
                reward_manager.term_rewards = torch.stack([1.25 * reward, -0.25 * reward], dim=1)
                command.robot_anchor_pos_w[:, 0] = torch.tensor(root_errors[index])
                manager.time_outs = torch.tensor([False, index == 2])
                logger.process_env_step(reward, torch.tensor([index == 1, False]), {})
            logger.log(0, 1, 8., 8., 4., {}, 8)
            reward = torch.tensor([2., 2.])
            reward_manager.term_rewards = torch.stack([1.25 * reward, -0.25 * reward], dim=1)
            manager.time_outs = torch.zeros(2, dtype=torch.bool)
            logger.process_env_step(reward, torch.zeros(2, dtype=torch.bool), {})
            logger.log(0, 1, .3, .2, .1, {}, 2)
            assert torch.equal(logger.returns, torch.tensor([4., 7.]))
        first, second = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {
        "Reward/total": 6., "Reward/step": 19. / 8.,
        "Reward/terms/root_pos": 7.5, "Reward/terms/action_rate": -1.5,
        "Episode/count": 2, "Episode/len_s": .05,
        "Episode/fail_frac": .5, "Episode/timeout_frac": .5,
        "Metrics/robot/root_xy_m": 29. / 6.,
        "Train/iter": 4, "Train/steps": 8, "Train/routes/robot_frac": 1.,
        "Perf/time/collect_s": 2., "Perf/time/learn_s": 3.,
        "Perf/time/wm_s": 2., "Perf/time/policy_s": 1., "Perf/fps": .8,
    }
    for key, value in expected.items():
        assert math.isclose(first[key], value, rel_tol=1e-6, abs_tol=1e-7), (key, first[key], value)
    assert second["Episode/count"] == 0
    assert "Reward/total" not in second and "Episode/len_s" not in second
    assert second["Train/steps"] == 10 and second["Perf/time/collect_s"] == .3
    assert second["Perf/fps"] == 1.
    initial = normalize({"Train/iter": 1, "Perf/collect_s": .2, "Loss/wm": None}, 0)
    assert initial["Perf/time/collect_s"] == .2 and "Loss/wm" not in initial
    definitions = metric_definitions(reward_manager.active_terms)
    assert definitions["Train/wm_grad_norm"] == definitions["Train/grad/wm"]
    assert "after native clipping" in definitions["Train/grad/wm"]["reduction"]
    assert all(len(key.split("/")) <= 3 and len(key) <= 64 for key in definitions)
    print(json.dumps({"passed": True, "wandb_initialized": False,
                      "completed_episode_values_verified": expected,
                      "empty_interval_omits_episode_means": True,
                      "first_and_subsequent_timing_deltas_verified": True,
                      "partial_episodes_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
