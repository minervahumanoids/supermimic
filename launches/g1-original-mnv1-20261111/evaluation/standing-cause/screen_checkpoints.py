"""Screen original frozen artifacts. Never change an actor or train a model."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import probe
from supermimic.runners.run_config import load_runner_cfg

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def main():
    torch.set_num_threads(4)
    cfg = load_runner_cfg(json.loads((WORK / 'stopped-by-user-20261112/launch-config.json').read_text()))
    contract = json.loads((WORK / 'evaluation-403000/plant-contract.json').read_text())
    original = dict(np.load(WORK / 'evaluation-403000/inputs/bones/robot/00.npz'))
    reports = []
    started = time.monotonic()
    for version in (25, 50, 175, 225, 250, 275, 325, 350, 375, 400):
        path = probe.checkpoint(version)
        for friction in (None, 1.0):
            label = f'v{version}_screen_' + ('nominal' if friction is None else 'friction_1')
            reports.append(probe.rollout(label, path, 'cpu', 'ieee', 'motion',
                original, contract, cfg, foot_friction=friction))
            probe.write(ROOT / 'checkpoint-screen.json', {
                'diagnostic_only': True, 'reports': reports,
                'elapsed_seconds': time.monotonic() - started,
                'optimizer_steps': 0, 'official_source_modified': False,
                'architecture_modified': False, 'new_training_started': False})


if __name__ == '__main__':
    main()
