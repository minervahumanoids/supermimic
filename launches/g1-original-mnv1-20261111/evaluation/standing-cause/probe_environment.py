"""Check the published benchmark friction and training arithmetic without learning."""
import json
from pathlib import Path
import numpy as np
import torch
import probe
from supermimic.runners.run_config import load_runner_cfg

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def main():
    torch.set_num_threads(4)
    cfg = load_runner_cfg(json.loads((WORK/'stopped-by-user-20261112/launch-config.json').read_text()))
    contract = json.loads((WORK/'evaluation-403000/plant-contract.json').read_text())
    original = dict(np.load(WORK/'evaluation-403000/inputs/bones/robot/00.npz'))
    paths = {100: ROOT/'checkpoints/v100/actor.pt', 403: WORK/'stopped-by-user-20261112/checkpoint-v403/actor.pt'}
    results = []
    for version in (100, 403):
        results.append(probe.rollout(f'v{version}_friction_1', paths[version], 'cpu', 'ieee',
            'motion', original, contract, cfg, foot_friction=1.0))
    results.append(probe.rollout('v403_tf32_batch32', paths[403], 'cuda:0', 'tf32',
        'motion', original, contract, cfg, inference_batch=32))
    for version in (75, 125, 150):
        path = probe.checkpoint(version)
        results.append(probe.rollout(f'v{version}_cpu', path, 'cpu', 'ieee', 'motion', original, contract, cfg))
    probe.write(ROOT/'environment-probes.json', {'diagnostic_only':True,'reports':results,
        'official_source_modified':False,'architecture_modified':False,'optimizer_steps':0})


if __name__ == '__main__':
    main()
