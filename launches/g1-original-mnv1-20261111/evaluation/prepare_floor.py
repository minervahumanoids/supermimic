"""Use the existing MDP support detector for constant PICO floor calibration."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

MDP=Path('/Users/mattia/Documents/ChatGPT/Retargeting/mdp')
sys.path.insert(0,str(MDP/'src'))
from motion_data_processing.metrics.human_support import compute_human_foot_support

root=Path(__file__).resolve().parent/'inputs/pico'
records=[]
for i in range(3):
    path=root/'human-frames'/f'{i:02d}.npz'
    with np.load(path) as archive:
        joints=archive['positions']
    result=compute_human_foot_support(SimpleNamespace(joints=joints,fps=50,metadata={'skeleton':'smpl24'}))
    floor=result.metrics['support_floor_z_m']
    assert isinstance(floor,float) and np.isfinite(floor)
    records.append({'clip_index':i,'floor_z_m':floor,'metrics':result.metrics,
                    'source_arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'original_human_modified':False})
receipt={'method':'MDP human_foot_support_v1, constant source-floor correction for G1 IK only',
         'source_file_sha256':hashlib.sha256((MDP/'src/motion_data_processing/metrics/human_support.py').read_bytes()).hexdigest(),
         'clips':records}
(root/'grounding.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({i:r['floor_z_m'] for i,r in enumerate(records)}))
