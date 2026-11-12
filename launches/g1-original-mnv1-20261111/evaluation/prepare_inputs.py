"""Freeze the approved source cohorts for the original G1 evaluation."""

import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for cohort in ('bones', 'pico'):
    source = (PROJECT/'experiments/master-baseline-review-20261104/evaluation-selection.json'
              if cohort == 'bones' else
              PROJECT/'experiments/gaussian64k-comparison-20261104/pico-inputs/selection.json')
    selection = json.loads(source.read_text())
    target = ROOT/'inputs'/cohort
    (target/'original-human').mkdir(parents=True, exist_ok=True)
    (target/'human-frames').mkdir(exist_ok=True)
    rows = []
    for i, original in enumerate(selection['clips']):
        clip = dict(original)
        origin = (Path(clip['input_root']) if cohort == 'bones' else
                  PROJECT/'experiments/gaussian64k-comparison-20261104/pico-inputs')
        human = origin/clip['original_human']
        assert human.is_file(), human
        copied = target/'original-human'/f'{i:02d}.pkl'
        shutil.copyfile(human, copied)
        data = joblib.load(copied)
        assert len(data['smpl_joints']) == clip['frames']
        n = clip['frames']
        pose = np.asarray(data['pose_aa']).reshape(n,24,3)
        rotations = np.zeros((n,24,4))
        up = Rotation.from_euler('x',90,degrees=True)
        for joint, parent in enumerate(PARENTS):
            local = Rotation.from_rotvec(pose[:,joint])
            world = up*local if parent < 0 else Rotation.from_quat(rotations[:,parent])*local
            rotations[:,joint] = world.as_quat()
        joints = np.asarray(data['smpl_joints'])
        pelvis = joints[:,0]
        world = joints-pelvis[:,None]+up.apply(np.asarray(data['transl'])+pelvis)[:,None]
        np.savez(target/'human-frames'/f'{i:02d}.npz', positions=world,
                 rotations_xyzw=rotations, source_time_s=np.arange(n)/50)
        excluded_source = clip.get('source', {}).get('source_clip') in {
            'faint_stand_up_lying_side_R_001__A493',
            'faint_stand_up_lying_side_R_001__A493_M'}
        clip.update(original_human=str(copied.relative_to(target)),
                    original_human_sha256=sha(copied), display_number=i+1,
                    training_membership=(cohort=='bones' and not excluded_source))
        if cohort == 'bones':
            source_name=clip['source']['source_clip']
            start=clip['source'].get('export_interval',clip['source']['source_interval'])['start_s']
            assert abs(start*50-round(start*50))<1e-6
            clip['g1_source_name']=source_name
            clip['g1_start_frame']=round(start*50)
        rows.append(clip)
    assert len(rows)==(30 if cohort=='bones' else 3)
    selection.update(clips=rows, authoritative_selection_sha256=sha(source),
                     robot='Unitree G1', source_clocks_preserved=True)
    (target/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps({'bones':30,'pico':3,'original_human_preserved':True}))
