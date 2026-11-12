"""Prepare the original G1 plant and matched references for the approved cohorts."""

import contextlib
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import types

import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,str(WORK))
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader
from supermimic.scripts.train import TrainConfig
from convert_bones import BonesMotion


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def positions(mapping, default=0):
    result=np.full(29,default,dtype=np.float64)
    for pattern,value in mapping.items():
        for i,name in enumerate(JOINT_NAMES):
            if re.fullmatch(pattern,name): result[i]=value
    return result


def save_reference(folder,index,clip,arrays,provenance):
    assert arrays['joint_pos'].shape==(clip['frames'],29)
    assert all(np.isfinite(arrays[k]).all() for k in
               ['joint_pos','joint_vel','root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w'])
    path=folder/'robot'/f'{index:02d}.npz'
    np.savez_compressed(path,**arrays,root_pos=arrays['root_pos_w'],
                        root_quat_wxyz=arrays['root_quat_w'],source_time_s=np.arange(clip['frames'])/50)
    return {'clip':clip['clip'],'path':path.name,'frames':clip['frames'],
            'sha256':sha(path),'provenance':provenance}


def derive(pose):
    motion=MotionLoader.__new__(MotionLoader)
    motion.motion_base_poss=torch.tensor(pose[:,:3],dtype=torch.float32)
    motion.motion_base_rots=torch.tensor(pose[:,3:7],dtype=torch.float32)
    motion.motion_dof_poss=torch.tensor(pose[:,7:],dtype=torch.float32)
    motion.output_dt=0.02
    motion.output_fps=50
    motion._compute_velocities()
    return motion.arrays()


def main():
    torch.set_num_threads(1)
    cfg=TrainConfig.from_task('SuperMimic-GenCoord-G1').env
    cfg.scene.num_envs=1
    scene=Scene(cfg.scene,device='cpu')
    plant=ROOT/'plant'
    scene.write(plant)
    model=mujoco.MjModel.from_xml_path(str(plant/'scene.xml'))
    cfg.sim.mujoco.apply(model)
    assert (model.nq,model.nv,model.nu)==(36,35,29)
    default=positions(cfg.scene.entities['robot'].init_state.joint_pos)
    scale=positions(cfg.actions['joint_pos'].scale,default=1)
    model_joints=['robot/'+name for name in JOINT_NAMES]
    addresses=[int(model.joint(n).qposadr[0]) for n in model_joints]
    dofs=[int(model.joint(n).dofadr[0]) for n in model_joints]
    qpos_to_joint={int(model.jnt_qposadr[model.actuator_trnid[a,0]]):a for a in range(model.nu)}
    actuator_ids=[qpos_to_joint[a] for a in addresses]
    contract={'robot':'Unitree G1','joint_names':list(JOINT_NAMES),'model_joint_names':model_joints,
        'qpos_addresses':addresses,'dof_addresses':dofs,'actuator_ids':actuator_ids,
        'default_joint_pos':default.tolist(),'action_scale':scale.tolist(),
        'physics':asdict(cfg.sim.mujoco),'decimation':cfg.decimation,'control_dt':0.02,
        'model_sha256':sha(plant/'scene.xml'),
        'asset_sha256':{str(p.relative_to(plant)):sha(p) for p in plant.rglob('*') if p.is_file()},
        'palm_sites':['robot/left_palm','robot/right_palm'],
        'source_commit':'abc124c1bd5976341034668af3d12c087f2f1f55',
        'mjlab_commit':'dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd',
        'clean_evaluation':True,'randomization':False}
    write(ROOT/'plant-contract.json',contract)
    bones=ROOT/'inputs/bones'
    selection=json.loads((bones/'selection.json').read_text())
    (bones/'robot').mkdir(exist_ok=True)
    missing={c['g1_source_name'] for c in selection['clips']
             if not (WORK/'data/npz'/f"{c['g1_source_name']}.npz").exists()}
    recovered={}
    if missing:
        with tarfile.open(WORK/'data/g1.tar.gz',mode='r|gz') as archive:
            for member in archive:
                name=Path(member.name).stem
                if member.isfile() and name in missing:
                    raw=archive.extractfile(member).read()
                    with contextlib.redirect_stdout(io.StringIO()):
                        recovered[name]=BonesMotion(name,raw,120).arrays()
                    if len(recovered)==len(missing): break
        assert set(recovered)==missing
    rows=[]
    for i,clip in enumerate(selection['clips']):
        name=clip['g1_source_name']
        path=WORK/'data/npz'/f'{name}.npz'
        full=dict(np.load(path,allow_pickle=False)) if path.exists() else recovered[name]
        lo=clip['g1_start_frame']; hi=lo+clip['frames']
        assert hi<=len(full['joint_pos']),(name,lo,hi,len(full['joint_pos']))
        arrays={k:(v[lo:hi].copy() if k not in ('fps','joint_names') else v)
                for k,v in full.items()}
        rows.append(save_reference(bones,i,clip,arrays,{
            'kind':'original BONES-SEED G1 release','source_name':name,
            'source_start_frame':lo,'source_fps':50,
            'source_archive_sha256':'52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a',
            'excluded_from_training':not clip['training_membership']}))
        print(json.dumps({'bones_reference':i+1,'clip':clip['clip']}),flush=True)
    write(bones/'robot/index.json',{'joint_names':model_joints,'clips':rows})

    # Retarget directly from the retained SMPL joint poses. No body-model fit or resampling.
    package=types.ModuleType('gmr_eval_backend')
    package.__path__=[str(ROOT/'gmr/general_motion_retargeting')]
    sys.modules['gmr_eval_backend']=package
    GMR=importlib.import_module('gmr_eval_backend.motion_retarget').GeneralMotionRetargeting
    original_ik=ROOT/'gmr/general_motion_retargeting/ik_configs/smplx_to_g1.json'
    ik=json.loads(original_ik.read_text())
    ik['robot_root_name']='robot/'+ik['robot_root_name']
    # Keep the pinned GMR toe frames on the exact official G1 kinematic plant.
    ik_spec=mujoco.MjSpec.from_file(str(plant/'scene.xml'))
    for side in ('left','right'):
        ik_spec.body(f'robot/{side}_ankle_roll_link').add_body(
            name=f'robot/{side}_toe_link',pos=[0.1,0,-0.02])
    for key in ('ik_match_table1','ik_match_table2'):
        ik[key]={'robot/'+name:value for name,value in ik[key].items()}
    (plant/'ik.xml').write_text(ik_spec.to_xml())
    write(plant/'ik.json',ik)
    pico=ROOT/'inputs/pico'
    selection=json.loads((pico/'selection.json').read_text())
    floor=json.loads((pico/'grounding.json').read_text())
    names=['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
           'left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
           'left_collar','right_collar','head','left_shoulder','right_shoulder',
           'left_elbow','right_elbow','left_wrist','right_wrist']
    (pico/'robot').mkdir(exist_ok=True)
    rows=[]
    for i,clip in enumerate(selection['clips']):
        frames=np.load(pico/'human-frames'/f'{i:02d}.npz')
        points=frames['positions'].copy()
        points[:,:,2]-=floor['clips'][i]['floor_z_m']
        rotations=frames['rotations_xyzw'][:,:22,[3,0,1,2]]
        retarget=GMR('smplx','unitree_g1',verbose=False,
                     robot_xml_path=str(plant/'ik.xml'),ik_config_path=str(plant/'ik.json'))
        order=[int(retarget.model.joint(name).qposadr[0]) for name in model_joints]
        poses=[]
        for frame in range(clip['frames']):
            human={name:(points[frame,j].copy(),rotations[frame,j].copy()) for j,name in enumerate(names)}
            q=retarget.retarget(human)
            poses.append(np.r_[q[:7],q[order]])
            if frame%500==0:
                print(json.dumps({'pico_clip':i+1,'retargeted_frame':frame,
                                  'frames':clip['frames']}),flush=True)
        arrays=derive(np.asarray(poses))
        rows.append(save_reference(pico,i,clip,arrays,{
            'kind':'MDP pinned GMR on original SMPL poses','gmr_commit':'af90c22c64759568c99e5e39bc1ced50eb7ef25c',
            'ik_original_sha256':sha(original_ik),'ik_adapted_sha256':sha(plant/'ik.json'),
            'source_human_sha256':clip['original_human_sha256'],
            'source_floor_z_m':floor['clips'][i]['floor_z_m'],
            'floor_method':floor['method'],'original_source_modified':False,
            'source_clock_preserved':True,'resampled':False}))
    write(pico/'robot/index.json',{'joint_names':model_joints,'clips':rows})
    write(ROOT/'references-ready.json',{'bones':30,'pico':3,'passed':True,
        'model_sha256':contract['model_sha256'],'plant_contract_sha256':sha(ROOT/'plant-contract.json')})
    print(json.dumps({'all_references_ready':True}),flush=True)


if __name__=='__main__': main()
