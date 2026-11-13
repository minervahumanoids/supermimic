"""Evaluate the frozen official G1 actor with standalone CPU MuJoCo."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from mjlab.sim import MujocoCfg
from supermimic.representations import create_representation
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor
from supermimic.tasks.task_spec import build_actor_obs_td

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
SOURCE='abc124c1bd5976341034668af3d12c087f2f1f55'
ACTOR_SHA='07e064bd4b6500b97a4a6b4af0060a801588abb8f0abdceb38138e2c40c0144b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def palms(model,data,ids,root):
    r=data.xmat[root].reshape(3,3)
    pos=(data.site_xpos[ids]-data.xpos[root])@r
    rotation=np.einsum('ij,njk->nik',r.T,data.site_xmat[ids].reshape(2,3,3))
    return pos.copy(),rotation.copy()


def evaluate(cohort,actor,representation,contract):
    inputs=ROOT/'inputs'/cohort
    selection=json.loads((inputs/'selection.json').read_text())
    index=json.loads((inputs/'robot/index.json').read_text())
    output=ROOT/'recordings'/cohort
    output.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(ROOT/'plant/scene.xml'))
    MujocoCfg(**contract['physics']).apply(model)
    assert model.opt.timestep*contract['decimation']==0.02
    root=int(model.body('robot/pelvis').id)
    sites=[int(model.site(n).id) for n in contract['palm_sites']]
    qa=np.array(contract['qpos_addresses']);va=np.array(contract['dof_addresses'])
    ai=np.array(contract['actuator_ids'])
    scale=np.array(contract['action_scale']);offset=np.array(contract['default_joint_pos'])
    limits=model.jnt_range[[int(model.joint(n).id) for n in contract['model_joint_names']]]
    midpoint=limits.mean(axis=1);half=(limits[:,1]-limits[:,0])*0.45
    trials=[]
    for clip_index,(clip,row) in enumerate(zip(selection['clips'],index['clips'],strict=True)):
        path=inputs/'robot'/row['path'];assert sha(path)==row['sha256']
        archive=dict(np.load(path,allow_pickle=False));n=clip['frames']
        motion=SimpleNamespace(**{name:torch.tensor(archive[name],dtype=torch.float32)
            for name in ['joint_pos','joint_vel','root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w']},
            motion_ends=torch.tensor([n]))
        data=mujoco.MjData(model);reference=mujoco.MjData(model)
        data.qpos[:3]=archive['root_pos_w'][0];data.qpos[3:7]=archive['root_quat_w'][0]
        data.qpos[qa]=np.clip(archive['joint_pos'][0],midpoint-half,midpoint+half)
        data.qvel[:3]=archive['root_lin_vel_w'][0]
        data.qvel[3:6]=Rotation.from_quat(data.qpos[3:7],scalar_first=True).inv().apply(archive['root_ang_vel_w'][0])
        data.qvel[va]=archive['joint_vel'][0]
        mujoco.mj_forward(model,data)
        records={key:[] for key in ['q','actual_root','actual_root_quat','reference_root',
                                   'reference_q','reference_frame','palm_cm','palm_deg','actions']}
        stop=None;pending=None
        for frame in range(n):
            reference.qpos[:3]=archive['root_pos_w'][frame]
            reference.qpos[3:7]=archive['root_quat_w'][frame]
            reference.qpos[qa]=archive['joint_pos'][frame]
            mujoco.mj_forward(model,reference)
            ap,ar=palms(model,data,sites,root);rp,rr=palms(model,reference,sites,root)
            records['q'].append(data.qpos[qa].copy())
            records['actual_root'].append(data.xpos[root].copy())
            records['actual_root_quat'].append(data.xquat[root].copy())
            records['reference_root'].append(archive['root_pos_w'][frame])
            records['reference_q'].append(archive['joint_pos'][frame])
            records['reference_frame'].append(frame)
            records['palm_cm'].append(np.linalg.norm(ap-rp,axis=1)*100)
            relative=np.einsum('nij,njk->nik',rr.transpose(0,2,1),ar)
            records['palm_deg'].append(Rotation.from_matrix(relative).magnitude()*180/np.pi)
            if pending is not None:
                stop={'reason':pending,'frame':frame,'time_s':frame/50}
                break
            if frame==n-1:
                break
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,root,velocity,0)
            state=torch.tensor(np.r_[data.xpos[root],data.xquat[root],data.qpos[qa],
                                    velocity[3:],velocity[:3],data.qvel[va]],dtype=torch.float32)[None]
            rep=representation.represent(state)
            _,goal=representation.compute_wm_goal(motion,torch.tensor([frame]),state[:,:7])
            obs=build_actor_obs_td({'motion':goal},rep[:,None])
            with torch.inference_mode():
                actions=actor(obs)[0].cpu().numpy().copy()
            assert np.isfinite(actions).all()
            records['actions'].append(actions)
            data.ctrl[ai]=offset+scale*actions
            for _ in range(contract['decimation']):mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop={'reason':'nonfinite simulator state','frame':frame,'time_s':frame/50,
                      'detected_source_frame':frame+1}
                break
            if abs(data.xpos[root,2]-archive['root_pos_w'][frame,2])>0.25:
                pending='native anchor height error > 0.25 m'
            elif abs(data.xmat[root].reshape(3,3)[2,2]-reference.xmat[root].reshape(3,3)[2,2])>0.8:
                pending='native projected gravity difference > 0.8'
        arrays={name:np.asarray(value) for name,value in records.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        filename=f'trial-{clip_index:04d}.npz'
        np.savez_compressed(output/filename,**arrays)
        drift=np.linalg.norm(arrays['actual_root'][:,:2]-arrays['reference_root'][:,:2],axis=1)*100
        trial={'clip':clip['clip'],'category':clip['category'],'route':'robot','condition':'clean',
            'repeat':0,'file':filename,'frames':len(arrays['q']),'requested_frames':n,
            'complete':stop is None and len(arrays['q'])==n,'stop':stop,
            'display_number':clip_index+1,'comparison_number':clip.get('comparison_number',clip_index+1),
            'root_xy_mean_cm':float(drift.mean()),'palm_position_mean_cm':float(arrays['palm_cm'].mean()),
            'palm_orientation_mean_deg':float(arrays['palm_deg'].mean()),
            'training_membership':clip['training_membership']}
        trials.append(trial)
        print(json.dumps({'cohort':cohort,**trial}),flush=True)
    report={'backend':'standalone CPU MuJoCo','native_common_plant':False,
        'source_identity_verified':True,'source_commit':SOURCE,'iteration':403000,
        'policy_sha256':ACTOR_SHA,'model_sha256':contract['model_sha256'],
        'plant_contract_sha256':sha(ROOT/'plant-contract.json'),'optimizer_steps':0,
        'failure':None,'smoke':False,'input_selection_sha256':sha(inputs/'selection.json'),
        'authoritative_selection_sha256':selection['authoritative_selection_sha256'],
        'clean':True,'control_dt':0.02,'original_future_goal_window':True,
        'palm_frames':contract['palm_sites'],
        'termination_definition':'Original G1 height and gravity gates, checked after each control transition against its pre-step reference',
        'metric_definition':'World root XY distance. Palm position and rotation-angle errors relative to the pelvis, mean across hands.',
        'trials':trials}
    write(output/'report.json',report)
    return report


def main():
    torch.set_num_threads(4)
    assert json.loads((ROOT/'references-ready.json').read_text())['passed']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK/'supermimic',text=True).strip()==SOURCE
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=WORK/'supermimic',text=True)
    path=WORK/'stopped-by-user-20261112/checkpoint-v403/actor.pt'
    assert sha(path)==ACTOR_SHA
    contract=json.loads((ROOT/'plant-contract.json').read_text())
    saved_wm=torch.load(WORK/'stopped-by-user-20261112/checkpoint-v403/world_model.pt',
                        map_location='cpu',weights_only=True)['dynamics']
    for key,values in [('action_scale',contract['action_scale']),
                       ('action_offset',contract['default_joint_pos'])]:
        torch.testing.assert_close(saved_wm[key].reshape(-1),torch.tensor(values),rtol=0,atol=1e-6)
    write(ROOT/'action-contract-parity.json',{'passed':True,'source':'Original saved world model buffers',
        'action_scale_verified':True,'action_offset_verified':True,'atol':1e-6})
    config=load_runner_cfg(json.loads((WORK/'stopped-by-user-20261112/launch-config.json').read_text()))
    representation=create_representation('gencoord',71,29,0.02,contract['decimation'],
                                         torch.tensor(contract['default_joint_pos']),actor_obs_mode=config.actor_obs)
    assert config.actor_obs=='rep_state' and config.history_length==1
    actor=build_actor(config,SimpleNamespace(representation=representation,num_goals=670,num_actions=29),'cpu')
    actor.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    actor.eval().requires_grad_(False)
    reports={cohort:evaluate(cohort,actor,representation,contract) for cohort in ('bones','pico')}
    assert sha(path)==ACTOR_SHA
    summary={cohort:{'clips':len(r['trials']),'complete':sum(t['complete'] for t in r['trials']),
        'requested_frames':sum(t['requested_frames'] for t in r['trials']),
        'recorded_frames':sum(t['frames'] for t in r['trials']),
        'mean_clip_root_xy_cm':float(np.mean([t['root_xy_mean_cm'] for t in r['trials']])),
        'mean_clip_palm_position_cm':float(np.mean([t['palm_position_mean_cm'] for t in r['trials']])),
        'mean_clip_palm_orientation_deg':float(np.mean([t['palm_orientation_mean_deg'] for t in r['trials']]))}
        for cohort,r in reports.items()}
    write(ROOT/'evaluation-summary.json',{'iteration':403000,'backend':'standalone CPU MuJoCo',
        'cohorts':summary,'completed_utc':time.time(),'policy_modified':False,'optimizer_steps':0})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
