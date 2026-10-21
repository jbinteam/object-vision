#!/usr/bin/env python3
"""Compare independent and previous-pose ICP on an identical synthetic sequence.

The generated asymmetric surface and full observations are not a live accuracy
benchmark. No messages are published and no camera is opened.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from object_pose_tracking.registration import FixedReferenceICP, ICPConfig, motion_delta
from object_pose_tracking.tracking import AcceptedPoseSeed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--frames',type=int,default=30)
    args=parser.parse_args()
    if not 2 <= args.frames <= 100:parser.error('frames must be between 2 and 100')
    rng=np.random.default_rng(124)
    direction=rng.normal(size=(1600,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    radius=1+.12*direction[:,0]*direction[:,1]+.08*direction[:,2]**3
    points=direction*radius[:,None]*[.07,.045,.03]+[.02,-.01,.008]
    report={'kind':'synthetic_full_surface_small_motion','frames':args.frames,'methods':{}}
    with tempfile.TemporaryDirectory(prefix='phase3-benchmark-') as directory:
        path=Path(directory)/'reference.ply'
        o3d.io.write_point_cloud(str(path),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
        for method in ('point_to_point','point_to_plane'):
            modes={}
            for tracking in (False,True):
                reg=FixedReferenceICP(path,ICPConfig(icp_type=method,voxel_size=0,icp_time_budget=2.))
                state=AcceptedPoseSeed(.75);rows=[]
                for index in range(args.frames):
                    expected=np.eye(4)
                    expected[:3,:3]=Rotation.from_euler('xyz',[3+index*.3,-4+index*.15,6+index*.5],degrees=True).as_matrix()
                    expected[:3,3]=[.1+index*.0007,-.03,.7]
                    observed=points@expected[:3,:3].T+expected[:3,3]
                    stamp=1_000_000_000+index*100_000_000
                    seed=state.initial_guess(stamp,stamp+10_000_000) if tracking else None
                    begin=time.perf_counter();result=reg.register(observed,initial_guess=seed)
                    elapsed=(time.perf_counter()-begin)*1000
                    trans,angle=motion_delta(expected,result.transform)
                    if tracking and result.accepted:state.accept(result.transform,stamp,stamp+20_000_000)
                    rows.append(dict(accepted=result.accepted,initial_guess='previous_pose' if seed is not None else 'centroid',
                                     iterations=result.iterations,icp_ms=result.icp_ms,total_ms=elapsed,
                                     translation_error_m=trans,rotation_error_rad=angle,reason=result.reason))
                samples=rows[1:]  # identical first acquisition is recorded separately
                modes['phase3' if tracking else 'phase2']=dict(
                    accepted=sum(r['accepted'] for r in rows),first_frame=rows[0],
                    subsequent={key:dict(median=float(np.median([r[key] for r in samples])),
                                         p95=float(np.percentile([r[key] for r in samples],95)))
                                for key in ('iterations','icp_ms','total_ms')},
                    max_translation_error_m=max(r['translation_error_m'] for r in rows),
                    max_rotation_error_rad=max(r['rotation_error_rad'] for r in rows),frames=rows)
            report['methods'][method]=modes
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text)
    print(json.dumps({method:{mode:{k:v for k,v in data.items() if k!='frames'} for mode,data in modes.items()}
                      for method,modes in report['methods'].items()},indent=2))


if __name__=='__main__':main()
