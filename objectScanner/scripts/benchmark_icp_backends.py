#!/usr/bin/env python3
"""Paired ICP timing on identical recorded or synthetic observations and seeds."""
import argparse
import cProfile
from dataclasses import replace
import io
import json
from pathlib import Path
import pstats
import numpy as np
from scipy.spatial.transform import Rotation
from object_pose_tracking.registration import ICPConfig, FixedReferenceICP, initial_rotation, motion_delta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path)
    parser.add_argument('--baseline',choices=['open3d','cached','cuda_legacy'],default='open3d')
    parser.add_argument('--candidate',choices=['cached','cuda'],default='cached')
    parser.add_argument('--reference',type=Path,default=Path('references/emergency_stop_ref/reference.ply'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-output',type=Path)
    args=parser.parse_args()
    if args.baseline==args.candidate:parser.error('Choose different baseline and candidate backends')
    names=(args.baseline,args.candidate)
    cfg=ICPConfig(icp_time_budget=2.)  # compare equal work, independent of wall-time cutoff
    engines={name:FixedReferenceICP(args.reference,replace(cfg,icp_backend=name)) for name in names}
    rotation=initial_rotation(args.reference,'reference_metadata')
    cases=[]
    labels=[]
    if args.recording:
        with np.load(args.recording,allow_pickle=False) as data:
            for i,seed in enumerate(data['seeds']):
                cases.append((data[f'cloud_{i}'].copy(),seed.copy() if np.isfinite(seed).all() else None))
                labels.append('recorded')
    else:
        points=engines[args.baseline].reference_points
        for i in range(30):
            t=np.eye(4);t[:3,:3]=rotation@Rotation.from_euler('xyz',[.1*i,.05*i,.2*i],degrees=True).as_matrix()
            t[:3,3]=[.1+.0004*i,-.03,.7]
            observed=points@t[:3,:3].T+t[:3,3]
            seed=t.copy();seed[:3,3]+=[.001,-.001,.001]
            cases.append((observed,seed))
            labels.append('full_warm')
        rng=np.random.default_rng(20260923)
        for i in range(15):
            t=np.eye(4);t[:3,:3]=rotation@Rotation.from_euler('xyz',[2+i*.3,-3,4+i*.5],degrees=True).as_matrix()
            t[:3,3]=[.1,-.03,.7]
            partial=points[points[:,0]>np.percentile(points[:,0],25)]
            observed=partial@t[:3,:3].T+t[:3,3]+rng.normal(0,.0007,partial.shape)
            seed=t.copy();seed[:3,3]+=[.002,-.001,.002]
            cases.append((observed,seed));labels.append('partial_noisy_warm')
            cases.append((observed,None));labels.append('partial_noisy_centroid')
    if not cases:raise ValueError('Recording contains no usable observations')
    for engine in engines.values():engine.register(cases[0][0],rotation=rotation,initial_guess=cases[0][1])
    rows=[];summaries={};profiles={}
    for repeat in range(2):
        for i,(points,seed) in enumerate(cases):
            results={}
            for name in (names if (i+repeat)%2==0 else names[::-1]):
                results[name]=engines[name].register(points,rotation=rotation,initial_guess=seed)
            baseline,cached=results[args.baseline],results[args.candidate]
            delta_m,delta_rad=motion_delta(baseline.transform,cached.transform)
            rows.append(dict(frame=i,scenario=labels[i],repeat=repeat,pose_difference_m=delta_m,pose_difference_rad=delta_rad,
                             **{name:dict(accepted=r.accepted,reason=r.reason,iterations=r.iterations,
                                          icp_ms=r.icp_ms,preparation_ms=r.preparation_ms,
                                          rmse_m=r.rmse if np.isfinite(r.rmse) else None,fitness=r.fitness)
                                for name,r in results.items()}))
    for name in engines:
        summaries[name]=dict(accepted=sum(row[name]['accepted'] for row in rows),
                            reasons={reason:sum(row[name]['reason']==reason for row in rows)
                                     for reason in sorted({row[name]['reason'] for row in rows})},
                            **{key:dict(median=float(np.median([row[name][key] for row in rows])),
                                        p95=float(np.percentile([row[name][key] for row in rows],95)))
                               for key in ('icp_ms','preparation_ms','iterations')})
        profiler=cProfile.Profile();profiler.enable()
        for index in np.linspace(0,len(cases)-1,min(10,len(cases)),dtype=int):
            points,seed=cases[index];engines[name].register(points,rotation=rotation,initial_guess=seed)
        profiler.disable();stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('tottime').print_stats(15)
        profiles[name]=stream.getvalue()
    same_outcomes=all(row[args.baseline]['accepted']==row[args.candidate]['accepted'] for row in rows)
    scenarios={label:{name:dict(
        accepted=sum(r[name]['accepted'] for r in rows if r['scenario']==label),
        attempts=sum(r['scenario']==label for r in rows),
        icp_median_ms=float(np.median([r[name]['icp_ms'] for r in rows if r['scenario']==label])),
        iterations_median=float(np.median([r[name]['iterations'] for r in rows if r['scenario']==label])))
        for name in engines} for label in sorted(set(labels))}
    both_accepted=[row for row in rows if row[args.baseline]['accepted'] and row[args.candidate]['accepted']]
    report=dict(input=str(args.recording) if args.recording else 'synthetic transforms of supplied reference',
                baseline=args.baseline,candidate=args.candidate,cases=len(cases),repeats=2,configuration=vars(cfg),summary=summaries,
                acceptance_agreement=same_outcomes,scenarios=scenarios,
                rejection_reason_agreement=all(r[args.baseline]['reason']==r[args.candidate]['reason'] for r in rows),
                max_accepted_pose_difference_m=max((r['pose_difference_m'] for r in both_accepted),default=None),
                max_accepted_pose_difference_rad=max((r['pose_difference_rad'] for r in both_accepted),default=None),rows=rows)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.profile_output:args.profile_output.write_text('\n'.join(name+'\n'+value for name,value in profiles.items()))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','configuration')},indent=2))


if __name__=='__main__':main()
