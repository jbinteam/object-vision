import json
from dataclasses import replace
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation
from object_pose_tracking.registration import (
    ICPConfig, FixedReferenceICP, centroid_initial_guess, initial_rotation, is_rigid, motion_delta)

CUDA_BACKEND = pytest.param('cuda', marks=pytest.mark.skipif(
    not o3d.core.cuda.is_available(), reason='CUDA Open3D/device required'))


def asymmetric_surface():
    rng = np.random.default_rng(124)
    v = rng.normal(size=(1600, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    radii = 1 + .12 * v[:, 0] * v[:, 1] + .08 * v[:, 2] ** 3
    return v * radii[:, None] * [.07, .045, .03] + [.02, -.01, .008]


def save_reference(path, points):
    assert o3d.io.write_point_cloud(str(path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))


@pytest.fixture
def reference(tmp_path):
    points = asymmetric_surface()
    path = tmp_path / 'reference.ply'
    save_reference(path, points)
    return path, points


def matrix(angles=(0, 0, 0), translation=(0, 0, 0)):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    transform[:3, 3] = translation
    return transform


@pytest.mark.parametrize('method', ['point_to_point', 'point_to_plane'])
@pytest.mark.parametrize('angles,translation', [((0,0,0),(0,0,0)), ((0,0,0),(.12,-.08,.7)),
                                             ((5,-8,12),(0,0,0)), ((5,-8,12),(.12,-.08,.7))])
def test_absolute_transform_and_reference_immutability(reference, method, angles, translation):
    path, points = reference
    expected = matrix(angles, translation)
    observed = points @ expected[:3,:3].T + expected[:3,3]
    registration = FixedReferenceICP(path, ICPConfig(icp_type=method, voxel_size=0, icp_time_budget=2.0))
    before = registration.reference_points.copy()
    result = registration.register(observed)
    assert result.accepted, result
    error_m, error_rad = motion_delta(expected, result.transform)
    assert error_m < 1e-5 and error_rad < 1e-3, (error_m, error_rad)
    np.testing.assert_array_equal(registration.reference_points, before)
    np.testing.assert_array_equal(np.asarray(registration._reference.points), before)
    assert not registration.reference_points.flags.writeable
    assert result.converged and result.correspondences == len(points)


@pytest.mark.parametrize('partial', [False, True])
def test_noise_and_partial_observation_with_controlled_seed(reference, partial):
    path, points = reference
    expected = matrix((15,-10,35),(.12,-.08,.7))
    rng = np.random.default_rng(30)
    visible = points[points[:,0] > -.025] if partial else points
    observed = visible @ expected[:3,:3].T + expected[:3,3] + rng.normal(0,.0002,visible.shape)
    reg = FixedReferenceICP(path, ICPConfig(icp_type='point_to_plane', voxel_size=0,
                                           icp_max_correspondence_distance=.008, icp_time_budget=2.))
    seed = expected.copy(); seed[:3,3] += [.001,-.001,.001]
    result = reg.register(observed, initial_guess=seed)
    assert result.accepted, result
    error_m, error_rad = motion_delta(expected, result.transform)
    assert error_m < .003 and error_rad < .06, (error_m,error_rad)


def test_centroid_seed_respects_nonidentity_rotation():
    source = np.array([[1,2,3],[2,3,4],[4,3,1]],float)
    expected = matrix((10,20,80),(.5,.3,.7))
    target = source @ expected[:3,:3].T + expected[:3,3]
    np.testing.assert_allclose(centroid_initial_guess(source,target,expected[:3,:3]), expected, atol=1e-12)


def test_metadata_is_orientation_hint_only(reference):
    path, points = reference
    expected = matrix((30,40,50),(100,200,300))
    path.with_name('frame.json').write_text(json.dumps({'units':'m','T_scan_obj':expected.tolist()}))
    np.testing.assert_allclose(initial_rotation(path,'reference_metadata'),expected[:3,:3])
    np.testing.assert_allclose(initial_rotation(path,'manual',(30,40,50)),expected[:3,:3])
    np.testing.assert_array_equal(initial_rotation(path),np.eye(3))


@pytest.mark.parametrize('backend',['cached','open3d',CUDA_BACKEND])
@pytest.mark.parametrize('method',['point_to_point','point_to_plane'])
def test_invalid_no_correspondences_and_limits(reference,backend,method):
    path, points = reference
    cfg = ICPConfig(icp_type=method,icp_backend=backend,voxel_size=0,icp_time_budget=2.)
    reg = FixedReferenceICP(path,cfg)
    for bad in [np.empty((0,3)), np.full((40,3),np.nan), np.zeros((40,3)), np.arange(40)[:,None]*np.ones((1,3))]:
        assert not reg.register(bad).accepted
    assert reg.register(points, initial_guess=matrix(translation=(10,10,10))).reason == 'insufficient_correspondences'
    bad = np.eye(4); bad[0,0] = 2
    assert reg.register(points,initial_guess=bad).reason == 'invalid_initial_guess'
    limited = FixedReferenceICP(path,replace(cfg,icp_max_iterations=1))
    assert not limited.register(points @ matrix((5,-8,12))[:3,:3].T).accepted
    timed = FixedReferenceICP(path,replace(cfg,icp_time_budget=1e-9))
    assert timed.register(points).reason == 'time_budget_exceeded'


@pytest.mark.parametrize('backend',['cached','open3d',CUDA_BACKEND])
def test_underconstrained_plane_is_rejected(tmp_path,backend):
    x,y=np.meshgrid(np.linspace(-.05,.05,15),np.linspace(-.05,.05,15))
    points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    path=tmp_path/'plane.pcd';save_reference(path,points)
    reg=FixedReferenceICP(path,ICPConfig(icp_backend=backend,voxel_size=0,normal_radius=.025,icp_time_budget=2.))
    result=reg.register(points+[0,0,.7])
    assert not result.accepted and result.reason=='underconstrained_point_to_plane'


def test_wrong_reference_rejected_and_rigid_checks(tmp_path):
    with pytest.raises(ValueError): FixedReferenceICP(tmp_path/'missing.ply')
    path=tmp_path/'bad.ply';save_reference(path,np.zeros((40,3)))
    with pytest.raises(ValueError): FixedReferenceICP(path)
    reflection=np.eye(4);reflection[0,0]=-1
    assert not is_rigid(reflection)
    assert not is_rigid(np.full((4,4),np.nan))


@pytest.mark.parametrize('backend',['cached','open3d',CUDA_BACKEND])
@pytest.mark.parametrize('method',['point_to_point','point_to_plane'])
def test_quality_gates_reject_noisy_fit(reference,backend,method):
    path,points=reference
    rng=np.random.default_rng(20)
    observed=points+rng.normal(0,.001,points.shape)
    reg=FixedReferenceICP(path,ICPConfig(icp_backend=backend,voxel_size=0,icp_type=method,rmse_threshold=1e-6,icp_time_budget=2.))
    result=reg.register(observed)
    assert not result.accepted and result.reason=='high_rmse'


@pytest.mark.parametrize('method', ['point_to_point','point_to_plane'])
def test_small_motion_sequence_uses_fixed_reference_and_previous_raw_pose(reference,method):
    from object_pose_tracking.tracking import AcceptedPoseSeed
    path,points=reference
    reg=FixedReferenceICP(path,ICPConfig(voxel_size=0,icp_type=method,icp_time_budget=2.))
    state=AcceptedPoseSeed(.75)
    before=reg.reference_points.copy()
    previous=None
    for index in range(15):
        expected=matrix((3+index*.3,-4+index*.15,6+index*.5),(.1+index*.0007,-.03,.7))
        observation=points@expected[:3,:3].T+expected[:3,3]
        stamp=1_000_000_000+index*100_000_000
        seed=state.initial_guess(stamp,stamp+10_000_000)
        if previous is None:assert seed is None
        else:np.testing.assert_array_equal(seed,previous)
        result=reg.register(observation,initial_guess=seed)
        assert result.accepted,result
        translation,rotation=motion_delta(expected,result.transform)
        assert translation<1e-5 and rotation<1e-3,(index,translation,rotation)
        state.accept(result.transform,stamp,stamp+20_000_000)
        previous=result.transform.copy()
    np.testing.assert_array_equal(reg.reference_points,before)
    np.testing.assert_array_equal(np.asarray(reg._reference.points),before)


@pytest.mark.parametrize('method',['point_to_point','point_to_plane'])
@pytest.mark.parametrize('partial',[False,True])
def test_cached_backend_matches_open3d_updates_and_quality(reference,method,partial):
    path,points=reference
    expected=matrix((5,-8,12),(.12,-.08,.7))
    rng=np.random.default_rng(40)
    source=points[points[:,0]>-.03] if partial else points
    observed=source@expected[:3,:3].T+expected[:3,3]+rng.normal(0,.0001,source.shape)
    seed=expected.copy();seed[:3,3]+=[.001,-.001,.001]
    common=ICPConfig(icp_type=method,voxel_size=0,icp_time_budget=2.,icp_max_correspondence_distance=.008)
    baseline=FixedReferenceICP(path,replace(common,icp_backend='open3d')).register(observed,initial_guess=seed)
    cached=FixedReferenceICP(path,replace(common,icp_backend='cached')).register(observed,initial_guess=seed)
    assert cached.accepted==baseline.accepted and cached.reason==baseline.reason
    translation,angle=motion_delta(baseline.transform,cached.transform)
    assert translation<1e-7 and angle<1e-6,(translation,angle)
    assert cached.correspondences==baseline.correspondences
    assert cached.fitness==pytest.approx(baseline.fitness,abs=1e-8)
    assert cached.rmse==pytest.approx(baseline.rmse,abs=1e-8)


def test_cached_workspace_reuses_search_and_preserves_inputs(reference,monkeypatch):
    from object_pose_tracking import icp_workspace
    path,points=reference
    tree_factory=icp_workspace.build_search_index;built=[]
    def count_tree(*args,**kwargs):
        built.append(1);return tree_factory(*args,**kwargs)
    monkeypatch.setattr(icp_workspace,'build_search_index',count_tree)
    reg=FixedReferenceICP(path,ICPConfig(icp_type='point_to_point',voxel_size=0,icp_time_budget=2.))
    pose=matrix((5,-8,12),(.1,.1,.7));observed=points@pose[:3,:3].T+pose[:3,3]
    before=observed.copy();result=reg.register(observed)
    assert result.accepted and result.iterations>1 and len(built)==1
    np.testing.assert_array_equal(observed,before)
    np.testing.assert_array_equal(np.asarray(reg._reference.points),points)


@pytest.mark.skipif(not o3d.core.cuda.is_available(), reason='CUDA Open3D/device required')
@pytest.mark.parametrize('method',['point_to_point','point_to_plane'])
@pytest.mark.parametrize('partial',[False,True])
def test_cuda_recovers_pose_and_preserves_gpu_reference(reference,method,partial):
    path,points=reference
    expected=matrix((5,-8,12),(.12,-.08,.7))
    source=points[points[:,0]>-.03] if partial else points
    rng=np.random.default_rng(40)
    observation=source@expected[:3,:3].T+expected[:3,3]+rng.normal(0,.0001,source.shape)
    seed=expected.copy();seed[:3,3]+=[.001,-.001,.001]
    before=observation.copy()
    cfg=ICPConfig(icp_backend='cuda',icp_type=method,voxel_size=0,icp_time_budget=3.,
                  icp_max_correspondence_distance=.008)
    gpu=FixedReferenceICP(path,cfg)
    gpu_reference=gpu.cuda_reference
    workspace=gpu.cuda_plane
    baseline=FixedReferenceICP(path,replace(cfg,icp_backend='cached')).register(observation,initial_guess=seed)
    for _ in range(2):
        result=gpu.register(observation,initial_guess=seed)
        assert result.accepted and baseline.accepted,(result,baseline)
        distance,angle=motion_delta(baseline.transform,result.transform)
        assert distance<1e-4 and angle<1e-3,(distance,angle)
        assert result.fitness==pytest.approx(baseline.fitness,abs=.002)
        assert result.rmse==pytest.approx(baseline.rmse,abs=1e-5)
        assert gpu.cuda_reference is gpu_reference
        assert gpu.cuda_plane is workspace
        if workspace is not None:
            assert workspace.rank_fallbacks == 0
    np.testing.assert_array_equal(gpu.cuda_reference.cpu().numpy(),points)
    np.testing.assert_array_equal(gpu.reference_points,points)
    np.testing.assert_array_equal(observation,before)
    if workspace is not None:
        np.testing.assert_array_equal(workspace.reference.cpu().numpy(),points)


def test_cuda_unavailable_fails_explicitly(reference,monkeypatch):
    monkeypatch.setattr(o3d.core.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA:0 is unavailable'):
        FixedReferenceICP(reference[0],ICPConfig(icp_backend='cuda'))
    with pytest.raises(ValueError,match='nonnegative integer'):
        ICPConfig(icp_cuda_device=-1)


@pytest.mark.skipif(not o3d.core.cuda.is_available(), reason='CUDA required')
def test_cuda_observability_fast_path_preserves_direct_svd_boundary(reference):
    import torch
    path,points=reference
    engine=FixedReferenceICP(path,ICPConfig(icp_backend='cuda',voxel_size=0)).cuda_plane
    engine.positions.copy_(torch.as_tensor(points,device=engine.device))
    engine.indices.copy_(torch.arange(len(points),device=engine.device))
    rng=np.random.default_rng(58)
    xy=rng.normal(size=(len(points),2));seen=set()
    for scale in (0.,1e-8,1e-6,1e-4,1.):
        normal=np.column_stack((xy*scale,np.ones(len(points))))
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        engine.normals[:len(points)].copy_(torch.as_tensor(normal,device=engine.device))
        engine.metrics_graph.replay()
        rank=engine.status.cpu().numpy()[7];seen.add(rank)
        jacobian=np.column_stack((np.cross(points-points.mean(axis=0),normal),normal))
        singular=np.linalg.svd(jacobian,compute_uv=False)
        expected=singular[-1]>singular[0]*1e-6
        if rank==1:
            assert expected  # GPU sufficient condition must never relax the gate.
        else:
            fallback=np.linalg.svd(engine.centered_jacobian.cpu().numpy(),compute_uv=False)
            assert (fallback[-1]>fallback[0]*1e-6)==expected
    assert seen=={0.,1.}


@pytest.mark.skipif(not o3d.core.cuda.is_available(), reason='CUDA required')
def test_cuda_switches_search_size_without_retaining_old_target(reference):
    path,points=reference
    cfg=ICPConfig(icp_backend='cuda',voxel_size=0,icp_time_budget=3.)
    engine=FixedReferenceICP(path,cfg)
    seed=matrix(translation=(.1,-.03,.7))
    small=points+seed[:3,3]
    large=np.tile(small,(6,1))
    for cloud in (small,large,small):
        result=engine.register(cloud,initial_guess=seed)
        assert result.accepted,result
        assert engine.cuda_plane.search is None
        distance,angle=motion_delta(seed,result.transform)
        assert distance<1e-4 and angle<1e-3
