"""Voxel accumulation equivalence against the previous float64 add.at path."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from unittest.mock import patch

from object_pose_tracking.geometry import CloudConfig, _voxel_groups, preprocess


def legacy_preprocess(points, config, colors=None):
    """Frozen numerical reference: preserve accumulation and final filter behavior."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape or colors.dtype != np.uint8:
            raise ValueError('Colors must be uint8 N x 3 matching the points')
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    if config.voxel_size > 0 and len(points):
        keys = np.floor(points / config.voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        points = (sums / counts[:, None]).astype(np.float32)
        if colors is not None:
            color_sums = np.zeros((len(counts), 3), dtype=np.float64)
            np.add.at(color_sums, inverse, colors)
            colors = np.rint(color_sums / counts[:, None]).clip(0, 255).astype(np.uint8)
    if config.outlier_neighbors > 0 and len(points) > 2:
        k = min(config.outlier_neighbors + 1, len(points))
        distances, _ = cKDTree(points).query(points, k=k, workers=1)
        means = distances[:, 1:].mean(axis=1)
        keep = means <= means.mean() + config.outlier_std_ratio * means.std()
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    points = np.ascontiguousarray(points)
    return points if colors is None else (points, np.ascontiguousarray(colors))


def cases():
    rng = np.random.default_rng(3840)
    centers = rng.integers(-20, 21, (120, 3)).astype(np.float32) * .01
    duplicate_voxels = centers[rng.integers(0, len(centers), 2500)]
    duplicate_voxels += rng.uniform(.0001, .008, duplicate_voxels.shape).astype(np.float32)
    return {
        'duplicates_and_signed_keys': duplicate_voxels,
        'nonfinite': np.vstack((duplicate_voxels, [np.nan, 0, 1], [0, np.inf, 1], [-np.inf, 0, 1])),
        'outliers': np.vstack((duplicate_voxels, [10, 10, 10], [-10, -10, -10])),
        'all_nonfinite': np.array([[np.nan, 0, 1], [0, np.inf, 1], [0, 0, -np.inf]]),
        'empty': np.empty((0, 3)),
        'single_voxel': np.tile([-.005, .005, .005], (127, 1)),
        'wide_key_spans_fallback': np.array([[-1e7, -1e7, -1e7], [1e7, 1e7, 1e7],
                                              [-1e7, -1e7, -1e7], [0, 0, 0]]),
    }


@pytest.mark.parametrize('name', list(cases()))
@pytest.mark.parametrize('neighbors', [0, 3, 20])
@pytest.mark.parametrize('with_colors', [False, True])
def test_voxel_accumulation_exactly_matches_legacy(name, neighbors, with_colors):
    points = cases()[name]
    colors = np.random.default_rng(4817).integers(0, 256, points.shape, dtype=np.uint8) if with_colors else None
    original_points = points.copy()
    original_colors = None if colors is None else colors.copy()
    config = CloudConfig(voxel_size=.01, outlier_neighbors=neighbors)
    actual = preprocess(points, config, colors)
    expected = legacy_preprocess(points, config, colors)
    actual_arrays = actual if with_colors else (actual,)
    expected_arrays = expected if with_colors else (expected,)
    for result, reference in zip(actual_arrays, expected_arrays):
        np.testing.assert_array_equal(result, reference, strict=True)
        assert result.flags.c_contiguous
    np.testing.assert_array_equal(points, original_points)
    if with_colors:
        np.testing.assert_array_equal(colors, original_colors)


def test_signed_voxel_order_and_bankers_rounding_keep_color_pairs():
    points = np.array([[.001, 0, 1], [-.001, 0, 1], [.002, 0, 1], [-.002, 0, 1]], dtype=np.float32)
    colors = np.array([[0, 1, 254], [10, 11, 20], [1, 2, 255], [11, 12, 21]], dtype=np.uint8)
    actual_points, actual_colors = preprocess(points, CloudConfig(voxel_size=.01, outlier_neighbors=0), colors)
    np.testing.assert_array_equal(actual_points, np.array([[-.0015, 0, 1], [.0015, 0, 1]], np.float32))
    np.testing.assert_array_equal(actual_colors, [[10, 12, 20], [0, 2, 254]])


def test_disabling_voxelization_preserves_noncontiguous_input_and_outlier_behavior():
    points = cases()['outliers'][::2]
    colors = np.random.default_rng(19).integers(0, 256, (len(points) * 2, 3), dtype=np.uint8)[::2]
    config = CloudConfig(voxel_size=0, outlier_neighbors=20)
    actual = preprocess(points, config, colors)
    expected = legacy_preprocess(points, config, colors)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference, strict=True)


@pytest.mark.parametrize('keys', [
    np.array([[0, 0, 0]], dtype=np.int64),
    np.zeros((10, 3), dtype=np.int64),
    np.array([[2, -1, 0], [-2, 5, 1], [-2, -3, 1], [-2, -3, 0], [2, -1, 0]], dtype=np.int64),
    np.array([[0, 0, 0], [0, 0, 100], [0, 0, -100]], dtype=np.int64),
    np.array([[np.iinfo(np.int64).min, 0, 0], [np.iinfo(np.int64).min + 1, 0, 0]], dtype=np.int64),
    np.array([[np.iinfo(np.int64).max, 0, 0], [np.iinfo(np.int64).max - 1, 0, 0]], dtype=np.int64),
    np.array([[0, 0, 0], [np.iinfo(np.int64).max - 1, 0, 0]], dtype=np.int64),
    np.array([[0, 0, 0], [0, np.iinfo(np.int64).max - 1, 0]], dtype=np.int64),
    np.array([[0, 0, 0], [0, 0, np.iinfo(np.int64).max - 1]], dtype=np.int64),
    np.array([[np.iinfo(np.int64).min, 0, 0], [np.iinfo(np.int64).max, 0, 0]], dtype=np.int64),
    np.array([[0, 0, 0], [np.iinfo(np.int64).max, 0, 0]], dtype=np.int64),
    np.array([[0, 0, 0], [1 << 22, 1 << 22, 1 << 22]], dtype=np.int64),
])
def test_packed_voxel_keys_keep_exact_sorted_inverse_and_counts(keys):
    expected_keys, expected_inverse, expected_counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    inverse, counts = _voxel_groups(keys)
    np.testing.assert_array_equal(inverse, expected_inverse, strict=True)
    np.testing.assert_array_equal(counts, expected_counts, strict=True)
    # The returned group IDs must still select keys in signed XYZ lexical order.
    actual_keys = np.stack([keys[np.flatnonzero(inverse == group)[0]] for group in range(len(counts))])
    np.testing.assert_array_equal(actual_keys, expected_keys, strict=True)


def test_packed_key_overflow_check_uses_python_ints_and_falls_back_before_arithmetic():
    keys = np.array([[np.iinfo(np.int64).min] * 3, [np.iinfo(np.int64).max] * 3], dtype=np.int64)
    # The range span is 2**64 and its product is 2**192; NumPy integer arithmetic
    # would wrap both bounds and falsely admit a colliding packed representation.
    with patch('object_pose_tracking.geometry.np.unique', wraps=np.unique) as unique:
        _voxel_groups(keys)
    unique.assert_called_once()
    assert unique.call_args.kwargs['axis'] == 0


def test_random_signed_packed_voxel_membership_and_counts_match_legacy():
    rng = np.random.default_rng(1278)
    for _ in range(30):
        offsets = rng.integers(-1000000, 1000000, size=3)
        ranges = rng.integers(1, 100, size=3)
        keys = rng.integers(0, ranges, size=(400, 3), dtype=np.int64) + offsets
        _, expected_inverse, expected_counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        inverse, counts = _voxel_groups(keys)
        np.testing.assert_array_equal(inverse, expected_inverse, strict=True)
        np.testing.assert_array_equal(counts, expected_counts, strict=True)
