"""Exact tiled Float64 nearest-neighbor search for small object clouds.

One GPU program per source point scans bounded target tiles. No N x M distance
matrix is allocated. Equal distances choose the smallest target index.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['n_target'])
def nearest_kernel(query, target, indices, squared_distances, radius_squared,
                   n_target, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(query + row * 3)
    y = tl.load(query + row * 3 + 1)
    z = tl.load(query + row * 3 + 2)
    best = tl.full((), float('inf'), tl.float64)
    best_index = tl.full((), 2147483647, tl.int32)
    offsets = tl.arange(0, BLOCK)
    for tile in range(tl.cdiv(n_target, BLOCK)):
        column = tile * BLOCK + offsets
        valid = column < n_target
        dx = x - tl.load(target + column * 3, valid, other=0.)
        dy = y - tl.load(target + column * 3 + 1, valid, other=0.)
        dz = z - tl.load(target + column * 3 + 2, valid, other=0.)
        distance = tl.where(valid, dx * dx + dy * dy + dz * dz, float('inf'))
        local_best = tl.min(distance, axis=0)
        local_index = tl.min(tl.where(distance == local_best, column, 2147483647), axis=0)
        replace = (local_best < best) | ((local_best == best) & (local_index < best_index))
        best_index = tl.where(replace, local_index, best_index)
        best = tl.minimum(best, local_best)
    accepted = best < tl.load(radius_squared)
    tl.store(indices + row, tl.where(accepted, best_index, -1))
    tl.store(squared_distances + row, tl.where(accepted, best, 0.))


def nearest(query, target, indices, distances, radius_squared, count):
    nearest_kernel[(len(query),)](query, target, indices, distances, radius_squared,
                                  count, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
