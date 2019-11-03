import numpy as np
from scipy import stats

def norm_test(data, axis):
    return stats.normaltest(data,axis).pvalue

def voxel_dist_operations(data, operations=[np.max, np.mean, np.var, np.min, np.median]):
    voxel_dist = np.stack(data, axis=-1)
    # apply each operation on each voxel distribution and create result matrices
    return {op.__name__:np.squeeze(np.apply_over_axes(op, voxel_dist,[-1])) for op in operations}

def samples_to_entropy(data, axis, bins=100):
    return np.apply_along_axis(lambda a: stats.entropy(np.histogram(a, bins=bins)[0]/len(a)), axis, data)