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

def normal_dist_entropy(variance):
    return 0.5*np.log(2*np.pi*np.e*variance)

def get_entropy_slices(var_matrix, num_slices=6, dist='normal'):
    if dist != 'normal':
        raise Exception("Not implemented for distribution {}".format(dist))
    assert num_slices > 0
    assert num_slices <= np.min(var_matrix.shape)
    
    min_var = np.min(var_matrix[var_matrix != 0 ])
    no_zeros = var_matrix
    no_zeros[no_zeros == 0] = min_var/100.0
    
    entropy = normal_dist_entropy(no_zeros)
    #TODO: Make it more general
    first_dim_entropy = np.sum(np.sum(entropy,-1),-1)
    second_dim_entropy = np.sum(np.sum(entropy,0),-1)
    third_dim_entropy  = np.sum(np.sum(entropy,0), 0)
        
    return np.array([
        np.argsort(first_dim_entropy)[-1*num_slices:],
        np.argsort(second_dim_entropy)[-1*num_slices:],
        np.argsort(third_dim_entropy)[-1*num_slices:]
    ])