import numpy as np
# TODO: Inplace method much more faster
# TODO: Aware of different scanners
# MARK: Sklearn does not work od 3D data
# TODO: Registration etc... https://mirtk.github.io
# https://nilearn.github.io
def normalize(data, feature_range=(0, 1), method="MinMax", min_data=None, max_data=None, copy=True):
    min_out, max_out = feature_range
    if method == "MinMax":
        if not min_data: np.min(data)
        if not max_data: np.max(data)

        scale = (max_out - min_out) / (max_data - min_data)
        if copy:
            return data * scale + min_out - min_data * scale
        else:
            data *= scale
            data += min_out - min_data * scale
    else:
        raise Exception("Unknown method {}".format(method))