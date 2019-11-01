import numpy as np

def normalize(data, feature_range=(0,1), method="MinMax"):
    min_out, max_out = feature_range
    if method == "MinMax":
        min_data, max_data = np.min(data), np.max(data)
        scale = (max_out - min_out) / (max_data - min_data)
        scaled_data = data * scale + min_out - min_data * scale
    else:
        raise Exception("Unknown method {}".format(method))

    assert np.max(scaled_data) <= max_out
    assert np.min(scaled_data) >= min_out
    return scaled_data
