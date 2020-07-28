"""
Tools for purposes of the thesis Deep Learning for MRI data by Tomas Karella.

This python library help the user with designing the Deep Learning models for MRI classification including preprocessing,
 loading data, training encoders or classifiers.

The DL part is powered by the [TensorFlow 2](https://www.tensorflow.org) framework and [Nipype](https://nipype.readthedocs.io/en/latest/) creates
the processing pipelines.


"""
from . import dataset, model_zoo, preprocess, train

__all__ = ["dataset", "model_zoo", "preprocess", "train"]
