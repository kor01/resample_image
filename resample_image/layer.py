import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple
from resample_image.ops import *


class ImageResampleLayer(layers.Layer):

  def __init__(
      self,
      kernel: str,
      grad_feature: bool,
      grad_coordinate: bool,
      strides=None):

    super(ImageResampleLayer, self).__init__()
    self.kernel = kernel
    self.grad_feature = grad_feature
    self.grad_coordinate = grad_coordinate

    if strides is not None:
      if isinstance(strides, int):
        strides = (strides, strides, strides)
      strides = tuple(strides)
      if len(strides) != 3:
        raise ValueError()

    self.strides = strides

  def call(self, inputs, **kwargs):

    feature, coordinate = inputs

    if self.strides is not None:
      H, W, D = self.strides
      coordinate = coordinate[:, ::H, ::W, ::D]

    mask, value = resample_image(
      feature,
      coordinate,
      grad_coordinate=self.grad_coordinate,
      grad_feature=self.grad_feature,
      method=self.kernel)

    return mask, value

  def get_config(self):
    return {"kernel": self.kernel,
            "grad_feature": self.grad_feature,
            "grad_coordinate": self.grad_coordinate,
            'strides': self.strides}


__all__ = ['ImageResampleLayer']
