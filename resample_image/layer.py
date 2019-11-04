from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple
from resample_image.ops import *


class ImageResampleLayer(layers.Layer):

  def __init__(
      self,
      kernel: str,
      grad_feature: bool,
      grad_coordinate: bool):
    super(ImageResampleLayer, self).__init__()
    self.kernel = kernel
    self.grad_feature = grad_feature
    self.grad_coordinate = grad_coordinate

  def call(self, inputs, **kwargs):

    feature, coordinate = inputs
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
            "grad_coordinate": self.grad_coordinate}


__all__ = ['ImageResampleLayer']
