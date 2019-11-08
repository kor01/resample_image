import tensorflow as tf
from resample_image.ops.native_ops import *


@tf.function
def bilinear_kernel(r):
  cr = 1 - r
  return tf.stack([cr, r], axis=-1)


@tf.function
def bilinear_resample_image(
    feature, coordinate,
    grad_feature: bool,
    grad_coordinate: bool,
    tfnative=False,
    onebased=False):

    return resample_image_generic(
      feature, coordinate,
      bilinear_kernel,
      grad_feature=grad_feature,
      grad_coordinate=grad_coordinate,
      tfnative=tfnative,
      onebased=onebased)


__all__ = ['bilinear_resample_image']
