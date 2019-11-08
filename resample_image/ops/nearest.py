import tensorflow as tf
from resample_image.ops.native_ops import *


@tf.function
def nearest_kernel(r):
  r = tf.round(r)
  cr = 1 - r
  return tf.stack([cr, r], axis=-2)


@tf.function
def nearest_resample_image(
    feature, coordinate,
    grad_feature: bool,
    grad_coordinate: bool,
    tfnative=False,
    onebase=False):

  if grad_coordinate:
    raise ValueError(
      "nearest kernel coordinate not differentiable")

  return resample_image_generic(
    feature, coordinate,
    nearest_kernel,
    grad_feature,
    False,
    tfnative=tfnative,
    onebase=onebase)


__all__ = ['nearest_resample_image']
