import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

MAX_K = 16

resample_image_ops = load_library.load_op_library(
  resource_loader.get_path_to_datafile("_resample_image_ops.so"))

if resample_image_ops is not None:
  resample_image_op = resample_image_ops.resample_image
  resample_image_gradient_op = resample_image_ops.resample_image_gradient
else:
  resample_image_op = None
  resample_image_gradient_op = None


@ops.RegisterGradient("ResampleImage")
def _resample_image_gradient(op: tf.Operation, mask_grad, value_grad):

  assert resample_image_ops is not None
  assert mask_grad is None

  grad_feature = op.get_attr("grad_feature")
  grad_coordinate = op.get_attr("grad_coordinate")
  feature, coordinate, kernel = op.inputs
  mask, value = op.outputs

  feature_grad, coordinate_grad = \
    resample_image_gradient_op(
      feature, coordinate, kernel, mask, value, value_grad,
      grad_feature=grad_feature, grad_coordinate=grad_coordinate)

  if not grad_feature:
    feature_grad = None

  if not grad_coordinate:
    coordinate_grad = None

  return feature_grad, None, coordinate_grad


ops.NotDifferentiable("ResampleImageGradient")


def im2col(feature, K):

  B, H, W, C = feature.shape
  H, W = H - K + 1, W - K + 1
  patches = []

  for i in range(K):
    for j in range(K):
      patches.append(feature[:, i: (i + H), j: (j + W)])

  feature = tf.stack(patches, axis=-1)
  return tf.reshape(feature, (B, H, W, C, K, K))


def resample_image_tfnative(
    feature,
    coordinate,
    weight,
    grad_feature,
    grad_coordinate):

  if not grad_feature:
    feature = tf.stop_gradient(feature)

  if not grad_coordinate:
    weight = tf.stop_gradient(weight)

  H, W = feature.shape[1:3]

  K = weight.shape[-2]
  feature = im2col(feature, K)

  coordinate = tf.roll(coordinate, shift=1, axis=-1)
  coordinate = tf.cast(coordinate, tf.int32)

  wx, wy = weight[..., 0], weight[..., 1]
  gathered = tf.gather_nd(feature, coordinate, batch_dims=1)
  value = tf.reduce_sum(wx[..., None, None, :] * gathered * wy[..., None, :, None], [-1, -2])

  cy, cx = coordinate[..., 0], coordinate[..., 1]

  mask = tf.greater_equal(cy, 0) & tf.less_equal(cy, H - K + 1)
  mask = mask & tf.greater_equal(cx, 0) & tf.less_equal(cx, W - K + 1)

  return mask, value


def resample_image_generic(
    feature,
    coordinate,
    kernel_fn,
    grad_feature,
    grad_coordinate,
    tfnative=False):

  if resample_image_ops is None and not tfnative:
    raise ValueError("native library not loaded")

  fcoordinate = tf.floor(coordinate)
  rcoordinate = coordinate - fcoordinate
  rx, ry = rcoordinate[..., 0], rcoordinate[..., 1]

  wx, wy = kernel_fn(rx), kernel_fn(ry)
  kernel = tf.stack([wx, wy], axis=-1)

  K = kernel.shape[-2]

  if K % 2 != 0 or K <= 0:
    raise ValueError("K should be symmetric (even) and positive")

  if K > MAX_K:
    raise NotImplemented(
      "change MAX_K and recompile binary to support"
      f" kernel radius > {MAX_K}x{MAX_K}")

  icoordinate = fcoordinate - (K / 2 - 1)

  if kernel.shape[-1] != 2:
    raise ValueError("resampling kernel's shape[-2] == 2")

  if tfnative:
    return resample_image_tfnative(
      feature, icoordinate, kernel, grad_feature, grad_coordinate)
  else:
    return resample_image_op(
      feature, icoordinate, kernel,
      grad_feature=grad_feature,
      grad_coordinate=grad_coordinate)


__all__ = ['resample_image_op',
           'resample_image_gradient_op',
           'resample_image_generic']
