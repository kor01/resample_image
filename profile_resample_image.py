from tensorflow.python.eager import profiler
import tensorflow as tf
import resampler

from resample_image import resample_image

#tf.debugging.set_log_device_placement(True)

B, H, W, C, K = 4, 240, 320, 32, 1

TFNATIVE = True

S = (240, 320, 32)

feature = tf.random.normal((B, H, W, C), dtype=tf.float32)
coordinate = tf.random.uniform((B,) + S + (2,), dtype=tf.float32, minval=0, maxval=W)


@tf.function
def get_gradient(feature, coordinate, tfnative):
  mask, value = resample_image(
    feature, coordinate, grad_feature=True, grad_coordinate=False, tfnative=tfnative)
  value_grad = tf.random.normal(value.shape, dtype=value.dtype)
  v = tf.reduce_sum(value * value_grad)
  grad, = tf.gradients(v, feature)
  return grad, value_grad


@tf.function
def get_gradient_gt(feature, coordinate, value_grad):
  value = resampler.resampler(feature, coordinate)
  v = tf.reduce_sum(value * value_grad)
  grad, = tf.gradients(v, feature)
  
  return grad


mask, value = resample_image(feature, coordinate, tfnative=TFNATIVE)
gt = resampler.resampler(feature, coordinate)
grad, value_grad = get_gradient(feature, coordinate, TFNATIVE)
value_grad = tf.where(mask[..., None], value_grad, 0)
grad_gt = get_gradient_gt(feature, coordinate, value_grad)

profiler.start()

for i in range(3):
  # feature_grad, _ = get_gradient(feature, coordinate, tfnative=TFNATIVE)
  # print(feature_grad.shape)
  mask, value = resample_image(feature, coordinate, tfnative=TFNATIVE)

profiler_result = profiler.stop()
profiler.save('./profile.log', profiler_result)
