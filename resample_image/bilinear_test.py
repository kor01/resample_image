import numpy as np
import tensorflow as tf
import resampler
from resample_image.ops import *

B, H, W, D, C = 4, 32, 64, 32, 8
onebased = True

feature = tf.random.normal((B, H, W, C))
coordinate = tf.random.uniform((B, H, W, D, 2), maxval=W)


with tf.GradientTape(persistent=True) as tape:

  tape.watch(feature)
  mask, evidence = resample_image(feature, coordinate, True, False, onebased=onebased)
  nevidence = resampler.resampler(feature, coordinate - 1)

  nevidence = tf.where(mask[..., None], nevidence, 0)

  np.testing.assert_allclose(
    evidence, nevidence, rtol=1e-6, atol=1e-6, equal_nan=False)

  grad = tf.random.normal(evidence.shape)

  loss = tf.reduce_sum(grad * evidence)
  nloss = tf.reduce_sum(grad * nevidence)

  evidence_grad = tape.gradient(loss, evidence)
  nevidence_grad = tape.gradient(nloss, nevidence)

  np.testing.assert_allclose(evidence_grad, nevidence_grad, atol=1e-5)

