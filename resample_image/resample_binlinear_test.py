import numpy as np
import tensorflow as tf
import resampler
from resample_image.ops import *

B, H, W, D, C = 4, 32, 64, 32, 8

feature = tf.random.normal((B, H, W, C))
coordinate = tf.random.uniform((B, H, W, D, 2), maxval=W)

with tf.GradientTape(persistent=True) as tape:

  tape.watch(feature)
  mask, evidence = resample_image(feature, coordinate, True, False)
  nevidence = resampler.resampler(feature, coordinate)

  nevidence = tf.where(mask[..., None], nevidence, 0)

  np.testing.assert_allclose(
    evidence, nevidence, rtol=1e-6, atol=1e-6, equal_nan=False)

  grad = tf.random.normal(evidence.shape)
  
