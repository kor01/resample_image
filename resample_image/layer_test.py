import numpy as np
import tensorflow as tf
from resample_image.layer import *
from resample_image.ops import *

B, H, W, D, C = 4, 32, 64, 32, 8
onebased = True

feature = tf.random.normal((B, H, W, C))
coordinate = tf.random.uniform((B, H, W, D, 2), maxval=W)

layer = ImageResampleLayer('bilinear', True, False, onebased=onebased)

with tf.GradientTape(persistent=True) as tape:
  tape.watch(feature)
  mask, evidence = layer((feature, coordinate))
  nmask, nevidence = resample_image(feature, coordinate, True, False, onebased=onebased)
  np.testing.assert_allclose(evidence, nevidence)
  np.testing.assert_array_equal(mask, nmask)
