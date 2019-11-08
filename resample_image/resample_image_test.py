B, H, W, D, C = 4, 32, 64, 32, 4

import numpy as np
import tensorflow as tf
from resample_image.ops import *

feature = tf.random.normal((B, H, W, C))
coordinate = tf.random.uniform((B, H, W, D, 2), maxval=W)

with tf.GradientTape(persistent=True) as tape:

  tape.watch(feature)
  mask, evidence = resample_image(feature, coordinate, True, False)
  print('evidence done')
  nmask, nevidence = resample_image(feature, coordinate, True, False, tfnative=True)
  print('evidence native done')
  print(np.allclose(evidence, nevidence, atol=5e-7))
  print(np.array_equal(mask, nmask))

  grad = tf.random.normal(evidence.shape)
  loss = tf.reduce_sum(grad * evidence)
  nloss = tf.reduce_sum(grad * nevidence)

  evidence_grad = tape.gradient(loss, feature)
  nevidence_grad = tape.gradient(nloss, feature)
  np.testing.assert_allclose(evidence_grad, nevidence_grad, atol=1e-5)
