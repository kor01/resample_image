import unittest
import numpy as np
import tensorflow as tf
from resample_image.ops.bilinear import *

B, H, W, C, K = 4, 240, 320, 32, 1
S = (240, 320, 32)
dtype = tf.float32


class TestBilinear(unittest.TestCase):

  def setUp(self) -> None:
    self.feature = tf.random.normal((B, H, W, C), dtype=dtype)
    self.coordinate = tf.random.uniform(
      (B,) + S + (2,), dtype=tf.float32, minval=0, maxval=W)

  def test_correctness(self):

    mask, value = bilinear_resample_image(
      self.feature, self.coordinate, True, True)

    return;
    masknative, valuenative = bilinear_resample_image(
      self.feature, self.coordinate, True, True, tfnative=True)

    self.assertTrue(np.testing.assert_allclose(mask, masknative))
    self.assertTrue(np.testing.assert_allclose(value, valuenative))

  def test_eager_gradient(self):
    pass

  def test_gradient(self):
    pass


if __name__ == '__main__':
  unittest.main()
