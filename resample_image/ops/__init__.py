from resample_image.ops.native_ops import *
from resample_image.ops.bilinear import *
from resample_image.ops.nearest import *


from resample_image.ops import *


METHOD_TABLE = {
  "bilinear": bilinear_resample_image,
  "nearest": nearest_resample_image}


def resample_image(
    feature, coordinate,
    grad_feature=True,
    grad_coordinate=True,
    method="bilinear",
    tfnative=False,
    onebased=False,
    **kwargs):

  if isinstance(method, str):
    return METHOD_TABLE[method](
      feature,
      coordinate,
      grad_feature=grad_feature,
      grad_coordinate=grad_coordinate,
      tfnative=tfnative,
      onebased=onebased,
      **kwargs)
  else:
    return resample_image_generic(
      feature,
      coordinate,
      method,
      grad_coordinate=grad_coordinate,
      grad_feature=grad_coordinate,
      tfnative=tfnative,
      onebased=onebased)


__all__ = ['resample_image']
