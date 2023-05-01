from typing import Tuple

import numpy as np

from predict360user.models.base_model import BaseModel


class PointNet(BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError
