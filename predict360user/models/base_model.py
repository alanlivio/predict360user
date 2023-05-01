from typing import Tuple

import numpy as np
from keras.models import Model


class BaseModel(Model):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError
