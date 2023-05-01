from abc import abstractclassmethod
from typing import Tuple

import numpy as np
from keras.models import Model


class BaseModel(Model):

  @abstractclassmethod
  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    pass

  @abstractclassmethod
  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    pass
