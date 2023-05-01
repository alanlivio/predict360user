from typing import Tuple

import numpy as np

from predict360user.models.base_model import BaseModel


class TrueSaliency(BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError


class ContentSaliency(BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError