import numpy as np
from tensorflow.keras.models import Model

from .base_model import BaseModel


class MM18(BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError
