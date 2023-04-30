import numpy as np
from tensorflow.keras.models import Model

from .base_model import BaseModel


class TrueSaliency(BaseModel):

  def predict(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError


class ContentSaliency(BaseModel):

  def predict(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError
