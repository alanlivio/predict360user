import numpy as np
from keras.models import Model

from .base_model import BaseModel


class TRACK(BaseModel):

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError