import numpy as np
from tensorflow.keras.models import Model

from .base_model import BaseModel


class CVPR18(BaseModel):

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError
