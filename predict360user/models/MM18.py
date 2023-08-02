import copy
from os.path import join
from typing import Tuple

import numpy as np
from keras.models import load_model
from tensorflow import keras

from predict360user.models.base_model import BaseModel
from predict360user.utils import DATADIR


class MM18(keras.Model, BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError

  def __init__(self, m_window: int, h_window: int) -> None:
    self.m_window, self.h_window = m_window, h_window
    saved_model = load_model(join(DATADIR ,'model3_360net_128_w16_h9_8000'))
    self = copy.copy(saved_model)
