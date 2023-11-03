from typing import Tuple

import numpy as np
from tensorflow import keras

from predict360user.model_config import BaseModel


class PointNet(keras.Model, BaseModel):
    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError
