from typing import Tuple

import numpy as np
from tensorflow import keras

from predict360user.estimator import Estimator, Config


class PointNet(Estimator):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # self.model: keras.Model = self.build()
        raise NotImplementedError

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError
