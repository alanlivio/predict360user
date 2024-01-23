from typing import Tuple

import numpy as np

from predict360user.model_wrapper import ModelWrapper, Config


class TrueSaliency(ModelWrapper):
    def __init__(self, cfg: Config) -> None:
        # self.cfg = cfg
        # self.model: keras.Model = self.build()
        raise NotImplementedError

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError


class ContentSaliency(ModelWrapper):
    def __init__(self, cfg: Config) -> None:
        # self.cfg = cfg
        # self.model: keras.Model = self.build()
        raise NotImplementedError

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError
