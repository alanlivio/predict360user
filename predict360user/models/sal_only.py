from typing import Tuple

import numpy as np

from predict360user.model_wrapper import ModelWrapper


class TrueSaliency(ModelWrapper):
    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError


class ContentSaliency(ModelWrapper):
    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError
