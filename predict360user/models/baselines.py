from typing import Tuple

import numpy as np

from predict360user.model_config import BaseModel
from predict360user.utils.math360 import rotationBetweenVectors


class NoMotion(BaseModel):
    def __init__(self, h_window) -> None:
        super().__init__()
        self.h_window = h_window

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.array, x_i) -> np.array:
        model_pred = np.repeat(traces[x_i : x_i + 1], self.h_window, axis=0)
        return model_pred


class Interpolation(BaseModel):
    def __init__(self, h_window) -> None:
        super().__init__()
        self.h_window = h_window

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.array, x_i) -> np.array:
        rotation = rotationBetweenVectors(traces[x_i - 2], traces[x_i - 1])
        prediction = [rotation.rotate(traces[x_i])]
        for _ in range(self.h_window - 1):
            prediction.append(rotation.rotate(prediction[-1]))
        return prediction


class Regression(BaseModel):
    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError
