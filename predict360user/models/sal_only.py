from typing import Tuple

import numpy as np

from predict360user.base_model import BaseModel, RunConfig


class TrueSaliency(BaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        # self.cfg = cfg
        # self.model: keras.Model = self.build()
        raise NotImplementedError

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict(self, traces: np.ndarray, x_i) -> np.ndarray:
        raise NotImplementedError


class ContentSaliency(BaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        # self.cfg = cfg
        # self.model: keras.Model = self.build()
        raise NotImplementedError

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict(self, traces: np.ndarray, x_i) -> np.ndarray:
        raise NotImplementedError
