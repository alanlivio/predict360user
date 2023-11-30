from typing import Tuple

import numpy as np

from predict360user.model_wrapper import ModelWrapper


class GNN(ModelWrapper):
    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.array, x_i) -> np.array:
        raise NotImplementedError
