import copy
from os.path import join
from typing import Tuple

import numpy as np
from keras.models import load_model
from tensorflow import keras

from predict360user.ingest import DATADIR
from predict360user.estimator import KerasEstimator, Config


class MM18(KerasEstimator):
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

    def build(self) -> None:
        self.m_window, self.h_window = self.cfg.m_window, self.cfg.h_window
        saved_model = load_model(join(DATADIR, "model3_360net_128_w16_h9_8000"))
        self = copy.copy(saved_model)
