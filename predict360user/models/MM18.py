import copy
from os.path import join
from typing import Tuple

import numpy as np
from keras.models import load_model

from predict360user.base_model import KerasBaseModel, RunConfig
from predict360user.data_ingestion import DATADIR


class MM18(KerasBaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        # self.model: keras.Model = self.build()
        raise NotImplementedError

    def generate_batch(self, traces_l: list[np.ndarray], x_i_l: list) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
        raise NotImplementedError

    def build(self) -> None:
        self.m_window, self.h_window = self.cfg.m_window, self.cfg.h_window
        saved_model = load_model(join(DATADIR, "model3_360net_128_w16_h9_8000"))
        self = copy.copy(saved_model)
