
import numpy as np
import pandas as pd

from predict360user.base_model import BaseModel, RunConfig
from predict360user.utils.math360 import rotationBetweenVectors


class NoMotion(BaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg)
    
    def fit(self, df_wins: pd.DataFrame) -> BaseModel:
        return self
    
    def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
        model_pred = np.repeat(traces[x_i : x_i + 1], self.h_window, axis=0)
        return model_pred


class Interpolation(BaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg)

    
    def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
        rotation = rotationBetweenVectors(traces[x_i - 2], traces[x_i - 1])
        prediction = [rotation.rotate(traces[x_i])]
        for _ in range(self.cfg.h_window - 1):
            prediction.append(rotation.rotate(prediction[-1]))
        return prediction


class Regression(BaseModel):
    
    def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
        raise NotImplementedError
