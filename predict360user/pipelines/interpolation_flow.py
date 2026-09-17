import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import _COMMON_PACKAGES, BaseTrainEvalFlow

log = logging.getLogger("interpolation_flow")


class InterpolationFlow(BaseTrainEvalFlow):
    model = "interpolation"

    def build_model(self, cfg):
        import numpy as np
        import pandas as pd
        from predict360user.base_model import BaseModel
        from predict360user.utils.math360 import rotationBetweenVectors

        class Interpolation(BaseModel):
            def __init__(self, config):
                super().__init__(config)
                self.history = {"loss": [0.0], "val_loss": [0.0]}

            def fit(self, df: pd.DataFrame) -> BaseModel:
                self.history = {
                    "loss": [0.0] * max(1, self.cfg.epochs),
                    "val_loss": [0.0] * max(1, self.cfg.epochs),
                }
                return self

            def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
                rotation = rotationBetweenVectors(traces[x_i - 2], traces[x_i - 1])
                prediction = [rotation.rotate(traces[x_i])]
                for _ in range(self.cfg.h_window - 1):
                    prediction.append(rotation.rotate(prediction[-1]))
                return np.array(prediction)

            def predict(self, df: pd.DataFrame) -> list:
                preds = []
                for _, row in df.iterrows():
                    m_win = np.asarray(row["m_window"])
                    rotation = rotationBetweenVectors(m_win[-2], m_win[-1])
                    prediction = [rotation.rotate(m_win[-1])]
                    for _ in range(self.cfg.h_window - 1):
                        prediction.append(rotation.rotate(prediction[-1]))
                    preds.append(np.array(prediction))
                return preds

        return Interpolation(cfg)

    @conda(
        python="3.11",
        packages=_COMMON_PACKAGES,
    )
    @step
    def train_and_eval_run(self):
        from predict360user.run_config import set_random_seed

        self.seed = self.input
        set_random_seed(self.seed)

        cfg = self.build_run_config(self.seed)
        model = self.build_model(cfg)
        model.fit(self.df)

        self.record_run_results(model)
        self.next(self.join_runs)


if __name__ == "__main__":
    InterpolationFlow()
