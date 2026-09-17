import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import _COMMON_PACKAGES, BaseTrainEvalFlow

log = logging.getLogger("no_motion_flow")


class NoMotionFlow(BaseTrainEvalFlow):
    model = "no_motion"

    def build_model(self, cfg):
        import numpy as np
        import pandas as pd
        from predict360user.base_model import BaseModel

        class NoMotion(BaseModel):
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
                return np.repeat(traces[x_i : x_i + 1], self.cfg.h_window, axis=0)

            def predict(self, df: pd.DataFrame) -> list:
                preds = []
                for _, row in df.iterrows():
                    if "trace" in row and row["trace"] is not None:
                        last_pos = np.asarray(row["trace"])
                    else:
                        last_pos = np.asarray(row["m_window"])[-1]
                    if last_pos.ndim == 2:
                        last_pos = last_pos[0]
                    pred = np.repeat(last_pos[np.newaxis, :], self.cfg.h_window, axis=0)
                    preds.append(pred)
                return preds

        return NoMotion(cfg)

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
    NoMotionFlow()
