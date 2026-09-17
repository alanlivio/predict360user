import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import (
    BaseTrainEvalFlow,
    _COMMON_PACKAGES,
)

log = logging.getLogger("pointnet_flow")


class PointNetFlow(BaseTrainEvalFlow):
    model = "pointnet"

    def build_model(self, cfg):
        from typing import Tuple
        import numpy as np
        from predict360user.base_model import BaseModel

        class PointNet(BaseModel):
            def __init__(self, config) -> None:
                super().__init__(config)
                raise NotImplementedError

            def generate_batch(
                self, traces_l: list[np.ndarray], x_i_l: list
            ) -> Tuple[list, list]:
                raise NotImplementedError

            def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
                raise NotImplementedError

        return PointNet(cfg)

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
        try:
            model = self.build_model(cfg)
            model.fit(self.df)
        except NotImplementedError:
            model = None

        self.record_run_results(model)
        self.next(self.join_runs)


if __name__ == "__main__":
    PointNetFlow()
