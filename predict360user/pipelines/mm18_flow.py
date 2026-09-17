import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import BaseTrainEvalFlow, _TF_PACKAGES

log = logging.getLogger("mm18_flow")


class MM18Flow(BaseTrainEvalFlow):
    model = "mm18"

    def build_model(self, cfg):
        import copy
        from os.path import join
        from typing import Tuple
        import numpy as np
        from keras.models import load_model

        from predict360user.base_model import BaseModel
        from predict360user.data_ingestion import DATADIR

        class MM18(BaseModel):
            def __init__(self, config) -> None:
                super().__init__(config)
                raise NotImplementedError

            def generate_batch(self, traces_l: list[np.ndarray], x_i_l: list) -> Tuple[list, list]:
                raise NotImplementedError

            def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
                raise NotImplementedError

            def get_model(self) -> None:
                saved_model = load_model(join(DATADIR, "model3_360net_128_w16_h9_8000"))
                self = copy.copy(saved_model)

        return MM18(cfg)

    @conda(
        python="3.11",
        packages=_TF_PACKAGES,
    )
    @step
    def train_and_eval_run(self):
        import os
        import sys

        os.environ["PATH"] = f"{sys.prefix}/bin:" + os.environ.get("PATH", "")
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={sys.prefix}"

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
    MM18Flow()
