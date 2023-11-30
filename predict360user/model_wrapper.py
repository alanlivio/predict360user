from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np
import pandas as pd

EVAL_RES_CSV = "eval_results.csv"
TRAIN_RES_CSV = "train_results.csv"
DEFAULT_SAVEDIR = "saved"
ENTROPY_NAMES_UNIQUE = ["low", "medium", "high"]


@dataclass
class ModelConf:
    batch_size: int = 128
    dataset_name: str = "all"
    epochs: int = 30
    gpu_id: str = ""
    h_window: int = 25
    init_window: int = 30
    lr: float = 0.0005
    m_window: int = 5
    model_name: str = "pos_only"
    savedir: str = "saved"
    train_size: float = 0.8
    test_size: float = 0.2
    run_name: str = ""


def build_run_name(cfg: ModelConf) -> None:
    cfg.run_name = cfg.model_name
    cfg.run_name += f",lr={cfg.lr!r}"
    if cfg.dataset_name != "all":
        cfg.run_name += f",ds={cfg.dataset_name}"


class ModelWrapper:
    cfg: ModelConf  # should be filled by child class

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        pass

    def predict_for_sample(self, traces: np.array, x_i: int) -> np.array:
        pass


def batch_generator(model, df_wins: pd.DataFrame, batch_size: int) -> Generator:
    assert batch_size
    while True:
        for start in range(0, len(df_wins), batch_size):
            end = (
                start + batch_size
                if start + batch_size <= len(df_wins)
                else len(df_wins)
            )
            traces_l = df_wins[start:end]["traces"].values
            x_i_l = df_wins[start:end]["trace_id"].values
            yield model.generate_batch(traces_l, x_i_l)
