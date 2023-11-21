from dataclasses import dataclass, MISSING
from typing import Generator, Tuple

import numpy as np
import pandas as pd

EVAL_RES_CSV = "eval_results.csv"
TRAIN_RES_CSV = "train_results.csv"
DEFAULT_SAVEDIR = "saved"
ENTROPY_NAMES = ["all", "low", "medium", "high", "nohigh", "nolow"]
ENTROPY_NAMES_UNIQUE = ["low", "medium", "high"]
ENTROPY_NAMES_AUTO = ["auto", "auto_m_window", "auto_since_start"]


@dataclass
class Config:
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
    tuning_entropy: str = ""
    train_entropy: str = "all"
    minsize: bool = False
    model_fullname: str = MISSING

def build_model_fullname(cfg: Config) -> None:
    cfg.model_fullname = cfg.model_name
    cfg.model_fullname += f",lr={cfg.lr!r}"
    if cfg.dataset_name != "all":
        cfg.model_fullname += f",ds={cfg.dataset_name}"
    assert cfg.train_entropy in ENTROPY_NAMES + ENTROPY_NAMES_AUTO
    if cfg.train_entropy != "all":
        cfg.model_fullname += f",filt={cfg.train_entropy}"
    if cfg.tuning_entropy:
        cfg.model_fullname += f",tuni={cfg.tuning_entropy}"
    if cfg.minsize:
        cfg.model_fullname += f",mins={cfg.minsize!r}"

class BaseModel:
    cfg: Config  # should be filled by child class

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
