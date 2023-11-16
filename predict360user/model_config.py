from dataclasses import dataclass
from os.path import join
from typing import Generator, Tuple

import numpy as np
import pandas as pd

EVAL_RES_CSV = "eval_results.csv"
TRAIN_RES_CSV = "train_results.csv"
DEFAULT_SAVEDIR = "saved"


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

    def __post_init__(self) -> None:
        self.model_fullname = self.model_name
        self.model_fullname += f",lr={self.lr!r}"
        if self.dataset_name != "all":
            self.model_fullname += f",ds={self.dataset_name}"
        if self.train_entropy != "all":
            self.model_fullname += f",actS={self.train_entropy}"
        if self.tuning_entropy:
            self.model_fullname += f",actS_last={self.tuning_entropy}"
        if self.minsize:
            self.model_fullname += f",minsize={self.minsize!r}"
        self.model_dir = join(self.savedir, self.model_fullname)
        self.model_path = join(self.model_dir, "weights.hdf5")
        self.train_csv_log_f = join(self.model_dir, TRAIN_RES_CSV)


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
