import logging
import os
from dataclasses import dataclass
from os.path import exists, join
from typing import Generator, Tuple

import absl
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow import keras
from tqdm.auto import tqdm
from wandb.keras import WandbMetricsLogger

import wandb
from predict360user.utils.math360 import orth_dist_cartesian

log = logging.getLogger()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
tqdm.pandas()

MODEL_NAMES = [
    "pos_only",
    "pos_only_3d",
    "no_motion",
    "interpolation",
    "TRACK",
    "CVPR18",
    "MM18",
    "most_salient_point",
]
MODELS_NAMES_NO_TRAIN = ["no_motion", "interpolation"]

EVAL_RES_CSV = "eval_results.csv"
TRAIN_RES_CSV = "train_results.csv"
DEFAULT_SAVEDIR = "saved"
ENTROPY_NAMES_UNIQUE = ["low", "medium", "high"]


# disable TF logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
    run_name: str = ""

    def __post_init__(self) -> None:
        self.run_name += f"{self.model_name},lr={self.lr!r}"
        if self.dataset_name != "all":
            self.run_name += f",ds={self.dataset_name}"

class ModelWrapper:
    cfg: Config  # should be filled by child class

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        pass

    def predict_for_sample(self, traces: np.array, x_i: int) -> np.array:
        pass

    def fit(self, df_wins: pd.DataFrame) -> None:
        pass

    def evaluate(self, df_wins: pd.DataFrame) -> dict:
        log.info("evaluate ...")

        # calculate predictions errors
        test_wins = df_wins[df_wins["partition"] == "test"]
        t_range = list(range(self.cfg.h_window))

        def _calc_pred_err(row) -> list[float]:
            # return np.random.rand(cfg.h_window)  # for debugging
            traces = row["traces"]
            x_i = row["trace_id"]
            pred_true = row["h_window"]
            # predict
            pred = self.predict_for_sample(traces, x_i)
            assert len(pred) == self.cfg.h_window
            error_per_t = [orth_dist_cartesian(pred[t], pred_true[t]) for t in t_range]
            return error_per_t

        tqdm.pandas(
            desc="evaluate",
            ascii=True,
            mininterval=60,  # one min
        )
        df_wins.loc[test_wins.index, t_range] = test_wins.progress_apply(
            _calc_pred_err, axis=1, result_type="expand"
        )
        assert df_wins.loc[test_wins.index, t_range].all().all()

        # calculate predications errors mean
        classes = [
            ("all", test_wins.index),
            ("low", test_wins.index[test_wins["actS_c"] == "low"]),
            # ("nohigh", test_wins.index[test_wins["actS_c"] != "high"]),
            ("medium", test_wins.index[test_wins["actS_c"] == "medium"]),
            # ("nolow", test_wins.index[test_wins["actS_c"] != "low"]),
            ("high", test_wins.index[test_wins["actS_c"] == "high"]),
        ]
        err_per_class_dict = {tup[0]: {} for tup in classes}
        for actS_c, idx in classes:
            # 1) mean per class (as wandb summary): # err_all, err_low, err_nohigh, err_medium,
            err_per_class_dict[actS_c]["mean"] = df_wins.loc[idx, t_range].values.mean()
            # 2) mean err per t per class
            class_err_per_t = df_wins.loc[idx, t_range].mean()
            data = [[x, y] for (x, y) in zip(t_range, class_err_per_t)]
            err_per_class_dict[actS_c]["mean_per_t"] = data
        if wandb.run:
            for actS_c, err in err_per_class_dict.items():
                wandb.run.summary[f"err_{actS_c}"] = err["mean"]
                table = wandb.Table(data=err["mean_per_t"], columns=["t", "err"])
                plot_id = f"test_err_per_t_class_{actS_c}"
                plot = wandb.plot.line(table, "t", "err", title=plot_id)
                wandb.log({plot_id: plot})
        return err_per_class_dict

    def batch_generator(self, df_wins: pd.DataFrame) -> Generator:
        while True:
            for start in range(0, len(df_wins), self.cfg.batch_size):
                end = (
                    start + self.cfg.batch_size
                    if start + self.cfg.batch_size <= len(df_wins)
                    else len(df_wins)
                )
                traces_l = df_wins[start:end]["traces"].values
                x_i_l = df_wins[start:end]["trace_id"].values
                yield self.generate_batch(traces_l, x_i_l)


class KerasModelWrapper(ModelWrapper):
    model: keras.Model

    def fit(self, df_wins: pd.DataFrame) -> None:
        log.info("train ...")
        if self.cfg.model_name in MODELS_NAMES_NO_TRAIN:
            return

        model_dir = join(self.cfg.savedir, self.cfg.run_name)
        train_csv_log_f = join(model_dir, TRAIN_RES_CSV)
        model_path = join(model_dir, "weights.hdf5")

        if not exists(model_dir):
            os.makedirs(model_dir)
        if exists(model_path):
            self.model.load_weights(model_path)
        log.info("model_path=" + model_path)

        train_wins = df_wins[df_wins["partition"] == "train"]
        val_wins = df_wins[df_wins["partition"] == "val"]
        # calc initial_epoch
        initial_epoch = 0
        if exists(train_csv_log_f):
            lines = pd.read_csv(train_csv_log_f)
            lines.dropna(how="all", inplace=True)
            done_epochs = int(lines.iloc[-1]["epoch"]) + 1
            initial_epoch = done_epochs
            log.info(f"train_csv_log_f has {initial_epoch} epochs ")

        # fit
        if self.cfg.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)
            log.info(f"set visible cpu to {self.cfg.gpu_id}")
        if initial_epoch >= self.cfg.epochs:
            log.info(
                f"train_csv_log_f has {initial_epoch}>={self.cfg.epochs}. not training."
            )
        else:
            steps_per_ep_train = np.ceil(len(train_wins) / self.cfg.batch_size)
            steps_per_ep_validate = np.ceil(len(val_wins) / self.cfg.batch_size)
            callbacks = [
                CSVLogger(train_csv_log_f, append=True),
                ModelCheckpoint(
                    model_path,
                    save_best_only=True,
                    save_weights_only=True,
                    mode="auto",
                    period=1,
                ),
            ]
            if wandb.run:
                callbacks += [WandbMetricsLogger(initial_global_step=initial_epoch)]
            self.model.fit_generator(
                generator=self.batch_generator(train_wins),
                validation_data=self.batch_generator(val_wins),
                steps_per_epoch=steps_per_ep_train,
                validation_steps=steps_per_ep_validate,
                epochs=self.cfg.epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=2,
            )
