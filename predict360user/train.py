import logging
import os
from dataclasses import dataclass
from os.path import basename, exists, join

import absl.logging
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from keras.callbacks import CSVLogger, ModelCheckpoint
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from wandb.keras import WandbMetricsLogger

import wandb
from predict360user.base_model import BaseModel, batch_generator
from predict360user.ingest import (
    count_entropy,
    get_class_name,
    get_class_thresholds,
    load_df_wins,
    split,
)
from predict360user.models import TRACK, Interpolation, NoMotion, PosOnly, PosOnly3D
from predict360user.registry import EVAL_RES_CSV, TRAIN_RES_CSV
from predict360user.utils.math360 import calc_actual_entropy, orth_dist_cartesian

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
ENTROPY_NAMES = ["all", "low", "medium", "high", "nohigh", "nolow"]
ENTROPY_AUTO_NAMES = ["auto", "auto_m_window", "auto_since_start"]

log = logging.getLogger(basename(__file__))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
tqdm.pandas()

# disable TF logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class TrainerCfg:
    batch_size: int = 128
    dataset_name: str = "all"
    epochs: int = 30
    gpu_id: str = "0"
    h_window: int = 25
    init_window: int = 30
    lr: float = 0.0005
    m_window: int = 5
    model_name: str = "pos_only"
    savedir: str = "saved"
    train_size: float = 0.8
    test_size: float = 0.2
    train_entropy: str = "all"
    minsize: bool = False

    def __post_init__(self) -> None:
        assert self.model_name in MODEL_NAMES
        assert self.train_entropy in ENTROPY_NAMES + ENTROPY_AUTO_NAMES

    def __str__(self) -> str:
        return OmegaConf.to_yaml(self)


cs = ConfigStore.instance()
cs.store(name="trainer", group="trainer", node=TrainerCfg)


class Trainer:
    def __init__(self, cfg: TrainerCfg) -> None:
        self.cfg = cfg
        self.using_auto = self.cfg.train_entropy.startswith("auto")
        self.model_fullname = self.cfg.model_name
        self.model_fullname += f",lr={self.cfg.lr!r}"
        if self.cfg.dataset_name != "all":
            self.model_fullname += f",ds={self.cfg.dataset_name}"
        if self.cfg.train_entropy != "all":
            self.model_fullname += f",actS={self.cfg.train_entropy}"
        if self.cfg.minsize:
            self.model_fullname += f",minsize={self.cfg.minsize!r}"
        self.model_dir = join(self.cfg.savedir, self.model_fullname)
        self.model_path = join(self.model_dir, "weights.hdf5")
        self.train_csv_log_f = join(self.model_dir, TRAIN_RES_CSV)

    def run(self) -> None:
        log.info("Trainer.run using:\n---\n" + OmegaConf.to_yaml(self.cfg) + "----")
        log.info(f"model_dir={self.model_dir}")
        self.df_wins = load_df_wins(
            dataset_name=self.cfg.dataset_name,
            init_window=self.cfg.init_window,
            h_window=self.cfg.h_window,
        )
        self.df_wins = split(
            self.df_wins,
            train_size=self.cfg.train_size,
            train_entropy=self.cfg.train_entropy,
            train_minsize=self.cfg.minsize,
            test_size=self.cfg.test_size,
        )
        self.build_model()
        # setting dirs avoid permisison problems at '/tmp/.config/wandb'
        os.environ["WANDB_DIR"] = self.cfg.savedir
        os.environ["WANDB_CONFIG_DIR"] = self.cfg.savedir
        _, n_low, n_medium, n_high = count_entropy(
            self.df_wins[self.df_wins["partition"] == "train"]
        )
        wandb.init(
            project="predict360user",
            config={
                "model_name": self.cfg.model_name,
                "train_entropy": self.cfg.train_entropy,
                "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr,
                "train_n_low": n_low,
                "train_n_medium": n_medium,
                "train_n_high": n_high,
            },
            name=self.model_fullname,
        )
        self.train()
        self.evaluate()
        wandb.finish()

    def build_model(self) -> None:
        if self.cfg.model_name == "pos_only":
            self.model = PosOnly(self.cfg)
        elif self.cfg.model_name == "pos_only_3d":
            self.model = PosOnly3D(self.cfg)
        elif self.cfg.model_name == "interpolation":
            self.model = Interpolation(self.cfg)
        elif self.cfg.model_name == "TRACK":
            self.model = TRACK(self.cfg)
        elif self.cfg.model_name == "no_motion":
            self.model = NoMotion(self.cfg)
        else:
            raise RuntimeError

    def _auto_select_model(self, traces: np.array, x_i: int) -> BaseModel:
        if self.cfg.train_entropy == "auto":
            window = traces
        elif self.cfg.train_entropy == "auto_m_window":
            window = traces[x_i - self.cfg.m_window : x_i]
        elif self.cfg.train_entropy == "auto_since_start":
            window = traces[0:x_i]
        else:
            raise RuntimeError()
        a_ent = calc_actual_entropy(window)
        actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_high)
        if actS_c == "low":
            return self.model_low
        if actS_c == "medium":
            return self.model_medium
        if actS_c == "high":
            return self.model_high
        raise RuntimeError()

    def train(self) -> None:
        log.info("train ...")
        if self.using_auto or (self.cfg.model_name in MODELS_NAMES_NO_TRAIN):
            return
        if not exists(self.model_dir):
            os.makedirs(self.model_dir)
        log.info("model_dir=" + self.model_dir)

        if exists(self.model_path):
            self.model.load_weights(self.model_path)

        train_wins = self.df_wins[self.df_wins["partition"] == "train"]
        val_wins = self.df_wins[self.df_wins["partition"] == "val"]
        # calc initial_epoch
        initial_epoch = 0
        if exists(self.train_csv_log_f):
            lines = pd.read_csv(self.train_csv_log_f)
            lines.dropna(how="all", inplace=True)
            done_epochs = int(lines.iloc[-1]["epoch"]) + 1
            assert done_epochs <= self.cfg.epochs
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
                CSVLogger(self.train_csv_log_f, append=True),
                ModelCheckpoint(
                    self.model_path,
                    save_best_only=True,
                    save_weights_only=True,
                    mode="auto",
                    period=1,
                ),
                WandbMetricsLogger(initial_global_step=initial_epoch),
            ]
            self.model.fit_generator(
                generator=batch_generator(self.model, train_wins, self.cfg.batch_size),
                validation_data=batch_generator(
                    self.model, val_wins, self.cfg.batch_size
                ),
                steps_per_epoch=steps_per_ep_train,
                validation_steps=steps_per_ep_validate,
                epochs=self.cfg.epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=2,
            )

    def evaluate(self) -> None:
        log.info("evaluate ...")
        if self.using_auto:  # will not use self.model
            prefix = join(
                self.cfg.savedir, f"{self.cfg.model_name},{self.cfg.dataset_name},actS,"
            )
            log.info("creating model auto ...")
            self.threshold_medium, self.threshold_high = get_class_thresholds(
                self.df_wins, "actS"
            )
            self.model_low = self.model.copy()
            self.model_low.load_weights(join(prefix + "low", "weights.hdf5"))
            self.model_medium = self.model.copy()
            self.model_medium.load_weights(join(prefix + "medium", "weights.hdf5"))
            self.model_high = self.model.copy()
            self.model_high.load_weights(join(prefix + "high", "weights.hdf5"))

        test_wins = self.df_wins[self.df_wins["partition"] == "test"]

        # calculate predictions errors
        t_range = list(range(self.cfg.h_window))

        def _calc_pred_err(row) -> list[float]:
            # return np.random.rand(self.cfg.h_window)  # for debugging
            traces = row["traces"]
            x_i = row["trace_id"]
            pred_true = row["h_window"]
            # predict
            if self.using_auto:
                pred = self._auto_select_model(traces, x_i).predict_for_sample(
                    traces, x_i
                )
            else:
                pred = self.model.predict_for_sample(traces, x_i)
            assert len(pred) == self.cfg.h_window
            error_per_t = [orth_dist_cartesian(pred[t], pred_true[t]) for t in t_range]
            return error_per_t

        tqdm.pandas(
            desc="evaluate",
            ascii=True,
            mininterval=60,  # one min
        )
        test_wins[t_range] = test_wins.progress_apply(
            _calc_pred_err, axis=1, result_type="expand"
        )
        assert test_wins[t_range].all().all()
        # save predications
        # 1) avg per class (as wandb summary): # err_all, err_low, err_nohigh, err_medium,
        # err_nolow, err_nolow, err_all, err_high
        # 2) avg err per t per class (as wandb line plots and as csv)
        targets = [
            ("all", pd.Series(True, test_wins.index)),
            ("low", test_wins["actS_c"] == "low"),
            ("nohigh", test_wins["actS_c"] != "high"),
            ("medium", test_wins["actS_c"] == "medium"),
            ("nolow", test_wins["actS_c"] != "low"),
            ("high", test_wins["actS_c"] == "high"),
        ]
        df_test_err_per_t = pd.DataFrame(
            columns=["model_name", "actS_c"] + t_range,
            dtype=np.float32,
        )
        for actS_c, idx in targets:
            # 1)
            class_err = test_wins.loc[idx, t_range].values.mean()
            wandb.run.summary[f"err_{actS_c}"] = class_err
            # 2)
            class_err_per_t = test_wins.loc[idx, t_range].mean()
            data = [[x, y] for (x, y) in zip(t_range, class_err_per_t)]
            table = wandb.Table(data=data, columns=["t", "err"])
            plot_id = f"test_err_per_t_class_{actS_c}"
            plot = wandb.plot.line(table, "t", "err", title=plot_id)
            wandb.log({plot_id: plot})
            # save new row on csv
            df_test_err_per_t.loc[len(df_test_err_per_t)] = [
                self.model_fullname,  # target model
                actS_c,  # target class
            ] + list(class_err_per_t)
        log.info("saving eval_results.csv")
        df_test_err_per_t.to_csv(join(self.model_dir, EVAL_RES_CSV), index=False)
