import logging
import os
from dataclasses import dataclass
from os.path import basename, exists, join, isdir
from typing import Generator
import absl.logging
import numpy as np
import pandas as pd
import plotly.express as px
import wandb
from wandb.keras import WandbMetricsLogger

from hydra.core.config_store import ConfigStore
from keras.callbacks import CSVLogger, ModelCheckpoint
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from .base_model import BaseModel, Interpolation, NoMotion
from predict360user.dataset import (
    Dataset,
    get_class_name,
    get_class_thresholds,
    count_entropy,
)
from predict360user.models import PosOnly, PosOnly3D, TRACK

from predict360user.utils import *

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
    gpu_id: int = 0
    h_window: int = 25
    init_window: int = 30
    lr: float = 0.0005
    m_window: int = 5
    model_name: str = "pos_only"
    savedir: str = "saved"
    train_size: float = 0.8
    test_size: float = 0.2
    train_entropy: str = "all"
    wandb_mode: str = "online"
    minsize: bool = False

    def __post_init__(self) -> None:
        assert self.model_name in MODEL_NAMES
        assert self.train_entropy in ENTROPY_NAMES + ENTROPY_AUTO_NAMES

    def __str__(self) -> str:
        return OmegaConf.to_yaml(self)


cs = ConfigStore.instance()
cs.store(name="trainer", group="trainer", node=TrainerCfg)


class Trainer:
    cfg: TrainerCfg

    def __init__(self, cfg: TrainerCfg) -> None:
        log.info("TrainerCfg:\n-------\n" + OmegaConf.to_yaml(cfg) + "-------")
        self.cfg = cfg
        self.using_auto = self.cfg.train_entropy.startswith("auto")
        self.model_fullname = self.cfg.model_name
        if self.cfg.dataset_name != "all":
            self.model_fullname += f",ds={self.cfg.dataset_name}"
        if self.cfg.train_entropy != "all":
            self.model_fullname += f",actS={self.cfg.train_entropy}"
        if self.cfg.minsize:
            self.model_fullname += f",minsize={self.cfg.minsize!r}"
        log.info(f"model_fullname={self.model_fullname}")
        self.model_dir = join(self.cfg.savedir, self.model_fullname)
        log.info(f"model_dir={self.model_dir}")
        self.train_csv_log_f = join(self.model_dir, "train_loss.csv")
        self.model_path = join(self.model_dir, "weights.hdf5")

    def run(self) -> None:
        self.build_data()
        self.build_model()
        # setting dirs avoid permisison problems at '/tmp/.config/wandb'
        os.environ["WANDB_DIR"] = self.cfg.savedir
        os.environ["WANDB_CONFIG_DIR"] = self.cfg.savedir
        _, n_low, n_medium, n_high = count_entropy(self.ds.x_train)
        wandb.init(
            project="predict360user",
            tags=[self.cfg.model_name, self.cfg.train_entropy],
            mode=self.cfg.wandb_mode,
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

    def generate_batchs(self, model: BaseModel, df_wins: pd.DataFrame) -> Generator:
        while True:
            for count, _ in enumerate(df_wins[:: self.cfg.batch_size]):
                end = (
                    count + self.cfg.batch_size
                    if count + self.cfg.batch_size <= len(df_wins)
                    else len(df_wins)
                )
                traces_l = [
                    self.ds.df.loc[row["ds"], row["user"], row["video"]]["traces"]
                    for _, row in df_wins[count:end].iterrows()
                ]
                x_i_l = [row["trace_id"] for _, row in df_wins[count:end].iterrows()]
                yield model.generate_batch(traces_l, x_i_l)

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

    def build_data(self) -> None:
        log.info("loading dataset ...")
        self.ds = Dataset(dataset_name=self.cfg.dataset_name, savedir=self.cfg.savedir)
        self.ds.partition(
            train_filter=self.cfg.train_entropy,
            train_size=self.cfg.train_size,
            test_size=self.cfg.test_size,
            minsize=self.cfg.minsize,
        )
        self.ds.create_wins(
            init_window=self.cfg.init_window, h_window=self.cfg.h_window
        )

    def train(self) -> None:
        log.info("train ...")
        if self.using_auto or (self.cfg.model_name in MODELS_NAMES_NO_TRAIN):
            return
        if not exists(self.model_dir):
            os.makedirs(self.model_dir)
        log.info("model_dir=" + self.model_dir)

        if exists(self.model_path):
            self.model.load_weights(self.model_path)

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
            steps_per_ep_train = np.ceil(
                len(self.ds.x_train_wins) / self.cfg.batch_size
            )
            steps_per_ep_validate = np.ceil(
                len(self.ds.x_val_wins) / self.cfg.batch_size
            )
            callbacks = [
                CSVLogger(self.train_csv_log_f, append=True),
                ModelCheckpoint(self.model_path, save_weights_only=True),
                WandbMetricsLogger(initial_global_step=initial_epoch),
            ]
            generator = self.generate_batchs(self.model, self.ds.x_train_wins)
            validation_data = self.generate_batchs(self.model, self.ds.x_val_wins)
            self.model.fit(
                x=generator,
                steps_per_epoch=steps_per_ep_train,
                validation_data=validation_data,
                validation_steps=steps_per_ep_validate,
                validation_freq=self.cfg.batch_size,
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
                self.ds.df, "actS"
            )
            self.model_low = self.model.copy()
            self.model_low.load_weights(join(prefix + "low", "weights.hdf5"))
            self.model_medium = self.model.copy()
            self.model_medium.load_weights(join(prefix + "medium", "weights.hdf5"))
            self.model_high = self.model.copy()
            self.model_high.load_weights(join(prefix + "high", "weights.hdf5"))

        # calculate predictions errors
        t_range = list(range(self.cfg.h_window))

        def _calc_pred_err(row) -> None:
            # return np.random.rand(self.cfg.h_window)  # for debugging
            # row.name return the index (user, video, time)
            ds, user, video, x_i = row["ds"], row["user"], row["video"], row["trace_id"]
            traces = self.ds.df.loc[ds, user, video]["traces"]
            # predict
            if self.using_auto:
                pred = self._auto_select_model(traces, x_i).predict_for_sample(
                    traces, x_i
                )
            else:
                pred = self.model.predict_for_sample(traces, x_i)
            assert len(pred) == self.cfg.h_window
            pred_true = traces[x_i + 1 : x_i + self.cfg.h_window + 1]
            error_per_t = [orth_dist_cartesian(pred[t], pred_true[t]) for t in t_range]
            return error_per_t

        tqdm.pandas(
            desc=f"evaluate model {self.model_fullname}",
            ascii=True,
            mininterval=5,
        )
        self.ds.x_test_wins[t_range] = self.ds.x_test_wins.progress_apply(
            _calc_pred_err, axis=1, result_type="expand"
        )
        assert self.ds.x_test_wins[t_range].all().all()
        # save predications
        # 1) avg per class as wandb summary: # err_all, err_low, err_nohigh, err_medium,
        # err_nolow, err_nolow, err_all, err_high
        # 2.1) avg err per t per class as wandb line plots
        # 2.2) avg err per t per class as csv to see by show_saved_train_pred_err
        targets = [
            ("all", pd.Series(True, self.ds.x_test_wins.index)),
            ("low", self.ds.x_test_wins["actS_c"] == "low"),
            ("nohigh", self.ds.x_test_wins["actS_c"] != "high"),
            ("medium", self.ds.x_test_wins["actS_c"] == "medium"),
            ("nolow", self.ds.x_test_wins["actS_c"] != "low"),
            ("high", self.ds.x_test_wins["actS_c"] == "high"),
        ]
        df_test_err_per_t = pd.DataFrame(
            columns=["model_name", "actS_c"] + t_range,
            dtype=np.float32,
        )
        for actS_c, idx in targets:
            class_err = round(np.nanmean(self.ds.x_test_wins[t_range].values), 4)
            # 1)
            wandb.run.summary[f"err_{actS_c}"] = class_err
            class_err_per_t = self.ds.x_test_wins.loc[idx, t_range].mean().round(4)
            new_row = [
                self.model_fullname,  # target model
                actS_c,  # target class
            ] + list(class_err_per_t)
            newid = len(df_test_err_per_t)
            df_test_err_per_t.loc[newid] = new_row
            # 2.1)
            data = [[x, y] for (x, y) in zip(t_range, class_err_per_t)]
            table = wandb.Table(data=data, columns=["t", "err"])
            plot_id = f"test_err_per_t_class_{actS_c}"
            plot = wandb.plot.line(table, "t", "err", title=plot_id)
            wandb.log({plot_id: plot})
        # 2.2)
        log.info("saving test_err_per_t.csv")
        df_test_err_per_t.to_csv(
            join(self.model_dir, "test_err_per_t.csv"), index=False
        )


# compare funcs using saved/ logs, as alternative to wandb


def show_saved_train_loss(savedir="saved") -> None:
    results_csv = "train_loss.csv"
    # find results_csv files
    csv_df_l = [
        (dir_name, pd.read_csv(join(savedir, dir_name, file_name)))
        for dir_name in os.listdir(savedir)
        if isdir(join(savedir, dir_name))
        for file_name in os.listdir(join(savedir, dir_name))
        if file_name == results_csv
    ]
    csv_df_l = [df.assign(model_name=dir_name) for (dir_name, df) in csv_df_l]
    assert csv_df_l, f"no <savedir>/<model>/{results_csv} files"
    df_compare = pd.concat(csv_df_l, ignore_index=True)

    # plot
    fig = px.line(
        df_compare,
        x="epoch",
        y="loss",
        color="model_name",
        title="compare_train_loss",
        width=800,
    )
    show_or_save(fig, savedir, "compare_train")


def show_saved_pred_err(
    savedir="saved", model_filter=None, entropy_filter=None
) -> None:
    results_csv = "test_err_per_t.csv"
    # find results_csv files
    csv_df_l = [
        pd.read_csv(join(savedir, dir_name, file_name))
        for dir_name in os.listdir(savedir)
        if isdir(join(savedir, dir_name))
        for file_name in os.listdir(join(savedir, dir_name))
        if file_name == results_csv
    ]
    assert csv_df_l, f"no <savedir>/<model>/{results_csv} files"
    df_compare = pd.concat(csv_df_l, ignore_index=True)

    # create vis table
    t_range = [c for c in df_compare.columns if c.isnumeric()]
    props = "text-decoration: underline"
    if model_filter:
        df_compare = df_compare.loc[df_compare["model_name"].isin(model_filter)]
    if entropy_filter:
        df_compare = df_compare.loc[df_compare["actS_c"].isin(entropy_filter)]
    output = (
        df_compare.sort_values(by=t_range)
        .style.background_gradient(axis=0, cmap="coolwarm")
        .highlight_min(subset=t_range, props=props)
        .highlight_max(subset=t_range, props=props)
    )
    show_or_save(output, savedir, "compare_evaluate")
