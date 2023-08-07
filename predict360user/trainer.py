import logging
import os
from dataclasses import dataclass
from os.path import basename, exists, join, isdir
from typing import Generator

import absl.logging
import numpy as np
import pandas as pd
import plotly.express as px

from hydra.core.config_store import ConfigStore
from keras.callbacks import CSVLogger, ModelCheckpoint
from omegaconf import OmegaConf
from sklearn.utils import shuffle
from tqdm.auto import tqdm

from predict360user.dataset import Dataset, get_class_name, get_class_thresholds
from predict360user.models import BaseModel, Interpolation, NoMotion, PosOnly, PosOnly3D
from predict360user.utils import calc_actual_entropy, orth_dist_cartesian, show_or_save

ARGS_ENTROPY_NAMES = ["all", "low", "medium", "high", "nohigh", "nolow", "allminsize"]
ARGS_MODEL_NAMES = [
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
ARGS_DS_NAMES = ["all", "david", "fan", "nguyen", "xucvpr", "xupami"]
ARGS_ENTROPY_AUTO_NAMES = ["auto", "auto_m_window", "auto_since_start"]
log = logging.getLogger(basename(__file__))

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
tqdm.pandas()


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

    def __post_init__(self) -> None:
        assert self.model_name in ARGS_MODEL_NAMES
        assert self.dataset_name in ARGS_DS_NAMES
        assert self.train_entropy in ARGS_ENTROPY_NAMES + ARGS_ENTROPY_AUTO_NAMES

    def __str__(self) -> str:
        return OmegaConf.to_yaml(self)


cs = ConfigStore.instance()
cs.store(name="trainer", group="trainer", node=TrainerCfg)


class Trainer:
    cfg: TrainerCfg

    def __init__(self, cfg: TrainerCfg) -> None:
        log.info("TrainerCfg:\n-------\n" + OmegaConf.to_yaml(cfg) + "-------")
        self.cfg = cfg

        if self.cfg.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)
            log.info(f"set visible cpu to {self.cfg.gpu_id}")

        # properties others
        self.using_auto = self.cfg.train_entropy.startswith("auto")
        self.entropy_type = "actS"
        if self.cfg.dataset_name == "all" and self.cfg.train_entropy == "all":
            self.model_fullname = self.cfg.model_name
        elif self.cfg.train_entropy == "all":
            self.model_fullname = f"{self.cfg.model_name},{self.cfg.dataset_name},,"
        else:
            self.model_fullname = f"{self.cfg.model_name},{self.cfg.dataset_name},{self.entropy_type},{self.cfg.train_entropy}"
        self.model_dir = join(self.cfg.savedir, self.model_fullname)
        self.train_csv_log_f = join(self.model_dir, "train_results.csv")
        self.model_path = join(self.model_dir, "weights.hdf5")
        self.model: BaseModel
        if self.cfg.model_name == "pos_only":
            self.model = PosOnly(self.cfg)
        elif self.cfg.model_name == "pos_only_3d":
            self.model = PosOnly3D(self.cfg)
        elif self.cfg.model_name == "interpolation":
            self.model = Interpolation(self.cfg)
        elif self.cfg.model_name == "no_motion":
            self.model = NoMotion(self.cfg)
        else:
            raise RuntimeError

    def generate_batchs(self, model: BaseModel, wins: list) -> Generator:
        while True:
            shuffle(wins, random_state=1)
            for count, _ in enumerate(wins[:: self.cfg.batch_size]):
                end = (
                    count + self.cfg.batch_size
                    if count + self.cfg.batch_size <= len(wins)
                    else len(wins)
                )
                traces_l = [
                    self.ds.get_traces(win["video"], win["user"])
                    for win in wins[count:end]
                ]
                x_i_l = [win["trace_id"] for win in wins[count:end]]
                yield model.generate_batch(traces_l, x_i_l)

    def drop_predict_cols(self) -> None:
        col_rm = [
            col
            for col in self.ds.df.columns
            for model in ARGS_MODEL_NAMES
            if col.startswith(model)
        ]
        self.ds.df.drop(col_rm, axis=1, errors="ignore", inplace=True)

    def _auto_select_model(self, traces: np.array, x_i) -> BaseModel:
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

    def run(self) -> None:
        if not exists(self.model_dir):
            os.makedirs(self.model_dir)
        log.info("model_dir=" + self.model_dir)

        if not hasattr(self, "ds"):
            log.info("loading dataset ...")
            # TODO: filter by dataset name
            self.ds = Dataset(savedir=self.cfg.savedir)
            self.ds.partition(
                entropy_filter=self.cfg.train_entropy,
                train_size=self.cfg.train_size,
                test_size=self.cfg.test_size,
            )
            self.ds.create_wins(
                init_window=self.cfg.init_window, h_window=self.cfg.h_window
            )

        if not self.using_auto and self.cfg.model_name not in MODELS_NAMES_NO_TRAIN:
            log.info("train ...")
            if exists(self.model_path):
                self.model.load_weights(self.model_path)

            # setting initial_epoch
            initial_epoch = 0
            if exists(self.train_csv_log_f):
                lines = pd.read_csv(self.train_csv_log_f)
                lines.dropna(how="all", inplace=True)
                done_epochs = int(lines.iloc[-1]["epoch"]) + 1
                assert done_epochs <= self.cfg.epochs
                initial_epoch = done_epochs
                log.info(f"train_csv_log_f has {initial_epoch} epochs ")

            if initial_epoch >= self.cfg.epochs:
                log.info(
                    f"train_csv_log_f has {initial_epoch}>={self.cfg.epochs}. not training."
                )
            else:
                # fit
                steps_per_ep_train = np.ceil(
                    len(self.ds.x_train_wins) / self.cfg.batch_size
                )
                steps_per_ep_validate = np.ceil(
                    len(self.ds.x_val_wins) / self.cfg.batch_size
                )
                csv_logger = CSVLogger(self.train_csv_log_f, append=True)
                # https://www.tensorflow.org/tutorials/keras/save_and_load
                model_checkpoint = ModelCheckpoint(
                    self.model_path, save_weights_only=True, verbose=1
                )
                callbacks = [csv_logger, model_checkpoint]
                generator = self.generate_batchs(self.model, self.ds.x_train_wins)
                validation_data = self.generate_batchs(self.model, self.ds.x_val_wins)
                self.model.fit(
                    x=generator,
                    verbose=1,
                    steps_per_epoch=steps_per_ep_train,
                    validation_data=validation_data,
                    validation_steps=steps_per_ep_validate,
                    validation_freq=self.cfg.batch_size,
                    epochs=self.cfg.epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                )

        log.info("evaluate ...")
        if self.using_auto:
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

        # auxiliary df based on x_test_wins to calculate error
        pred_range = range(self.cfg.h_window)
        df = pd.DataFrame(self.ds.x_test_wins).set_index(["user", "video", "trace_id"])
        df.sort_index(inplace=True)

        def _save_pred(row) -> None:
            # row.name return the index (user, video, time)
            user, video, x_i = row.name[0], row.name[1], row.name[2]
            traces = self.ds.get_traces(video, user)
            # predict
            if self.using_auto:
                pred = self._auto_select_model(traces, x_i).predict_for_sample(
                    traces, x_i
                )
            else:
                pred = self.model.predict_for_sample(traces, x_i)
            assert len(pred) == self.cfg.h_window
            pred_true = traces[x_i + 1 : x_i + self.cfg.h_window + 1]
            error_per_t = [
                orth_dist_cartesian(pred[t], pred_true[t]) for t in pred_range
            ]
            # save prediction
            return error_per_t

        # calculate predictions
        tqdm.pandas(desc=f"evaluate model {self.model_fullname}")
        df = pd.concat(
            [df, df.progress_apply(_save_pred, axis=1, result_type="expand")], axis=1
        )

        # save at evaluate_results.csv
        columns = ["model_name", "S_class", "mean_err"]
        df_evaluate_res = pd.DataFrame(columns=columns + pred_range, dtype=np.float32)
        targets = [
            ("all", pd.Series(True, df.index)),
            ("low", df["actS_c"] == "low"),
            ("nolow", df["actS_c"] != "low"),
            ("medium", df["actS_c"] == "medium"),
            ("nohigh", df["actS_c"] != "high"),
            ("high", df["actS_c"] == "high"),
        ]
        for S_class, idx in targets:
            mean_err = np.nanmean(df[pred_range].values)
            newid = len(df_evaluate_res)
            new_row = [self.model_fullname, S_class, mean_err] + list(
                df.loc[idx, pred_range].mean()
            )
            df_evaluate_res.loc[newid] = new_row

        log.info(f"saving evaluate_results.csv")
        df_evaluate_res.to_csv(
            join(self.model_dir, "evaluate_results.csv"), index=False
        )

    #
    # compare-related methods TODO: replace then by a log in a model registry
    #

    def compare_train(self) -> None:
        results_csv = "train_results.csv"
        # find results_csv files
        csv_df_l = [
            (dir_name, pd.read_csv(join(self.cfg.savedir, dir_name, file_name)))
            for dir_name in os.listdir(self.cfg.savedir)
            if isdir(join(self.cfg.savedir, dir_name))
            for file_name in os.listdir(join(self.cfg.savedir, dir_name))
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
        show_or_save(fig, self.cfg.savedir, "compare_train")

    def compare_evaluate(self, model_filter=None, entropy_filter=None) -> None:
        results_csv = "evaluate_results.csv"
        # find results_csv files
        csv_df_l = [
            pd.read_csv(join(self.cfg.savedir, dir_name, file_name))
            for dir_name in os.listdir(self.cfg.savedir)
            if isdir(join(self.cfg.savedir, dir_name))
            for file_name in os.listdir(join(self.cfg.savedir, dir_name))
            if file_name == results_csv
        ]
        assert csv_df_l, f"no <savedir>/<model>/{results_csv} files"
        df_compare = pd.concat(csv_df_l, ignore_index=True)

        # create vis table
        pred_range = [str(col) for col in range(self.cfg.h_window)]
        props = "text-decoration: underline"
        if model_filter:
            df_compare = df_compare.loc[df_compare["model_name"].isin(model_filter)]
        if entropy_filter:
            df_compare = df_compare.loc[df_compare["S_class"].isin(entropy_filter)]
        output = (
            df_compare.sort_values(by=pred_range)
            .style.background_gradient(axis=0, cmap="coolwarm")
            .highlight_min(subset=pred_range, props=props)
            .highlight_max(subset=pred_range, props=props)
        )
        show_or_save(output, self.cfg.savedir, "compare_evaluate")
