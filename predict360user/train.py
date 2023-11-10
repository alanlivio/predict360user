import datetime
import logging
import os
import sys
from os.path import abspath, basename, exists, isabs, isdir, join
from types import MethodType

import absl.logging
import IPython
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from keras.callbacks import CSVLogger, ModelCheckpoint
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from wandb.keras import WandbMetricsLogger

import wandb
from predict360user.ingest import (
    count_entropy,
    get_class_name,
    get_class_thresholds,
    load_df_wins,
    split,
)
from predict360user.model_config import (
    DEFAULT_SAVEDIR,
    EVAL_RES_CSV,
    TRAIN_RES_CSV,
    BaseModel,
    Config,
    batch_generator,
)
from predict360user.models import TRACK, Interpolation, NoMotion, PosOnly, PosOnly3D
from predict360user.utils.math360 import calc_actual_entropy, orth_dist_cartesian

log = logging.getLogger(basename(__file__))

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


def train_and_eval(cfg: Config) -> None:
    assert cfg.model_name in MODEL_NAMES
    assert cfg.train_entropy in ENTROPY_NAMES + ENTROPY_AUTO_NAMES
    log.info("train_and_eval using config:\n---\n" + OmegaConf.to_yaml(cfg) + "----")
    log.info(f"model_dir={cfg.model_dir}")
    df_wins = load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = split(
        df_wins,
        train_size=cfg.train_size,
        train_entropy=cfg.train_entropy,
        train_minsize=cfg.minsize,
        test_size=cfg.test_size,
    )
    _, n_low, n_medium, n_high = count_entropy(df_wins[df_wins["partition"] == "train"])
    wandb.init(
        project="predict360user",
        config={
            "model_name": cfg.model_name,
            "train_entropy": cfg.train_entropy,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "train_n_low": n_low,
            "train_n_medium": n_medium,
            "train_n_high": n_high,
        },
        name=cfg.model_fullname,
    )
    model = build_model(cfg)
    train(cfg, model, df_wins)
    evaluate(cfg, model, df_wins)
    wandb.finish()


def build_model(cfg) -> None:
    if cfg.model_name == "pos_only":
        return PosOnly(cfg)
    elif cfg.model_name == "pos_only_3d":
        return PosOnly3D(cfg)
    elif cfg.model_name == "interpolation":
        return Interpolation(cfg)
    elif cfg.model_name == "TRACK":
        return TRACK(cfg)
    elif cfg.model_name == "no_motion":
        return NoMotion(cfg)
    else:
        # TODO: initialize using config _target_
        raise RuntimeError


def train(cfg: Config, model: BaseModel, df_wins: pd.DataFrame) -> None:
    log.info("train ...")
    if cfg.train_entropy.startswith("auto") or (
        cfg.model_name in MODELS_NAMES_NO_TRAIN
    ):
        return
    if not exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)
    log.info("model_dir=" + cfg.model_dir)

    if exists(cfg.model_path):
        model.load_weights(cfg.model_path)

    train_wins = df_wins[df_wins["partition"] == "train"]
    val_wins = df_wins[df_wins["partition"] == "val"]
    # calc initial_epoch
    initial_epoch = 0
    if exists(cfg.train_csv_log_f):
        lines = pd.read_csv(cfg.train_csv_log_f)
        lines.dropna(how="all", inplace=True)
        done_epochs = int(lines.iloc[-1]["epoch"]) + 1
        assert done_epochs <= cfg.epochs
        initial_epoch = done_epochs
        log.info(f"train_csv_log_f has {initial_epoch} epochs ")

    # fit
    if cfg.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        log.info(f"set visible cpu to {cfg.gpu_id}")
    if initial_epoch >= cfg.epochs:
        log.info(f"train_csv_log_f has {initial_epoch}>={cfg.epochs}. not training.")
    else:
        steps_per_ep_train = np.ceil(len(train_wins) / cfg.batch_size)
        steps_per_ep_validate = np.ceil(len(val_wins) / cfg.batch_size)
        callbacks = [
            CSVLogger(cfg.train_csv_log_f, append=True),
            ModelCheckpoint(
                cfg.model_path,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                period=1,
            ),
            WandbMetricsLogger(initial_global_step=initial_epoch),
        ]
        model.fit_generator(
            generator=batch_generator(model, train_wins, cfg.batch_size),
            validation_data=batch_generator(model, val_wins, cfg.batch_size),
            steps_per_epoch=steps_per_ep_train,
            validation_steps=steps_per_ep_validate,
            epochs=cfg.epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=2,
        )


def evaluate(cfg: Config, model: BaseModel, df_wins: pd.DataFrame) -> None:
    log.info("evaluate ...")
    if cfg.train_entropy.startswith("auto"):  # will not use model
        _set_predict_by_entropy(model)

    test_wins = df_wins[df_wins["partition"] == "test"]

    # calculate predictions errors
    t_range = list(range(cfg.h_window))

    def _calc_pred_err(row) -> list[float]:
        # return np.random.rand(cfg.h_window)  # for debugging
        traces = row["traces"]
        x_i = row["trace_id"]
        pred_true = row["h_window"]
        # predict
        pred = model.predict_for_sample(traces, x_i)
        assert len(pred) == cfg.h_window
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
    # save predications
    # 1) avg per class (as wandb summary): # err_all, err_low, err_nohigh, err_medium,
    # err_nolow, err_nolow, err_all, err_high
    # 2) avg err per t per class (as wandb line plots and as csv)
    targets = [
        ("all", test_wins.index),
        ("low", test_wins.index[test_wins["actS_c"] == "low"]),
        ("nohigh", test_wins.index[test_wins["actS_c"] != "high"]),
        ("medium", test_wins.index[test_wins["actS_c"] == "medium"]),
        ("nolow", test_wins.index[test_wins["actS_c"] != "low"]),
        ("high", test_wins.index[test_wins["actS_c"] == "high"]),
    ]
    df_test_err_per_t = pd.DataFrame(
        columns=["model_name", "actS_c"] + t_range,
        dtype=np.float32,
    )
    for actS_c, idx in targets:
        # 1)
        class_err = df_wins.loc[idx, t_range].values.mean()
        wandb.run.summary[f"err_{actS_c}"] = class_err
        # 2)
        class_err_per_t = df_wins.loc[idx, t_range].mean()
        data = [[x, y] for (x, y) in zip(t_range, class_err_per_t)]
        table = wandb.Table(data=data, columns=["t", "err"])
        plot_id = f"test_err_per_t_class_{actS_c}"
        plot = wandb.plot.line(table, "t", "err", title=plot_id)
        wandb.log({plot_id: plot})
        # save new row on csv
        df_test_err_per_t.loc[len(df_test_err_per_t)] = [
            cfg.model_fullname,  # target model
            actS_c,  # target class
        ] + list(class_err_per_t)
    log.info("saving eval_results.csv")
    df_test_err_per_t.to_csv(join(cfg.model_dir, EVAL_RES_CSV), index=False)


def _set_predict_by_entropy(model: BaseModel, cfg: Config, df_wins) -> BaseModel:
    prefix = join(cfg.savedir, f"{cfg.model_name},{cfg.dataset_name},actS,")
    threshold_medium, threshold_high = get_class_thresholds(df_wins, "actS")
    model_low = model.copy()
    model_low.load_weights(join(prefix + "low", "weights.hdf5"))
    model_medium = model.copy()
    model_medium.load_weights(join(prefix + "medium", "weights.hdf5"))
    model_high = model.copy()
    model_high.load_weights(join(prefix + "high", "weights.hdf5"))

    def _predict_by_entropy(self, traces: np.array, x_i: int) -> BaseModel:
        if cfg.train_entropy == "auto":
            window = traces
        elif cfg.train_entropy == "auto_m_window":
            window = traces[x_i - cfg.m_window : x_i]
        elif cfg.train_entropy == "auto_since_start":
            window = traces[0:x_i]
        else:
            raise RuntimeError()
        a_ent = calc_actual_entropy(window)
        actS_c = get_class_name(a_ent, threshold_medium, threshold_high)
        if actS_c == "low":
            return model_low.predict_for_sample(self, traces, x_i)
        if actS_c == "medium":
            return model_medium.predict_for_sample(self, traces, x_i)
        if actS_c == "high":
            return model_high.predict_for_sample(self, traces, x_i)
        else:
            raise RuntimeError()

    model.predict_for_sample = MethodType(_predict_by_entropy, model)
    return model


def show_or_save(output, savedir=DEFAULT_SAVEDIR, title="") -> None:
    if "ipykernel" in sys.modules:
        IPython.display.display(output)
    else:
        if not title:
            if isinstance(output, go.Figure):
                title = output.layout.title.text
            else:
                title = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        html_file = join(savedir, title + ".html")
        if isinstance(output, go.Figure):
            output.write_html(html_file)
        else:
            output.to_html(html_file)
        if not isabs(html_file):
            html_file = abspath(html_file)
        log.info(f"compare_train saved on {html_file}")


def compare_train_results(savedir="saved", model_filter=[]) -> None:
    # find results_csv files
    csv_df_l = [
        (dir_name, pd.read_csv(join(savedir, dir_name, file_name)))
        for dir_name in os.listdir(savedir)
        if isdir(join(savedir, dir_name))
        for file_name in os.listdir(join(savedir, dir_name))
        if file_name == TRAIN_RES_CSV
    ]
    csv_df_l = [df.assign(model_name=dir_name) for (dir_name, df) in csv_df_l]
    assert csv_df_l, f"no <savedir>/<model>/{TRAIN_RES_CSV} files"
    df_compare = pd.concat(csv_df_l, ignore_index=True)
    if model_filter:
        df_compare = df_compare.loc[df_compare["model_name"].isin(model_filter)]

    # plot
    fig = px.line(
        df_compare,
        x="epoch",
        y="loss",
        color="model_name",
        title="compare_loss",
        width=800,
    )
    show_or_save(fig, savedir, "compare_loss")
    fig = px.line(
        df_compare,
        x="epoch",
        y="val_loss",
        color="model_name",
        title="compare_val_loss",
        width=800,
    )
    show_or_save(fig, savedir, "compare_val_loss")


def compare_eval_results(savedir="saved", model_filter=[], entropy_filter=[]) -> None:
    # find results_csv files
    csv_df_l = [
        pd.read_csv(join(savedir, dir_name, file_name))
        for dir_name in os.listdir(savedir)
        if isdir(join(savedir, dir_name))
        for file_name in os.listdir(join(savedir, dir_name))
        if file_name == EVAL_RES_CSV
    ]
    assert csv_df_l, f"no <savedir>/<model>/{EVAL_RES_CSV} files"
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
