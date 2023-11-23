from omegaconf import OmegaConf
import logging
from dataclasses import dataclass
import wandb
from sklearn.utils import shuffle
import logging
import os
from os.path import exists, join

import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint
from wandb.keras import WandbMetricsLogger

from predict360user.train import build_model, evaluate
from predict360user.ingest import count_entropy, load_df_wins, split
from predict360user.model_config import (
    batch_generator,
    ModelConf,
    ENTROPY_NAMES_UNIQUE,
    build_run_name,
    TRAIN_RES_CSV,
)

log = logging.getLogger()


@dataclass
class RunConf(ModelConf):
    train_entropy: str = ""


def run(cfg: RunConf) -> None:
    build_run_name(cfg)
    assert cfg.train_entropy in ENTROPY_NAMES_UNIQUE
    cfg.run_name += f",btuni={cfg.train_entropy}"
    log.info(f"run conf is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")

    # -- load dataset --
    df_wins = load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = split(
        df_wins,
        train_size=cfg.train_size,
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
        name=cfg.run_name,
        resume=True,
    )

    # -- fit --
    model = build_model(cfg)

    train_wins = df_wins[df_wins["partition"] == "train"]
    val_wins = df_wins[df_wins["partition"] == "val"]

    train_wins = (train_wins[train_wins["partition"] == "train"],)
    begin = shuffle(train_wins[train_wins["actS_c"] != cfg.train_entropy])
    end = shuffle(train_wins[train_wins["actS_c"] == cfg.train_entropy])
    train_wins = train_wins.concat([begin, end])
    assert train_wins.iloc[-1]["actS_c"] == cfg.train_entropy

    model_dir = join(cfg.savedir, cfg.run_name)
    train_csv_log_f = join(model_dir, TRAIN_RES_CSV)
    model_path = join(model_dir, "weights.hdf5")

    if not exists(model_dir):
        os.makedirs(model_dir)
    if exists(model_path):
        model.load_weights(model_path)
    log.info("model_path=" + model_path)

    # calc initial_epoch
    initial_epoch = 0
    if exists(train_csv_log_f):
        lines = pd.read_csv(train_csv_log_f)
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
            CSVLogger(train_csv_log_f, append=True),
            ModelCheckpoint(
                model_path,
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

    # evaluate and log to wandb
    err_per_class_dict = evaluate(cfg, model, df_wins)
    for actS_c, err in err_per_class_dict.items():
        wandb.run.summary[f"err_{actS_c}"] = err["mean"]
        table = wandb.Table(data=err["mean_per_t"], columns=["t", "err"])
        plot_id = f"test_err_per_t_class_{actS_c}"
        plot = wandb.plot.line(table, "t", "err", title=plot_id)
        wandb.log({plot_id: plot})
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = OmegaConf.merge(OmegaConf.structured(RunConf), OmegaConf.from_cli())
    run(cfg)
