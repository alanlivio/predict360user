import logging
import math
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = ""


def run(cfg: RunConfig, resume=False) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},tuni2={cfg.train_entropy}"
    wandb.init(project="predict360user", name=cfg.name, resume=resume)
    log.info(f"-- run {cfg.name} with {cfg}")
    
    # set seed
    p3u.set_random_seed(cfg.seed)

    # load dataset
    df = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df = p3u.split(
        df,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
    )

    # split for tuning
    train = df[df["partition"] == "train"]
    val = df[df["partition"] == "val"]
    train_tuning = train[train["actS_c"] == cfg.train_entropy]
    val_tuning = val[val["actS_c"] == cfg.train_entropy]
    df_pretuning = df.drop(train_tuning.index).drop(val_tuning.index)
    df_tuning = pd.concat([train_tuning, val_tuning])

    # log train len
    len_keys = ["train.all", "train.low", "train.medi", "train.high"]
    len_values = p3u.count_entropy(df_pretuning[df_pretuning["partition"] == "train"])
    wandb.run.summary.update(dict(zip(len_keys, len_values)))
    len_keys = ["tuni.all", "tuni.low", "tuni.medi", "tuni.high"]
    len_values = p3u.count_entropy(df_tuning[df_tuning["partition"] == "train"])
    wandb.run.summary.update(dict(zip(len_keys, len_values)))

    # fit
    model = p3u.get_model(cfg)
    model.fit(df_pretuning)

    # tuning for more 1/3 epochs
    model.cfg.initial_epoch = wandb.run.step
    model.cfg.epochs = math.ceil(cfg.epochs * (1.33))
    model.cfg.lr = 0.0001
    log.info(f"-- tuni {cfg.train_entropy} with {model.cfg=}")
    model.fit(df_tuning)

    # evaluate model
    model.evaluate(df)

    # finish run
    wandb.finish()
    del df
    del model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for seed in range(0, 3):
        cfg = RunConfig(**oc.from_cli())  # type: ignore
        cfg.seed = seed
        try:
            run(cfg)
        except:
            run(cfg, resume=True)
