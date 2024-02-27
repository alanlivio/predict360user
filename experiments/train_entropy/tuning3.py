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
    cfg.name = f"{cfg.model},tuni3={cfg.train_entropy}"
    wandb.init(project="predict360user", name=cfg.name, resume=resume)
    log.info(f"\nruning {cfg.name} with {cfg}\n")

    # set seed
    p3u.set_random_seed(cfg.seed)

    # load dataset
    df_wins = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df_wins = p3u.split(
        df_wins,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
    )

    # split for tuning
    train_wins = df_wins[df_wins["partition"] == "train"]
    val_wins = df_wins[df_wins["partition"] == "val"]
    train_wins_tuning = train_wins[train_wins["actS_c"] == cfg.train_entropy]
    val_wins_tuning = val_wins[val_wins["actS_c"] == cfg.train_entropy]
    df_wins_pretuning = df_wins.drop(train_wins_tuning.index).drop(
        val_wins_tuning.index
    )
    df_wins_tuning = pd.concat([train_wins_tuning, val_wins_tuning])

    # log train len
    len_keys = ["train_len", "train_len_low", "train_len_medium", "train_len_high"]
    len_values = p3u.count_entropy(df_wins_pretuning[df_wins_pretuning["partition"] == "train"])
    wandb.run.summary.update(dict(zip(len_keys, len_values)))
    len_keys = ["tuni_len", "tuni_len_low", "tuni_len_medium", "tuni_len_high"]
    len_values = p3u.count_entropy(df_wins_tuning[df_wins_tuning["partition"] == "train"])
    wandb.run.summary.update(dict(zip(len_keys, len_values)))

    # fit
    model = p3u.get_model(cfg)
    model.fit(df_wins_pretuning)

    # tuning for more 1/3 epochs
    model.cfg.initial_epoch = wandb.run.step
    model.cfg.epochs = math.ceil(cfg.epochs * (1.33))
    model.cfg.lr = 0.0001
    log.info(f"\ntuning for {cfg.train_entropy} with {model.cfg=}\n")
    model.model.layers[0].trainable = False
    model.model.layers[1].trainable = False
    model.model.layers[2].trainable = False
    model.fit(df_wins_tuning)

    # evaluate model
    model.evaluate(df_wins)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = RunConfig(**oc.from_cli())  # type: ignore
    for seed in range(0, 3):
        cfg.seed = seed
        try:
            run(cfg)
        except:
            run(cfg, resume=True)
