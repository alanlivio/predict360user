from omegaconf import OmegaConf
import logging
from dataclasses import dataclass
import wandb
from sklearn.utils import shuffle
import logging

import pandas as pd

from predict360user.train import build_model
from predict360user.ingest import count_entropy, load_df_wins, split
from predict360user.model_wrapper import (
    ModelConf,
    ENTROPY_NAMES_UNIQUE,
    build_run_name,
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
    )

    # -- fit --
    model = build_model(cfg)

    # train_wins_for_fit
    train_wins = df_wins[df_wins["partition"] == "train"]
    begin = shuffle(train_wins[train_wins["actS_c"] != cfg.train_entropy])
    end = shuffle(train_wins[train_wins["actS_c"] == cfg.train_entropy])
    train_wins_for_fit = pd.concat([begin, end])
    assert train_wins_for_fit.iloc[-1]["actS_c"] == cfg.train_entropy

    # df_wins_new with train parition ordered
    val_wins = df_wins[df_wins["partition"] == "val"]
    df_wins_for_fit = pd.concat([train_wins_for_fit, val_wins])

    # fit model
    model.fit(df_wins_for_fit)

    # evaluate model
    model.evaluate(df_wins)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = OmegaConf.merge(OmegaConf.structured(RunConf), OmegaConf.from_cli())
    run(cfg)
