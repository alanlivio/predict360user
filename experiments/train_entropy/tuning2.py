import logging
import math
from dataclasses import asdict, dataclass

import pandas as pd
from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = ""


def run(cfg: RunConfig) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.experiment_name = f"{cfg.model},tuni2={cfg.train_entropy}"
    wandb.init(project="predict360user", name=cfg.experiment_name, config=asdict(cfg))
    log.info(f"run {cfg.experiment_name} config is: \n--\n" + oc.to_yaml(cfg) + "--")

    # seed
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
    _, n_low, n_medium, n_high = p3u.count_entropy(
        df_wins[df_wins["partition"] == "train"]
    )
    wandb.run.log({"trn_low": n_low, "trn_med": n_medium, "trn_hig": n_high})

    # split for tuning
    train_wins = df_wins[df_wins["partition"] == "train"]
    val_wins = df_wins[df_wins["partition"] == "val"]
    train_wins_tuning = train_wins[train_wins["actS_c"] == cfg.train_entropy]
    val_wins_tuning = val_wins[val_wins["actS_c"] == cfg.train_entropy]
    df_wins_pretuning = df_wins.drop(train_wins_tuning.index).drop(
        val_wins_tuning.index
    )
    df_wins_tuning = pd.concat([train_wins_tuning, val_wins_tuning])

    # fit
    model = p3u.build_model(cfg)
    tuning_epochs_prc = 0.33
    model.fit(df_wins_pretuning)
    del model

    # tuning
    model = p3u.build_model(cfg)
    cfg.epochs = math.ceil(cfg.epochs * (1 + tuning_epochs_prc))
    cfg.lr = 0.0001
    model.fit(df_wins_tuning)

    # evaluate model
    model.evaluate(df_wins)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = RunConfig(**oc.from_cli())  # type: ignore
    run(cfg)
