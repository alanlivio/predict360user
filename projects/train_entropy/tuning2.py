from omegaconf import OmegaConf
import logging
from dataclasses import dataclass
import math
import pandas as pd
import wandb

from predict360user.train import build_model, fit_keras, evaluate
from predict360user.ingest import count_entropy, load_df_wins, split
from predict360user.model_wrapper import ModelConf, build_run_name, ENTROPY_NAMES_UNIQUE

log = logging.getLogger()


@dataclass
class RunConf(ModelConf):
    train_entropy: str = ""


def run(cfg: RunConf) -> None:
    build_run_name(cfg)
    assert cfg.train_entropy in ENTROPY_NAMES_UNIQUE
    cfg.run_name += f",tuni2={cfg.train_entropy}"
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

    # split for tuning
    train_wins = df_wins[df_wins["partition"] == "train"]
    val_wins = df_wins[df_wins["partition"] == "val"]
    tuning_prc = 0.33
    train_wins_tuning = train_wins[train_wins["actS_c"] == cfg.train_entropy].sample(
        frac=tuning_prc, random_state=1
    )
    val_wins_tuning = val_wins[val_wins["actS_c"] == cfg.train_entropy].sample(
        frac=tuning_prc, random_state=1
    )
    wins_pretuning = df_wins.drop(train_wins_tuning.index).drop(val_wins_tuning.index)
    wins_tuning = pd.concat([train_wins_tuning, val_wins_tuning])

    # fit 1
    model.fit(wins_pretuning)
    del model
    # fit 2
    cfg.epochs = math.ceil(cfg.epochs + cfg.epochs * tuning_prc)
    cfg.lr = 0.0001
    model = build_model(cfg)
    model.fit(wins_tuning)

    # evaluate model
    model.evaluate(cfg, model, df_wins)
    wandb.finish()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = OmegaConf.merge(OmegaConf.structured(RunConf), OmegaConf.from_cli())
    run(cfg)
