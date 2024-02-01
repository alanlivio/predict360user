from omegaconf import OmegaConf
import logging
from dataclasses import dataclass
import wandb
from sklearn.utils import shuffle
import logging

import pandas as pd
import predict360user as p3u

log = logging.getLogger()


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = ""


def run(cfg: RunConfig) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    run_name = f"{cfg.model_name},btuni={cfg.train_entropy}"
    wandb.init(project="predict360user", name=run_name)
    wandb.run.log({"model": cfg.model_name, "batch": cfg.batch_size, "lr": cfg.lr})
    log.info(f"run {run_name} config is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")

    # seed
    cfg.set_random_seed()
    
    # -- load dataset --
    df_wins = p3u.load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
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

    # -- fit --
    model = p3u.build_model(cfg)

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
    cfg = RunConfig(**OmegaConf.from_cli())
    run(cfg)
