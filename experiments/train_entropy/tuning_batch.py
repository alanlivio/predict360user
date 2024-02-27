import logging
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf as oc
from sklearn.utils import shuffle

import predict360user as p3u
import wandb

log = logging.getLogger()


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = ""


def run(cfg: RunConfig, resume=False) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},btuni={cfg.train_entropy}"
    wandb.init(project="predict360user", name=cfg.name, resume=resume)
    log.info(f"\nruning {cfg.name} with {cfg}\n")
    
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
    wandb.run.summary.update({"trn_low": n_low, "trn_med": n_medium, "trn_hig": n_high})

    # train_wins_for_fit
    train_wins = df_wins[df_wins["partition"] == "train"]
    begin = shuffle(train_wins[train_wins["actS_c"] != cfg.train_entropy])
    end = shuffle(train_wins[train_wins["actS_c"] == cfg.train_entropy])
    train_wins_for_fit = pd.concat([begin, end])  # type: ignore
    assert train_wins_for_fit.iloc[-1]["actS_c"] == cfg.train_entropy

    # df_wins_new with train parition ordered
    val_wins = df_wins[df_wins["partition"] == "val"]
    df_wins_for_fit = pd.concat([train_wins_for_fit, val_wins])

    # fit model
    model = p3u.get_model(cfg)
    model.fit(df_wins_for_fit)

    # evaluate model
    model.evaluate(df_wins)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = RunConfig(**oc.from_cli())  # type: ignore
    for seed in range(0,3):
        cfg.seed = seed
        try:
            run(cfg)
        except:
            run(cfg, resume=True)