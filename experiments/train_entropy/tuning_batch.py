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
    log.info(f"==> run {cfg.name} with {cfg}")
    
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

    # tuning split
    train = df[df["partition"] == "train"]
    begin = shuffle(train[train["actS_c"] != cfg.train_entropy])
    end = shuffle(train[train["actS_c"] == cfg.train_entropy])
    train_for_fit = pd.concat([begin, end])  # type: ignore
    assert train_for_fit.iloc[-1]["actS_c"] == cfg.train_entropy
    val = df[df["partition"] == "val"]
    df_for_fit = pd.concat([train_for_fit, val])

    # log train len
    len_keys = ["train.all", "train.low", "train.medi", "train.high"]
    len_values = p3u.count_entropy(train_for_fit)
    wandb.run.summary.update(dict(zip(len_keys, len_values)))

    # fit model
    model = p3u.get_model(cfg)
    model.fit(df_for_fit)

    # evaluate model
    model.evaluate(df)

    # finish run
    wandb.finish()
    del df
    del model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for seed in range(0, 3):
        CFG = RunConfig(**oc.from_cli())  # type: ignore
        CFG.seed = seed
        try:
            run(CFG)
        except:
            run(CFG, resume=True)
