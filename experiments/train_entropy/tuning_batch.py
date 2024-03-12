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


def run(cfg: RunConfig, **kwargs) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},btuni={cfg.train_entropy}"
    wandb.init(project=cfg.project, name=cfg.name, **kwargs)
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
        seed=cfg.seed,
    )

    # tuning split
    train = df[df["partition"] == "train"]
    assert not train.empty
    begin = shuffle(train[train["actS_c"] != cfg.train_entropy])
    end = shuffle(train[train["actS_c"] == cfg.train_entropy])
    train_for_fit = pd.concat([begin, end])  # type: ignore
    assert train_for_fit.iloc[-1]["actS_c"] == cfg.train_entropy
    val = df[df["partition"] == "val"]
    assert not val.empty
    df_for_fit = pd.concat([train_for_fit, val])

    # log train len
    len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
    len_values = p3u.count_entropy(train_for_fit)
    wandb.log(dict(zip(len_keys, len_values)), step=0)

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
    args = oc.from_cli()
    nseeds = args.pop('nseeds', 1)
    for seed in range(0, nseeds):
        CFG = RunConfig(**args)  # type: ignore
        CFG.seed = seed
        try:
            run(CFG)
        except:
            log.info(f"==> rerun falied {wandb.run.id} ")
            run(CFG, resume="must", id=wandb.run.id) # resume using same id
