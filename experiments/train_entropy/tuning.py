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


def run(cfg: RunConfig, **kwargs) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},tuni={cfg.train_entropy}"
    wandb.init(project="predict360user", name=cfg.name, **kwargs)
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

    # split for tuning
    train = df[df["partition"] == "train"]
    assert not train.empty
    val = df[df["partition"] == "val"]
    assert not val.empty
    train_tuning = train[train["actS_c"] == cfg.train_entropy]
    val_tuning = val[val["actS_c"] == cfg.train_entropy]
    df_pretuning = df.drop(train_tuning.index).drop(val_tuning.index)
    df_tuning = pd.concat([train_tuning, val_tuning])

    # log train len
    len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
    len_values = p3u.count_entropy(df_pretuning[df_pretuning["partition"] == "train"])
    wandb.log(dict(zip(len_keys, len_values)))
    len_keys = ["samples/tuni/all", "samples/tuni/low", "samples/tuni/medi", "samples/tuni/high"]
    len_values = p3u.count_entropy(df_tuning[df_tuning["partition"] == "train"])
    wandb.log(dict(zip(len_keys, len_values)))

    # fit
    model = p3u.get_model(cfg)
    model.fit(df_pretuning)

    # tuning for more 1/3 epochs
    model.cfg.initial_epoch = wandb.run.step
    model.cfg.epochs = math.ceil(cfg.epochs * (1.33))
    log.info(f"==> tuni {cfg.train_entropy} with {model.cfg=}")
    model.fit(df_tuning)

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
            run(CFG, resume="must", id=wandb.run.id) # resume using same id
