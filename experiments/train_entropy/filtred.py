import logging
import math
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf as oc
from sklearn.model_selection import train_test_split

import predict360user as p3u
import wandb

log = logging.getLogger()


def split_train_filtred(
    df: pd.DataFrame,
    train_entropy: str,
    train_size=p3u.RunConfig.train_size,
    test_size=p3u.RunConfig.test_size,
    seed=None,
    val_size=0.25,
) -> pd.DataFrame:
    assert train_entropy in p3u.ENTROPY_NAMES
    df["partition"] = "discarted"  # sanity check
    log.info(f"{train_size=} (with {val_size=}), {test_size=}")

    # split train (with full size) and test
    train, test = train_test_split(
        df,
        random_state=seed,
        train_size=1 - test_size,
        test_size=test_size,
        stratify=df["actS_c"],
    )

    # sample train like was sample full len(df) to create similar size
    n_train = min(math.ceil(len(df) * train_size), len(train[train["actS_c"] == train_entropy]))
    train = train[train["actS_c"] == train_entropy].sample(n=n_train, random_state=seed)
    train, val = train_test_split(
        train,
        random_state=seed,
        test_size=val_size,
        stratify=train["actS_c"],
    )
    log.info("filtred train trajecs are " + p3u.count_entropy_str(train))
    log.info("filtred train.val trajecs are " + p3u.count_entropy_str(val))

    # save partition as column
    df.loc[train.index, "partition"] = "train"
    df.loc[val.index, "partition"] = "val"
    df.loc[test.index, "partition"] = "test"

    return df


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = "all"
    train_minsize: bool = False


def run(cfg: RunConfig, resume=False) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},filt={cfg.train_entropy}"
    if cfg.train_minsize:
        cfg.name += f",mins={cfg.train_minsize!r}"
    wandb.init(project="predict360user", name=cfg.name, resume=resume)
    log.info("")
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
    df = split_train_filtred(
        df,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        train_entropy=cfg.train_entropy,
        seed=cfg.seed,
    )

    # log train len
    len_keys = ["train.all", "train.low", "train.medi", "train.high"]
    len_values = p3u.count_entropy(df[df["partition"] == "train"])
    wandb.run.summary.update(dict(zip(len_keys, len_values)))

    # fit model
    model = p3u.get_model(cfg)
    model.fit(df)

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
