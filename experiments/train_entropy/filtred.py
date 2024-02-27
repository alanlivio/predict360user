import logging
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf as oc

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
    train_minsize=False,
) -> pd.DataFrame:
    assert train_entropy in p3u.ENTROPY_NAMES
    df["partition"] = "discarted"
    log.info(f"{train_size=} (with {val_size=}), {test_size=}")

    # split train and test
    train, test = p3u.train_test_split(
        df,
        random_state=seed,
        train_size=train_size,
        test_size=test_size,
        stratify=df["actS_c"],
    )
    log.info("train trajecs are " + p3u.count_entropy_str(train))

    # filter by given entropy
    filtered = train[train["actS_c"] == train_entropy]
    assert len(filtered)

    # filter for limiting to smallest class size in the train
    if train_minsize:
        target_size = train["actS_c"].value_counts().min()
        # stratify https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas
        n_current_classes = len(filtered["actS_c"].unique())
        n_sample_per_class = int(target_size / n_current_classes)
        filtered = filtered.groupby("actS_c", group_keys=False).apply(
            lambda x: x.sample(n=n_sample_per_class, random_state=seed)
        )

    # split train and val
    train_before_val_split = len(filtered)
    train, val = p3u.train_test_split(
        filtered,
        random_state=seed,
        test_size=val_size,
        stratify=filtered["actS_c"],
    )
    log.info("filtred train trajecs are " + p3u.count_entropy_str(train))
    log.info("filtred train.val trajecs are " + p3u.count_entropy_str(val))

    # save partition as column
    df.loc[train.index, "partition"] = "train"
    df.loc[val.index, "partition"] = "val"
    df.loc[test.index, "partition"] = "test"
    train_len = len(df[df["partition"] == "train"])
    val_len = len(df[df["partition"] == "val"])
    assert (train_len + val_len) == train_before_val_split

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
    log.info(f"-- run {cfg.name} with {cfg}")
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
        train_minsize=cfg.train_minsize,
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
        cfg = RunConfig(**oc.from_cli())  # type: ignore
        cfg.seed = seed
        try:
            run(cfg)
        except:
            run(cfg, resume=True)
