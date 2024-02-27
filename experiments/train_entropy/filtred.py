import logging
from dataclasses import dataclass

from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


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
    log.info(f"-- runing {cfg.name} with {cfg}")
    # set seed
    p3u.set_random_seed(cfg.seed)

    # load dataset
    df = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df = p3u.split_train_filtred(
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
