import logging

from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


def run(cfg: p3u.RunConfig, resume=False) -> None:
    cfg.name = cfg.model
    wandb.init(project="predict360user", name=cfg.name, resume=resume)
    log.info(f"\nruning {cfg.name} with {cfg}\n")

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
        df, train_size=cfg.train_size, test_size=cfg.test_size, seed=cfg.seed
    )

    # log train len
    len_keys = ["train_len", "train_len_low", "train_len_medium", "train_len_high"]
    len_values = p3u.count_entropy(df[df["partition"] == "train"])
    wandb.run.summary.update(dict(zip(len_keys, len_values)))

    # fit model
    model = p3u.get_model(cfg)
    model.fit(df)

    # evaluate model
    model.evaluate(df)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = p3u.RunConfig(**oc.from_cli())  # type: ignore
    for seed in range(0, 3):
        cfg.seed = seed
        try:
            run(cfg)
        except:
            run(cfg, resume=True)
