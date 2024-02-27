import logging

from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


def run(cfg: p3u.RunConfig, resume=False) -> None:
    cfg.name = cfg.model
    wandb.init(project="predict360user", name=cfg.name, resume=resume)
    log.info(f"\nruning {cfg.name} with {cfg}\n")

    # seed
    p3u.set_random_seed(cfg.seed)

    # ingestion
    df_wins = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df_wins = p3u.split(
        df_wins, train_size=cfg.train_size, test_size=cfg.test_size, seed=cfg.seed
    )
    _, n_low, n_medium, n_high = p3u.count_entropy(
        df_wins[df_wins["partition"] == "train"]
    )
    wandb.run.summary.update({"trn_low": n_low, "trn_med": n_medium, "trn_hig": n_high})

    # fit model
    model = p3u.get_model(cfg)
    model.fit(df_wins)

    # evaluate model
    model.evaluate(df_wins)
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
