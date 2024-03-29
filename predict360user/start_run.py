import logging

from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


def run(cfg: p3u.RunConfig, **kwargs) -> None:
    cfg.name = cfg.model
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
        df, train_size=cfg.train_size, test_size=cfg.test_size, seed=cfg.seed
    )

    # log train len
    len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
    len_values = p3u.count_entropy(df[df["partition"] == "train"])
    wandb.log(dict(zip(len_keys, len_values)), step=0)

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
    args = oc.from_cli()
    nseeds = args.pop('nseeds', 1)
    for seed in range(0, nseeds):
        CFG = p3u.RunConfig(**args)  # type: ignore
        CFG.seed = seed
        try:
            run(CFG)
        except:
            log.info(f"==> rerun falied {wandb.run.id} ")
            run(CFG, resume="must", id=wandb.run.id) # resume using same id

