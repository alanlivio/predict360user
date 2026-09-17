import logging

from omegaconf import OmegaConf as oc

import predict360user as p3u

log = logging.getLogger()


def run(cfg: p3u.RunConfig, **kwargs) -> dict:
    cfg.name = cfg.model
    log.info(f"==> run {cfg.name} with {cfg}")

    p3u.set_random_seed(cfg.seed)

    df = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df = p3u.split(
        df, train_size=cfg.train_size, test_size=cfg.test_size, seed=cfg.seed
    )

    len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
    len_values = p3u.count_entropy(df[df["partition"] == "train"])
    log.info(f"Dataset partitions: {dict(zip(len_keys, len_values))}")

    model = p3u.get_model(cfg)
    model.fit(df)

    eval_results = model.evaluate(df)
    log.info(f"Evaluation completed for seed {cfg.seed}: {eval_results}")

    del df
    del model
    return eval_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = oc.from_cli()
    nseeds = args.pop("nseeds", 1)
    results = []
    for seed in range(0, nseeds):
        cfg = p3u.RunConfig(**args)
        cfg.seed = seed
        res = run(cfg)
        results.append(res)

