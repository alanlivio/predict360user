from omegaconf import OmegaConf
import logging
import wandb
import predict360user as p3u

log = logging.getLogger()


def run(cfg: p3u.RunConfig) -> None:
    run_name = cfg.model_name
    wandb.init(project="predict360user", name=run_name)
    wandb.run.log({"model": cfg.model_name, "batch": cfg.batch_size, "lr": cfg.lr})
    log.info(f"run {run_name} config is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")
    
    # seed
    cfg.set_random_seed()
    
    # ingestion
    df_wins = p3u.load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = p3u.split(
        df_wins,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        seed=cfg.seed
    )
    _, n_low, n_medium, n_high = p3u.count_entropy(
        df_wins[df_wins["partition"] == "train"]
    )
    wandb.run.log({"trn_low": n_low, "trn_med": n_medium, "trn_hig": n_high})

    # fit model
    model = p3u.build_model(cfg)
    model.fit(df_wins)

    # evaluate model
    model.evaluate(df_wins)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = p3u.RunConfig(**OmegaConf.from_cli())
    run(cfg)
