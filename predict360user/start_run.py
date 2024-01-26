from omegaconf import OmegaConf
import logging
from predict360user.model_wrapper import Config
from predict360user.train import build_model
import wandb
from predict360user.ingest import count_entropy, load_df_wins, split

log = logging.getLogger()


def run(cfg: Config) -> None:
    run_name = cfg.model_name
    log.info(f"run {run_name} config is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")
    wandb.init(project="predict360user", name=cfg.model_name)
    wandb.run.log({"model": cfg.model_name, "batch": cfg.batch_size, "lr": cfg.lr})
    # load dataset
    df_wins = load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = split(
        df_wins,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
    )
    _, n_low, n_medium, n_high = count_entropy(df_wins[df_wins["partition"] == "train"])
    wandb.run.log({"trn_low": n_low, "trn_med": n_medium, "trn_hig": n_high})
    # fit model
    model = build_model(cfg)
    model.fit(df_wins)

    # evaluate model
    model.evaluate(df_wins)
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = Config(**OmegaConf.from_cli())
    run(cfg)
