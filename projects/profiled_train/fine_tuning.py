from omegaconf import OmegaConf
import logging
from os.path import basename
import wandb
import math

from predict360user.model_config import Config, ENTROPY_NAMES
from predict360user.train import build_model, fit_keras, evaluate
from predict360user.ingest import count_entropy, load_df_wins, split

log = logging.getLogger(basename(__file__))


def main(cfg: Config) -> None:
    assert cfg.tuning_entropy in ENTROPY_NAMES
    log.info("config:\n--\n" + OmegaConf.to_yaml(cfg) + "--")
    log.info(f"model_dir={cfg.model_dir}")

    # -- load dataset --
    df_wins = load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = split(
        df_wins,
        train_size=cfg.train_size,
        train_entropy=cfg.train_entropy,
        train_minsize=cfg.minsize,
        test_size=cfg.test_size,
    )
    _, n_low, n_medium, n_high = count_entropy(df_wins[df_wins["partition"] == "train"])
    wandb.init(
        project="predict360user",
        config={
            "model_name": cfg.model_name,
            "train_entropy": cfg.train_entropy,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "train_n_low": n_low,
            "train_n_medium": n_medium,
            "train_n_high": n_high,
        },
        name=cfg.model_fullname,
    )

    # -- fit --
    model = build_model(cfg)
    train_wins = df_wins[df_wins["partition"] == "train"]

    # split for tuning
    tuning_prc = 0.33
    train_wins_tuning = train_wins[train_wins["actS_c"] == cfg.tuning_entropy].sample(
        frac=tuning_prc, random_state=1
    )
    train_wins_pretuning = train_wins.drop(train_wins_tuning.index)
    epochs_pretuning = math.floor(cfg.epochs * (1 - tuning_prc))
    epochs_tuning = math.ceil(cfg.epochs * tuning_prc)
    # fit
    cfg.epochs = epochs_pretuning
    fit_keras(cfg, model, train_wins_pretuning)
    cfg.epochs = epochs_tuning
    fit_keras(cfg, model, train_wins_tuning)

    # evaluate and log to wandb
    err_per_class_dict = evaluate(cfg, model, df_wins)
    for actS_c, err in err_per_class_dict.items():
        wandb.run.summary[f"err_{actS_c}"] = err["mean"]
        table = wandb.Table(data=err["mean_per_t"], columns=["t", "err"])
        plot_id = f"test_err_per_t_class_{actS_c}"
        plot = wandb.plot.line(table, "t", "err", title=plot_id)
        wandb.log({plot_id: plot})
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = Config(**OmegaConf.from_cli())
    main(cfg)
