from omegaconf import OmegaConf
import logging
from os.path import basename
import wandb
from sklearn.utils import shuffle

from predict360user.model_config import Config, ENTROPY_NAMES
from predict360user.train import build_model, fit_keras, evaluate
from predict360user.ingest import count_entropy, load_df_wins, split

log = logging.getLogger(basename(__file__))
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def main(cfg: Config) -> None:
    assert cfg.tuning_entropy in ENTROPY_NAMES
    log.info("used config:\n---\n" + OmegaConf.to_yaml(cfg) + "----")
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

    # split tuning
    tuning_prc = 0.25
    epochs_pretuning = int(cfg.epochs * (1 - tuning_prc))
    epochs_tuning = int(cfg.epochs * tuning_prc)
    train_wins_tuning = shuffle(
        train_wins[train_wins["actS_c"] == cfg.tuning_entropy].sample(
            int(train_wins.size * tuning_prc)
        )
    )
    train_wins_pretuning = shuffle(train_wins.loc[~train_wins_tuning.index])
    # fit all
    cfg.epochs = epochs_pretuning
    fit_keras(cfg, model, train_wins_pretuning)
    # fit tuning
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
    cfg = Config(**OmegaConf.from_cli())
    cfg.tuning_entropy = "low"
    main(cfg)
