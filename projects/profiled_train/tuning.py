from omegaconf import OmegaConf
import logging
from dataclasses import dataclass
import math
import wandb

from predict360user.train import build_model, fit_keras, evaluate
from predict360user.ingest import count_entropy, load_df_wins, split
from predict360user.model_config import ModelConf, ENTROPY_NAMES, build_run_name

log = logging.getLogger()


@dataclass
class RunConf(ModelConf):
    tuning_entropy: str = ""


def run(cfg: RunConf) -> None:
    build_run_name(cfg)
    assert cfg.tuning_entropy in ENTROPY_NAMES
    cfg.run_name += f",tuni={cfg.tuning_entropy}"
    log.info(f"run conf is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")

    # -- load dataset --
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
        name=cfg.run_name,
        resume=True,
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = OmegaConf.merge(OmegaConf.structured(RunConf), OmegaConf.from_cli())
    run(cfg)
