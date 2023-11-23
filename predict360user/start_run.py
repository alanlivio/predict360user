from omegaconf import OmegaConf
import logging
from predict360user.model_config import ModelConf, build_run_name
from predict360user.train import build_model, fit_keras, evaluate
import wandb
from predict360user.ingest import count_entropy, load_df_wins, split

log = logging.getLogger()


def run(cfg: ModelConf) -> None:
    build_run_name(cfg)
    log.info(f"config for {cfg.run_name}\n--\n" + OmegaConf.to_yaml(cfg) + "--")

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
    wandb.init(
        project="predict360user",
        config={
            "model_name": cfg.model_name,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "train_n_low": n_low,
            "train_n_medium": n_medium,
            "train_n_high": n_high,
        },
        name=cfg.run_name,
        resume=True,
    )

    # fit model
    model = build_model(cfg)
    fit_keras(cfg, model, df_wins)

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
    cfg = OmegaConf.merge(OmegaConf.structured(ModelConf), OmegaConf.from_cli())
    run(cfg)
