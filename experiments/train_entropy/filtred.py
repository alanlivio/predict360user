from omegaconf import OmegaConf
import logging
import numpy as np
from os.path import join
from dataclasses import dataclass
from predict360user.model_wrapper import ModelWrapper, ModelConf, build_run_name
from predict360user.train import build_model
import wandb
from predict360user.ingest import (
    count_entropy,
    load_df_wins,
    split_train_filtred,
    get_class_name,
    calc_actual_entropy,
    get_class_thresholds,
)
from predict360user.model_wrapper import (
    ModelConf,
    build_run_name,
)

ENTROPY_NAMES = ["all", "low", "medium", "high", "nohigh", "nolow"]
ENTROPY_NAMES_AUTO = ["auto", "auto_m_window", "auto_since_start"]

log = logging.getLogger()


def _set_predict_by_entropy(model: ModelWrapper, cfg: ModelConf, df_wins) -> ModelWrapper:
    prefix = join(cfg.savedir, f"{cfg.model_name},{cfg.dataset_name},actS,")
    threshold_medium, threshold_high = get_class_thresholds(df_wins, "actS")
    model_low = model.copy()
    model_low.load_weights(join(prefix + "low", "weights.hdf5"))
    model_medium = model.copy()
    model_medium.load_weights(join(prefix + "medium", "weights.hdf5"))
    model_high = model.copy()
    model_high.load_weights(join(prefix + "high", "weights.hdf5"))

    def _predict_by_entropy(self, traces: np.array, x_i: int) -> ModelWrapper:
        if cfg.train_entropy == "auto":
            window = traces
        elif cfg.train_entropy == "auto_m_window":
            window = traces[x_i - cfg.m_window : x_i]
        elif cfg.train_entropy == "auto_since_start":
            window = traces[0:x_i]
        else:
            raise RuntimeError()
        a_ent = calc_actual_entropy(window)
        actS_c = get_class_name(a_ent, threshold_medium, threshold_high)
        if actS_c == "low":
            return model_low.predict_for_sample(self, traces, x_i)
        if actS_c == "medium":
            return model_medium.predict_for_sample(self, traces, x_i)
        if actS_c == "high":
            return model_high.predict_for_sample(self, traces, x_i)
        else:
            raise RuntimeError()

    model.predict_for_sample = (_predict_by_entropy, model)
    return model


@dataclass
class RunConf(ModelConf):
    train_entropy: str = "all"
    minsize: bool = False


def run(cfg: RunConf) -> None:
    assert cfg.train_entropy in ENTROPY_NAMES + ENTROPY_NAMES_AUTO
    build_run_name(cfg)
    if cfg.train_entropy != "all":
        cfg.run_name += f",filt={cfg.train_entropy}"
    if cfg.minsize:
        cfg.run_name += f",mins={cfg.minsize!r}"
    log.info(f"run conf is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")

    # load dataset
    df_wins = load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = split_train_filtred(
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
        name=cfg.run_name,
    )

    # fit model
    model = build_model(cfg)
    if cfg.train_entropy.startswith("auto"):  # will not use model
        _set_predict_by_entropy(model)
    model.fit(df_wins)

    # evaluate and log to wandb
    model.evaluate(df_wins)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = OmegaConf.merge(OmegaConf.structured(RunConf), OmegaConf.from_cli())
    run(cfg)
