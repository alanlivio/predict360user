from omegaconf import OmegaConf
import logging
import numpy as np
from os.path import join
from dataclasses import dataclass
import wandb
import predict360user as p3u

ENTROPY_NAMES = ["all", "low", "medium", "high", "nohigh", "nolow"]
ENTROPY_NAMES_AUTO = ["auto", "auto_m_window", "auto_since_start"]

log = logging.getLogger()


def _set_predict_by_entropy(
    model: p3u.ModelWrapper, cfg: p3u.Config, df_wins
) -> p3u.ModelWrapper:
    prefix = join(cfg.savedir, f"{cfg.model_name},{cfg.dataset_name},actS,")
    model.threshold_medium, model.threshold_high = p3u.get_class_thresholds(
        df_wins, "actS"
    )
    model.model_low = model.copy()
    model.model_low.load_weights(join(prefix + "low", "weights.hdf5"))
    model.model_medium = model.copy()
    model.model_medium.load_weights(join(prefix + "medium", "weights.hdf5"))
    model.model_high = model.copy()
    model.model_high.load_weights(join(prefix + "high", "weights.hdf5"))

    def _predict_by_entropy(self, traces: np.array, x_i: int) -> p3u.ModelWrapper:
        if cfg.train_entropy == "auto":
            window = traces
        elif cfg.train_entropy == "auto_m_window":
            window = traces[x_i - cfg.m_window : x_i]
        elif cfg.train_entropy == "auto_since_start":
            window = traces[0:x_i]
        else:
            raise RuntimeError()
        a_ent = p3u.calc_actual_entropy(window)
        actS_c = p3u.get_class_name(a_ent, self.threshold_medium, self.threshold_high)
        if actS_c == "low":
            return self.model_low.predict_for_sample(self, traces, x_i)
        if actS_c == "medium":
            return self.model_medium.predict_for_sample(self, traces, x_i)
        if actS_c == "high":
            return self.model_high.predict_for_sample(self, traces, x_i)
        else:
            raise RuntimeError()

    model.predict_for_sample = _predict_by_entropy
    return model


@dataclass
class RunConfig(p3u.Config):
    train_entropy: str = "all"
    minsize: bool = False


def run(cfg: RunConfig) -> None:
    run_name = f"{cfg.model_name},filt={cfg.train_entropy}"
    if cfg.minsize:
        run_name += f",mins={cfg.minsize!r}"
    log.info(f"run {run_name} config is: \n--\n" + OmegaConf.to_yaml(cfg) + "--")
    assert cfg.train_entropy in ENTROPY_NAMES + ENTROPY_NAMES_AUTO
    wandb.init(project="predict360user", name=run_name)
    wandb.run.log({"model": cfg.model_name, "batch": cfg.batch_size, "lr": cfg.lr})

    # load dataset
    df_wins = p3u.load_df_wins(
        dataset_name=cfg.dataset_name,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
    )
    df_wins = p3u.split_train_filtred(
        df_wins,
        train_size=cfg.train_size,
        train_entropy=cfg.train_entropy,
        train_minsize=cfg.minsize,
        test_size=cfg.test_size,
    )
    _, n_low, n_medium, n_high = p3u.count_entropy(
        df_wins[df_wins["partition"] == "train"]
    )
    wandb.run.log({"trn_low": n_low, "trn_med": n_medium, "trn_hig": n_high})

    # fit model
    model = p3u.build_model(cfg)
    if cfg.train_entropy.startswith("auto"):  # will not use model
        _set_predict_by_entropy(model)
    model.fit(df_wins)

    # evaluate and log to wandb
    model.evaluate(df_wins)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = RunConfig(**OmegaConf.from_cli())
    run(cfg)
