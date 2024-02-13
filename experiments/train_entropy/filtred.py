import logging
from dataclasses import asdict, dataclass

from omegaconf import OmegaConf as oc

import predict360user as p3u
import wandb

log = logging.getLogger()


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = "all"
    train_minsize: bool = False


def run(cfg: RunConfig) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},filt={cfg.train_entropy}"
    if cfg.train_minsize:
        cfg.name += f",mins={cfg.train_minsize!r}"
    wandb.init(project="predict360user", name=cfg.name, config=asdict(cfg))
    log.info(f"run {cfg.name} config is: \n--\n" + oc.to_yaml(cfg) + "--")

    # seed
    cfg.set_random_seed()
    
    # load dataset
    df_wins = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window
    )
    df_wins = p3u.split_train_filtred(
        df_wins,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        train_entropy=cfg.train_entropy,
        train_minsize=cfg.train_minsize,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = RunConfig(oc.to_container(oc.from_cli()))
    run(cfg)
