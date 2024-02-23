from predict360user.base_model import BaseModel
from predict360user.data_exploration import (
    show_entropy_histogram,
    show_entropy_histogram_per_partition,
    show_traject,
    show_trajects_representative,
)
from predict360user.data_ingestion import (
    ENTROPY_NAMES,
    count_entropy,
    count_entropy_str,
    load_df_trajecs,
    load_df_wins,
    split,
    split_train_filtred,
)
from predict360user.models import Interpolation, NoMotion, PosOnly, PosOnly3D
from predict360user.run_config import RunConfig, set_random_seed


def get_model(cfg: RunConfig) -> BaseModel:
    if cfg.model == "pos_only":
        return PosOnly(cfg)
    elif cfg.model == "pos_only_3d":
        return PosOnly3D(cfg)
    elif cfg.model == "interpolation":
        return Interpolation(cfg)
    elif cfg.model == "no_motion":
        return NoMotion(cfg)
    else:
        raise NotImplementedError
