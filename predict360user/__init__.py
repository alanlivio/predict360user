from predict360user.data_ingestion import (
    ENTROPY_NAMES,
    load_df_trajecs,
    load_df_wins,
    split,
    split_train_filtred,
    count_entropy_str,
    count_entropy,
)
from predict360user.data_exploration import (
    show_entropy_histogram,
    show_traject,
    show_trajects_representative,
    show_entropy_histogram_per_partition,
)
from predict360user.run_config import RunConfig, set_random_seed
from predict360user.base_model import BaseModel, KerasModel
from predict360user.models import PosOnly, PosOnly3D, Interpolation, NoMotion


def build_model(cfg: RunConfig) -> BaseModel:
    if cfg.model_name == "pos_only":
        return PosOnly(cfg)
    elif cfg.model_name == "pos_only_3d":
        return PosOnly3D(cfg)
    elif cfg.model_name == "interpolation":
        return Interpolation(cfg)
    elif cfg.model_name == "no_motion":
        return NoMotion(cfg)
    else:
        raise NotImplementedError
