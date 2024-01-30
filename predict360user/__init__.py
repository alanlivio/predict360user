from predict360user.data_ingestion import *
from predict360user.data_exploration import *
from predict360user.base_model import *
from predict360user.models import *

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