from predict360user.ingest import *
from predict360user.explore import *
from predict360user.estimator import *
from predict360user.models import *
from . import utils

def build_model(cfg: Config) -> Estimator:
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