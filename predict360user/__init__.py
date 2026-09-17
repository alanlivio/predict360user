from predict360user.base_model import BaseModel
try:
    from predict360user.pipelines import (
        BaseTrainEvalFlow,
        BenchmarkFlow,
        CVPR18Flow,
        ExperimentFlow,
        GNNFlow,
        InterpolationFlow,
        MM18Flow,
        NoMotionFlow,
        PointNetFlow,
        PosOnly3DFlow,
        PosOnlyFlow,
        SalOnlyFlow,
        TRACKFlow,
        TrainSplitFlow,
    )
except ImportError:
    pass
from predict360user.run_config import RunConfig, set_random_seed

try:
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
except ImportError:
    pass

_FLOW_MAP = {
    "no_motion": NoMotionFlow,
    "interpolation": InterpolationFlow,
    "pos_only": PosOnlyFlow,
    "pos_only_3d": PosOnly3DFlow,
    "cvpr18": CVPR18Flow,
    "track": TRACKFlow,
    "gnn": GNNFlow,
    "mm18": MM18Flow,
    "pointnet": PointNetFlow,
    "sal_only": SalOnlyFlow,
}


def get_model(cfg: RunConfig) -> BaseModel:
    flow_cls = _FLOW_MAP.get(cfg.model)
    if flow_cls is None:
        raise NotImplementedError(f"Model '{cfg.model}' not implemented in flows")
    flow = flow_cls(use_cli=False)
    return flow.build_model(cfg)
