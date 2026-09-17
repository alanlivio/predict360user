from predict360user.pipelines.base_train_eval_flow import BaseTrainEvalFlow
from predict360user.pipelines.benchmark_flow import BenchmarkFlow
from predict360user.pipelines.cvpr18_flow import CVPR18Flow
from predict360user.pipelines.gnn_flow import GNNFlow
from predict360user.pipelines.interpolation_flow import InterpolationFlow
from predict360user.pipelines.mm18_flow import MM18Flow
from predict360user.pipelines.no_motion_flow import NoMotionFlow
from predict360user.pipelines.pointnet_flow import PointNetFlow
from predict360user.pipelines.pos_only_3d_flow import PosOnly3DFlow
from predict360user.pipelines.pos_only_flow import PosOnlyFlow
from predict360user.pipelines.sal_only_flow import SalOnlyFlow
from predict360user.pipelines.track_flow import TRACKFlow
from predict360user.pipelines.train_split_flow import TrainSplitFlow

ExperimentFlow = NoMotionFlow

__all__ = [
    "BaseTrainEvalFlow",
    "BenchmarkFlow",
    "TrainSplitFlow",
    "NoMotionFlow",
    "InterpolationFlow",
    "PosOnlyFlow",
    "PosOnly3DFlow",
    "CVPR18Flow",
    "TRACKFlow",
    "GNNFlow",
    "MM18Flow",
    "PointNetFlow",
    "SalOnlyFlow",
    "ExperimentFlow",
]
