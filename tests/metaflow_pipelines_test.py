import unittest

import numpy as np
import pandas as pd

import predict360user as p3u
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
from predict360user.pipelines.base_train_eval_flow import _TF_PACKAGES

ALL_MODEL_FLOWS = (
    NoMotionFlow,
    InterpolationFlow,
    PosOnlyFlow,
    PosOnly3DFlow,
    CVPR18Flow,
    TRACKFlow,
    GNNFlow,
    MM18Flow,
    PointNetFlow,
    SalOnlyFlow,
)


class MetaflowPipelinesTestCase(unittest.TestCase):

    def test_model_flows_structure(self) -> None:
        for flow_cls in ALL_MODEL_FLOWS:
            self.assertTrue(issubclass(flow_cls, BaseTrainEvalFlow))
            flow = flow_cls(use_cli=False)
            self.assertTrue(hasattr(flow, "start"))
            self.assertTrue(hasattr(flow, "prepare_data"))
            self.assertTrue(hasattr(flow, "train_and_eval_run"))
            self.assertTrue(hasattr(flow, "join_runs"))
            self.assertTrue(hasattr(flow, "end"))

    def test_train_split_flow_does_not_have_conda(self) -> None:
        flow = TrainSplitFlow(use_cli=False)
        self.assertTrue(hasattr(flow, "start"))
        self.assertTrue(hasattr(flow, "load_data"))
        self.assertTrue(hasattr(flow, "split_data"))
        self.assertTrue(hasattr(flow, "end"))

        load_decos = getattr(TrainSplitFlow.load_data, "decorators", [])
        split_decos = getattr(TrainSplitFlow.split_data, "decorators", [])
        self.assertEqual([d for d in load_decos if getattr(d, "name", None) == "conda"], [])
        self.assertEqual([d for d in split_decos if getattr(d, "name", None) == "conda"], [])

    def test_conda_decorators_on_model_training_steps(self) -> None:
        for flow_cls in ALL_MODEL_FLOWS:
            decos = getattr(flow_cls.train_and_eval_run, "decorators", [])
            conda_decos = [d for d in decos if getattr(d, "name", None) == "conda"]
            self.assertEqual(
                len(conda_decos),
                1,
                f"{flow_cls.__name__} must have exactly one @conda decorator on train_and_eval_run",
            )
            packages = conda_decos[0].attributes["packages"]
            self.assertIn("scikit-learn", packages)
            self.assertIn("numpy", packages)

        tf_flows = (PosOnlyFlow, PosOnly3DFlow, CVPR18Flow, TRACKFlow, MM18Flow)
        for flow_cls in tf_flows:
            decos = getattr(flow_cls.train_and_eval_run, "decorators", [])
            conda_decos = [d for d in decos if getattr(d, "name", None) == "conda"]
            packages = conda_decos[0].attributes["packages"]
            self.assertIn("tensorflow", packages, f"{flow_cls.__name__} must specify tensorflow")
            self.assertEqual(packages["tensorflow"], _TF_PACKAGES["tensorflow"])

        base_flows = (NoMotionFlow, InterpolationFlow, GNNFlow, PointNetFlow, SalOnlyFlow)
        for flow_cls in base_flows:
            decos = getattr(flow_cls.train_and_eval_run, "decorators", [])
            conda_decos = [d for d in decos if getattr(d, "name", None) == "conda"]
            packages = conda_decos[0].attributes["packages"]
            self.assertNotIn("tensorflow", packages, f"{flow_cls.__name__} should not have tensorflow")

    def test_baseline_models_predict_and_evaluate_without_wandb(self) -> None:
        cfg = p3u.RunConfig(model="no_motion", h_window=5, m_window=3)
        p3u.set_random_seed(123)
        model = p3u.get_model(cfg)

        sample_row = {
            "m_window": np.tile(np.array([1.0, 0.0, 0.0]), (3, 1)),
            "trace": np.array([0.0, 0.0, 1.0]),
            "h_window": np.tile(np.array([0.0, 0.0, 1.0]), (5, 1)),
            "actS_c": "low",
            "partition": "test",
        }
        df = pd.DataFrame([sample_row, sample_row])
        df.loc[0, "actS_c"] = "low"
        df.loc[1, "actS_c"] = "medium"

        model.fit(df)
        preds = model.predict(df)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0].shape, (5, 3))

        eval_res = model.evaluate(df)
        self.assertIn("all", eval_res)
        self.assertIn("mean", eval_res["all"])
        self.assertIn("per_t", eval_res["all"])
        self.assertEqual(len(eval_res["all"]["per_t"]), 5)

    def test_interpolation_model_predict_and_evaluate(self) -> None:
        cfg = p3u.RunConfig(model="interpolation", h_window=4, m_window=3)
        model = p3u.get_model(cfg)

        pt1 = np.array([1.0, 0.0, 0.0])
        pt2 = np.array([0.0, 1.0, 0.0])
        pt3 = np.array([0.0, 0.0, 1.0])
        sample_row = {
            "m_window": np.array([pt1, pt2, pt3]),
            "trace": pt3,
            "h_window": np.array([pt1, pt2, pt3, pt1]),
            "actS_c": "high",
            "partition": "test",
        }
        df = pd.DataFrame([sample_row])
        model.fit(df)
        preds = model.predict(df)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape, (4, 3))

        eval_res = model.evaluate(df)
        self.assertIn("high", eval_res)
        self.assertEqual(len(eval_res["high"]["per_t"]), 4)

    def test_benchmark_flow_structure(self) -> None:
        flow = BenchmarkFlow(use_cli=False)
        self.assertTrue(hasattr(flow, "start"))
        self.assertTrue(hasattr(flow, "run_model"))
        self.assertTrue(hasattr(flow, "compare_models"))
        self.assertTrue(hasattr(flow, "end"))


if __name__ == "__main__":
    unittest.main()
