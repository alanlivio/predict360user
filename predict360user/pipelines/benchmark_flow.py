import logging
import os
import subprocess
import sys
from pathlib import Path

from metaflow import Flow, FlowSpec, Parameter, card, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger("benchmark_flow")

_FLOW_MODULE_MAP = {
    "no_motion": ("no_motion_flow.py", "NoMotionFlow"),
    "interpolation": ("interpolation_flow.py", "InterpolationFlow"),
    "pos_only": ("pos_only_flow.py", "PosOnlyFlow"),
    "pos_only_3d": ("pos_only_3d_flow.py", "PosOnly3DFlow"),
    "cvpr18": ("cvpr18_flow.py", "CVPR18Flow"),
    "track": ("track_flow.py", "TRACKFlow"),
    "gnn": ("gnn_flow.py", "GNNFlow"),
    "mm18": ("mm18_flow.py", "MM18Flow"),
    "pointnet": ("pointnet_flow.py", "PointNetFlow"),
    "sal_only": ("sal_only_flow.py", "SalOnlyFlow"),
}


class BenchmarkFlow(FlowSpec):
    models = Parameter("models", help="Comma-separated list of models to benchmark", default="no_motion,interpolation")
    dataset = Parameter("dataset", help="Dataset name", default="david")
    n_runs = Parameter("n_runs", help="Number of experimental runs to perform", default=2)
    epochs = Parameter("epochs", help="Training epochs", default=2)
    batch_size = Parameter("batch_size", help="Batch size", default=128)
    lr = Parameter("lr", help="Learning rate", default=0.0005)
    init_window = Parameter("init_window", help="Init window size", default=30)
    h_window = Parameter("h_window", help="Horizon window size", default=25)
    m_window = Parameter("m_window", help="Memory window size", default=5)
    train_size = Parameter("train_size", help="Train partition ratio", default=0.8)
    test_size = Parameter("test_size", help="Test partition ratio", default=0.2)
    seed_start = Parameter("seed_start", help="Starting random seed", default=0)
    output_dir = Parameter("output_dir", help="Output directory for comparison figures", default="saved/figures")
    flow_env = Parameter("flow_env", help="Execution environment for model flows (conda/local)", default="conda")

    @step
    def start(self):
        log.info(f"Starting BenchmarkFlow on dataset={self.dataset}")
        raw_models = self.models.strip().lower()
        if raw_models == "all":
            self.parsed_models = list(_FLOW_MODULE_MAP.keys())
        elif raw_models == "baselines":
            self.parsed_models = ["no_motion", "interpolation", "pos_only", "pos_only_3d"]
        else:
            self.parsed_models = [m.strip().lower() for m in self.models.split(",") if m.strip()]

        for m in self.parsed_models:
            if m not in _FLOW_MODULE_MAP:
                raise ValueError(f"Unknown model '{m}'. Supported models: {list(_FLOW_MODULE_MAP.keys())}")

        os.makedirs(self.output_dir, exist_ok=True)
        self.next(self.run_model, foreach="parsed_models")

    @step
    def run_model(self):
        self.model_name = self.input
        script_name, flow_class = _FLOW_MODULE_MAP[self.model_name]
        script_path = os.path.join(_REPO_ROOT, "predict360user", "pipelines", script_name)

        cmd = [
            sys.executable,
            script_path,
            f"--environment={self.flow_env}",
            "run",
            f"--dataset={self.dataset}",
            f"--n_runs={self.n_runs}",
            f"--epochs={self.epochs}",
            f"--batch_size={self.batch_size}",
            f"--lr={self.lr}",
            f"--init_window={self.init_window}",
            f"--h_window={self.h_window}",
            f"--m_window={self.m_window}",
            f"--train_size={self.train_size}",
            f"--test_size={self.test_size}",
            f"--seed_start={self.seed_start}",
        ]
        log.info(f"Executing model flow: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
        subprocess.check_call(cmd, env=env, cwd=_REPO_ROOT)

        latest_run = Flow(flow_class).latest_successful_run
        if latest_run is None:
            raise RuntimeError(f"Could not find successful run for flow '{flow_class}'")

        join_step = latest_run["join_runs"]
        task_data = join_step.task.data

        self.model_result = {
            "model": self.model_name,
            "flow_name": flow_class,
            "run_id": latest_run.id,
            "aggregate_metrics": getattr(task_data, "aggregate_metrics", {}),
            "run_records": getattr(task_data, "run_records", []),
            "all_loss_curves": getattr(task_data, "all_loss_curves", []),
            "all_val_loss_curves": getattr(task_data, "all_val_loss_curves", []),
            "all_eval_results": getattr(task_data, "all_eval_results", []),
        }
        self.next(self.compare_models)

    @card(type="default")
    @step
    def compare_models(self, inputs):
        from predict360user.pipelines.card_utils import render_benchmark_card

        self.merge_artifacts(inputs, exclude=["model_name", "model_result"])
        self.benchmark_records = [inp.model_result for inp in inputs]

        self.comparison_summary = render_benchmark_card(
            dataset=self.dataset,
            benchmark_records=self.benchmark_records,
            output_dir=self.output_dir,
        )
        self.next(self.end)

    @step
    def end(self):
        log.info("BenchmarkFlow successfully finished across all models.")
        for r in self.benchmark_records:
            log.info(f"Model {r['model']}: {r.get('aggregate_metrics', {})}")


if __name__ == "__main__":
    BenchmarkFlow()
