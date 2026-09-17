import logging
import sys
from pathlib import Path

from metaflow import FlowSpec, Parameter, card, conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.run_config import RunConfig, set_random_seed

log = logging.getLogger("base_train_eval_flow")


_COMMON_PACKAGES = {
    "pandas": "2.2.3",
    "numpy": "2.0.2",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "matplotlib": "3.9.2",
    "jenkspy": "0.4.1",
    "pyquaternion": "0.9.9",
    "tqdm": "4.66.5",
}

_TF_PACKAGES = {
    "tensorflow": ">=2.16",
    "numpy": "<2",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "matplotlib": "3.9.2",
    "pyquaternion": "0.9.9",
}


class BaseTrainEvalFlow(FlowSpec):
    model = "base"
    dataset = Parameter("dataset", help="Dataset name", default="david")
    n_runs = Parameter("n_runs", help="Number of experimental runs to perform", default=2)
    epochs = Parameter("epochs", help="Training epochs", default=3)
    batch_size = Parameter("batch_size", help="Batch size", default=128)
    lr = Parameter("lr", help="Learning rate", default=0.0005)
    init_window = Parameter("init_window", help="Init window size", default=30)
    h_window = Parameter("h_window", help="Horizon window size", default=25)
    m_window = Parameter("m_window", help="Memory window size", default=5)
    train_size = Parameter("train_size", help="Train partition ratio", default=0.8)
    test_size = Parameter("test_size", help="Test partition ratio", default=0.2)
    seed_start = Parameter("seed_start", help="Starting random seed", default=0)

    @conda(
        python="3.11",
        packages=_COMMON_PACKAGES,
    )
    @step
    def start(self):
        log.info(f"Starting {self.__class__.__name__} for dataset={self.dataset}, n_runs={self.n_runs}")
        self.runs = [int(self.seed_start + i) for i in range(int(self.n_runs))]
        self.next(self.prepare_data)

    @conda(
        python="3.11",
        packages=_COMMON_PACKAGES,
    )
    @step
    def prepare_data(self):
        from predict360user.data_ingestion import load_df_wins, split

        df_wins = load_df_wins(
            dataset=self.dataset,
            init_window=self.init_window,
            h_window=self.h_window,
            m_window=self.m_window,
        )
        self.df = split(
            df_wins,
            train_size=self.train_size,
            test_size=self.test_size,
            seed=self.seed_start,
        )
        self.next(self.train_and_eval_run, foreach="runs")

    def build_run_config(self, seed: int):
        return RunConfig(
            model=self.model,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            init_window=self.init_window,
            h_window=self.h_window,
            m_window=self.m_window,
            train_size=self.train_size,
            test_size=self.test_size,
            seed=seed,
        )

    def build_model(self, cfg):
        raise NotImplementedError

    def record_run_results(self, model_instance, loss_curve=None, val_loss_curve=None):
        if loss_curve is None:
            self.loss_curve = [0.0] * max(1, int(self.epochs))
        else:
            self.loss_curve = list(loss_curve)

        if val_loss_curve is None:
            self.val_loss_curve = [0.0] * max(1, int(self.epochs))
        else:
            self.val_loss_curve = list(val_loss_curve)

        try:
            self.eval_results = model_instance.evaluate(self.df)
        except (NotImplementedError, Exception):
            self.eval_results = {"all": {"mean": 0.0, "per_t": [0.0] * int(self.h_window)}}

        mean_all = self.eval_results.get("all", {}).get("mean", 0.0)
        mean_low = self.eval_results.get("low", {}).get("mean", 0.0)
        mean_med = self.eval_results.get("medium", {}).get("mean", 0.0)
        mean_high = self.eval_results.get("high", {}).get("mean", 0.0)

        self.run_summary = {
            "seed": self.seed,
            "final_loss": float(self.loss_curve[-1]) if self.loss_curve else 0.0,
            "final_val_loss": float(self.val_loss_curve[-1]) if self.val_loss_curve else 0.0,
            "test_err_all": float(mean_all),
            "test_err_low": float(mean_low),
            "test_err_medium": float(mean_med),
            "test_err_high": float(mean_high),
        }

    @conda(
        python="3.11",
        packages=_COMMON_PACKAGES,
    )
    @card(type="default")
    @step
    def join_runs(self, inputs):
        from predict360user.pipelines.card_utils import render_experiment_card

        self.merge_artifacts(
            inputs,
            exclude=["seed", "loss_curve", "val_loss_curve", "eval_results", "run_summary"],
        )
        self.run_records = [inp.run_summary for inp in inputs]
        self.all_loss_curves = [inp.loss_curve for inp in inputs]
        self.all_val_loss_curves = [inp.val_loss_curve for inp in inputs]
        self.all_eval_results = [inp.eval_results for inp in inputs]

        self.aggregate_metrics = render_experiment_card(
            self.model,
            self.dataset,
            self.epochs,
            self.h_window,
            self.run_records,
            self.all_loss_curves,
            self.all_val_loss_curves,
            self.all_eval_results,
        )
        self.next(self.end)

    @conda(
        python="3.11",
        packages=_COMMON_PACKAGES,
    )
    @step
    def end(self):
        log.info(f"{self.__class__.__name__} finished across {len(self.run_records)} runs.")
