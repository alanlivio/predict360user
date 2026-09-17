import logging
import sys
from pathlib import Path

from metaflow import FlowSpec, Parameter, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import predict360user as p3u

log = logging.getLogger("train_split_flow")


class TrainSplitFlow(FlowSpec):
    dataset = Parameter("dataset", help="Dataset name", default="david")
    init_window = Parameter("init_window", help="Init buffer window size", default=30)
    h_window = Parameter("h_window", help="Prediction horizon window size", default=25)
    m_window = Parameter("m_window", help="Memory window size", default=5)
    train_size = Parameter("train_size", help="Train partition ratio", default=0.8)
    test_size = Parameter("test_size", help="Test partition ratio", default=0.2)
    seed = Parameter("seed", help="Random seed for splitting", default=42)

    @step
    def start(self):
        p3u.set_random_seed(self.seed)
        self.next(self.load_data)

    @step
    def load_data(self):
        self.df = p3u.load_df_wins(
            dataset=self.dataset,
            init_window=self.init_window,
            h_window=self.h_window,
            m_window=self.m_window,
        )
        self.next(self.split_data)

    @step
    def split_data(self):
        self.df = p3u.split(
            self.df,
            train_size=self.train_size,
            test_size=self.test_size,
            seed=self.seed,
        )
        self.train_count = int(len(self.df[self.df["partition"] == "train"]))
        self.val_count = int(len(self.df[self.df["partition"] == "val"]))
        self.test_count = int(len(self.df[self.df["partition"] == "test"]))
        self.entropy_distribution = p3u.count_entropy(self.df[self.df["partition"] == "train"])
        self.next(self.end)

    @step
    def end(self):
        log.info(
            f"TrainSplitFlow completed. Train: {self.train_count}, Val: {self.val_count}, Test: {self.test_count}"
        )


if __name__ == "__main__":
    TrainSplitFlow()
