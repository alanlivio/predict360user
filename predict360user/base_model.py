from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generator, Sequence

try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = object

from predict360user.run_config import RunConfig

log = logging.getLogger()

ENTROPY_NAMES = ("low", "medium", "high")


class BaseModel(BaseEstimator, ABC):
    """Base class for models.

    Keyword arguments:
    cfg  ==> runConfig
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def predict(self, df) -> Sequence:
        """model predict

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame from load_df_wins() and split()

        Returns
        -------
        BaseModel
            model following sklearn.BaseModel
        """
        ...

    def fit(self, df) -> BaseModel:
        """model fit

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame from load_df_wins() and split()

        Returns
        -------
        BaseModel
            model following sklearn.BaseModel
        """
        return self

    def evaluate(self, df, target_class=None) -> dict:
        """evaluate model

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame from load_df_wins() and split()

        Returns
        -------
        dict
            prediction error per h_window t
        """
        import numpy as np
        import pandas as pd
        from predict360user.utils.math360 import orth_dist_cartesian

        log.info("evaluate ...")
        assert "partition" in df.columns
        test_idx = df[df["partition"] == "test"].index
        assert len(test_idx)

        pred = self.predict(df.loc[test_idx])
        assert len(test_idx) == len(pred)

        def _calc_pred_err(row) -> list[float]:
            pred_row = row["pred"]
            pred_true = row["h_window"]
            error_per_t = [orth_dist_cartesian(pred_row[t], pred_true[t]) for t in t_range]
            return error_per_t

        df = df.loc[test_idx].assign(pred=pred)
        t_range = list(range(self.cfg.h_window))
        df.loc[test_idx, t_range] = df.loc[test_idx].apply(
            _calc_pred_err, axis=1, result_type="expand"
        )
        assert df.loc[test_idx, t_range].notna().all().all()

        test_wins = df.loc[test_idx]
        classes = [
            ("all", test_idx),
            ("low", test_wins.index[test_wins["actS_c"] == "low"]),
            ("medium", test_wins.index[test_wins["actS_c"] == "medium"]),
            ("high", test_wins.index[test_wins["actS_c"] == "high"]),
        ]
        if target_class:
            assert target_class in ENTROPY_NAMES
            classes = [c for c in classes if c[0] == target_class]
            assert classes
        err_dict_per_class = {}
        for cls, idx in classes:
            if len(idx) > 0:
                mean_val = float(df.loc[idx, t_range].values.mean())
                class_err_per_t = [float(v) for v in df.loc[idx, t_range].mean()]
            else:
                mean_val = 0.0
                class_err_per_t = [0.0] * len(t_range)
            err_dict_per_class[cls] = {
                "mean": mean_val,
                "per_t": class_err_per_t,
            }
        return err_dict_per_class


def batch_generator_fn(batch_size: int, df, fn: Callable) -> Generator:
    while True:
        for start in range(0, len(df), batch_size):
            end = start + batch_size if start + batch_size <= len(df) else len(df)
            yield fn(df[start:end])
