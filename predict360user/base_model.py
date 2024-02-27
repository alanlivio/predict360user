from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generator, Sequence

import pandas as pd
from sklearn.base import BaseEstimator

import wandb
from predict360user.run_config import RunConfig
from predict360user.utils.math360 import orth_dist_cartesian

log = logging.getLogger()

class BaseModel(BaseEstimator, ABC):
    """Base class for models.

    Keyword arguments:
    cfg  -- RunConfig
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Sequence:
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

    def fit(self, df: pd.DataFrame) -> BaseModel:
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

    def evaluate(self, df: pd.DataFrame) -> dict:
        """evalate model

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame from load_df_wins() and split()

        Returns
        -------
        dict
            prediction error per h_window t
        """
        log.info("evaluate ...")
        assert "partition" in df.columns
        test_idx = df[df["partition"] == "test"].index
        assert len(test_idx)
        
        # predict
        pred = self.predict(df.loc[test_idx])
        assert len(test_idx) == len(pred)

        # calculate predict errors per t
        def _calc_pred_err(row) -> list[float]:
            # return np.random.rand(self.cfg.h_window)  # for debugging
            pred = row["pred"]
            pred_true = row["h_window"]
            error_per_t = [orth_dist_cartesian(pred[t], pred_true[t]) for t in t_range]
            return error_per_t

        df = df.loc[test_idx].assign(pred=pred)
        t_range = list(range(self.cfg.h_window))
        df.loc[test_idx, t_range] = df.loc[test_idx].apply(
            _calc_pred_err, axis=1, result_type="expand"
        )  # type: ignore
        assert df.loc[test_idx, t_range].all().all()

        # calculate predict errors pert t mean
        test_wins = df.loc[test_idx]
        classes = [
            ("all", test_idx),
            ("low", test_wins.index[test_wins["actS_c"] == "low"]),
            ("medium", test_wins.index[test_wins["actS_c"] == "medium"]),
            ("high", test_wins.index[test_wins["actS_c"] == "high"]),
        ]
        err_per_class_dict = {tup[0]: {} for tup in classes}
        for actS_c, idx in classes:
            # 1) mean per class (as wandb summary): # err_all, err_low, err_high, err_medium,
            err_per_class_dict[actS_c]["mean"] = df.loc[idx, t_range].values.mean()
            # 2) mean err per t per class
            class_err_per_t = df.loc[idx, t_range].mean()
            data = [[x, y] for (x, y) in zip(t_range, class_err_per_t)]
            err_per_class_dict[actS_c]["mean_per_t"] = data
        if wandb.run:
            for actS_c, err in err_per_class_dict.items():
                wandb.run.summary[f"err_{actS_c}"] = err["mean"]
                table = wandb.Table(data=err["mean_per_t"], columns=["t", "err"])
                plot_id = f"test_err_per_t_class_{actS_c}"
                plot = wandb.plot.line(table, "t", "err", title=plot_id)
                wandb.log({plot_id: plot})
        return err_per_class_dict


def batch_generator_fn(
    batch_size: int, df: pd.DataFrame, fn: Callable
) -> Generator:
    while True:
        for start in range(0, len(df), batch_size):
            end = (
                start + batch_size
                if start + batch_size <= len(df)
                else len(df)
            )
            yield fn(df[start:end])
