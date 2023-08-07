import logging
import multiprocessing
import os
import pickle
from os.path import basename, exists
from typing import Literal

import jenkspy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

from predict360user.tileset import TILESET_DEFAULT, TileSet
from predict360user.utils import (
    DEFAULT_SAVEDIR,
    ENTROPY_CLASS_COLORS,
    HMDDIR,
    calc_actual_entropy,
)

DATASETS = {
    "david": {"size": 1083},
    "fan": {"size": 300},
    "nguyen": {"size": 432},
    "xucvpr": {"size": 6654},
    "xupami": {"size": 4408},
}
log = logging.getLogger(basename(__file__))


def get_class_thresholds(df, col: str) -> tuple[float, float]:
    _, threshold_medium, threshold_high, _ = jenkspy.jenks_breaks(df[col], n_classes=3)
    return threshold_medium, threshold_high


def get_class_name(
    x: float, threshold_medium: float, threshold_high: float
) -> Literal["low", "medium", "high"]:
    return (
        "low" if x < threshold_medium else ("medium" if x < threshold_high else "high")
    )


def filter_df_by_entropy(df: pd.DataFrame, entropy_filter: str) -> pd.DataFrame:
    if entropy_filter == "all":
        return df
    min_size = df["actS_c"].value_counts().min()
    if entropy_filter == "allminsize":  # 3 classes-> n = min_size/3
        filter_df = df
    elif entropy_filter == "nohigh":  # 2 classes-> n = min_size/2
        filter_df = df[df["actS_c"] != "high"]
    elif entropy_filter == "nolow":  # 2 classes-> n = min_size/2
        filter_df = df[df["actS_c"] != "low"]
    else:  # 1 class-> n = min_size
        filter_df = df[df["actS_c"] == entropy_filter]
    nunique = len(filter_df["actS_c"].unique())
    n = int(min_size / nunique)
    return filter_df.groupby("actS_c").apply(lambda x: x.sample(n=n, random_state=1))


def count_entropy_str(df: pd.DataFrame) -> tuple[int, int, int, int]:
    a_len = len(df)
    l_len = len(df[df["actS_c"] == "low"])
    m_len = len(df[df["actS_c"] == "medium"])
    h_len = len(df[df["actS_c"] == "high"])
    return "total of {}: {} low, {} medium, {} high".format(a_len, l_len, m_len, h_len)


class Dataset:
    """:class:`Dataset` stores the original dataset in memory.
    It provides functions for data preprocessing, such user clustering by entropy, and analyses, such as tileset usage.
    Features are stored as :class:`pandas.DataFrame`.
    Attributes:
        df (str): pandas.DataFrame.
    """

    def __init__(self, dataset_name="all", savedir=DEFAULT_SAVEDIR) -> None:
        assert dataset_name in ["all"] + list(DATASETS.keys())
        self.savedir = savedir
        self.dataset_name = dataset_name
        self.pickle_file = os.path.join(savedir, "df_trajects.pickle")

    @property
    def df(self) -> pd.DataFrame:
        if not hasattr(self, "_df"):
            if exists(self.pickle_file):
                with open(self.pickle_file, "rb") as f:
                    log.info(f"loading df from {self.pickle_file}")
                    self._df = pickle.load(f)
            else:
                log.info(f"there is no {self.pickle_file}")
                log.info(f"loading df from {HMDDIR}")
                self._df = self._load_df_trajects_from_hmp()
                log.info(f"calculating entropy")
                self.calc_traces_entropy()
                self.dump()
        return self._df

    def _load_df_trajects_from_hmp(self) -> pd.DataFrame:
        # save cwd and move to head_motion_prediction for invoking funcs
        cwd = os.getcwd()
        os.chdir(HMDDIR)
        from .head_motion_prediction.David_MMSys_18 import Read_Dataset as david
        from .head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
        from .head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
        from .head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
        from .head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami
        DATASETS['david']['pkg'] = david
        DATASETS['fan']['pkg'] = fan
        DATASETS['nguyen']['pkg'] = nguyen
        DATASETS['xucvpr']['pkg'] = xucvpr
        DATASETS['xupami']['pkg'] = xupami
        if self.dataset_name == "all":
            target = DATASETS
        else:
            target = { self.dataset_name: DATASETS[self.dataset_name]}
        n_traces=100
        def _load_dataset_xyz(key, value) -> pd.DataFrame:
            # create_and_store_sampled_dataset()
            # stores csv at head_motion_prediction/<dataset>/sampled_dataset
            if len(os.listdir(value['pkg'].OUTPUT_FOLDER)) < 2:
                value['pkg'].create_and_store_sampled_dataset()
            # load_sample_dateset() process head_motion_prediction/<dataset>/sampled_dataset
            # and return a dict with:
            # {<video1>:{
            #   <user1>:[time-stamp, x, y, z],
            #    ...
            #  },
            #  ...
            # }"
            dataset = value['pkg'].load_sampled_dataset()
            # convert dict to DataFrame
            data = [
                (
                    key,
                    key + "_" + user,
                    key + "_" + video,
                    # np.around(dataset[user][video][:n_traces, 0], decimals=2),
                    dataset[user][video][:n_traces, 1:],
                )
                for user in dataset.keys()
                for video in dataset[user].keys()
            ]
            tmpdf = pd.DataFrame(
                data,
                columns=[
                    "ds",  # e.g., david
                    "user",  # e.g., david_0
                    "video",  # e.g., david_10_Cows
                    # 'times',
                    "traces",  # [[x,y,z], ...]
                ],
            )
            # assert and check
            assert len(tmpdf["ds"]) == value['size']
            return tmpdf

        # create df for each dataset
        df = pd.concat(
            [ _load_dataset_xyz(k,v) for k, v in target.items()],
            ignore_index=True
        ).convert_dtypes()
        assert not df.empty
        # back to cwd
        os.chdir(cwd)
        return df

    def dump(self) -> None:
        log.info(f"saving df to {self.pickle_file}")
        with open(self.pickle_file, "wb") as f:
            pickle.dump(self.df, f)

    def dump_column(self, column) -> None:
        log.info(f"update column {column} to {self.pickle_file}")
        if exists(self.pickle_file):
            with multiprocessing.Lock():
                with open(self.pickle_file, "rb") as f:
                    tmpdf = pickle.load(f)
                tmpdf[column] = self.df[column]
                with open(self.pickle_file, "wb") as f:
                    pickle.dump(tmpdf, f)
        else:
            self.dump()

    def get_trajects(self, video: str, user: str) -> np.array:
        rows = self.df.query(f"user=='{user}' and video=='{video}'")
        assert not rows.empty
        return rows

    def get_traject_random(self) -> pd.Series:
        return self.df.sample(1)

    def get_traces(self, video: str, user: str) -> np.array:
        row = self.df.query(f"user=='{user}' and video=='{video}'")
        assert not row.empty
        return row["traces"].iloc[0]

    def get_trace_random(
        self,
    ) -> np.array:
        traject_ar = self.get_traject_random()["traces"].iloc[0]
        trace = traject_ar[np.random.randint(len(traject_ar - 1))]
        return trace

    def get_video_ids(self) -> np.array:
        return self.df["video"].unique()

    def get_user_ids(self) -> np.array:
        return self.df["user"].unique()

    def get_ds_ids(self) -> np.array:
        return self.df["ds"].unique()

    def calc_traces_entropy(self) -> None:
        self.df.drop(["actS", "actS_c"], axis=1, errors="ignore", inplace=True)
        # calc actS
        self.df["actS"] = self.df["traces"].progress_apply(calc_actual_entropy)
        assert not self.df["actS"].isnull().any()
        # calc trajects_entropy_class
        threshold_medium, threshold_high = get_class_thresholds(self.df, "actS")
        self.df["actS_c"] = self.df["actS"].progress_apply(
            get_class_name, args=(threshold_medium, threshold_high)
        )
        assert not self.df["actS_c"].isnull().any()

    def calc_traces_entropy_hmp(self) -> None:
        self.df.drop(["hmpS", "hmpS_c"], axis=1, errors="ignore", inplace=True)

        # calc hmpS
        if not "traces_hmp" in self.df.columns:

            def _calc_traject_hmp(traces) -> np.array:
                return np.apply_along_axis(TILESET_DEFAULT.request, 1, traces)

            np_hmps = self.df["traces"].progress_apply(_calc_traject_hmp)
            self.df["traces_hmp"] = pd.Series(np_hmps)
            assert not self.df["traces_hmp"].isnull().any()

        def _hmp_entropy(traject) -> float:
            return scipy.stats.entropy(np.sum(traject, axis=0).reshape((-1)))

        self.df["hmpS"] = self.df["traces_hmp"].progress_apply(_hmp_entropy)
        assert not self.df["hmpS"].isnull().any()

        # calc hmpS_c
        threshold_medium, threshold_high = get_class_thresholds(self.df, "hmpS")
        self.df["hmpS_c"] = self.df["hmpS"].progress_apply(
            get_class_name, args=(threshold_medium, threshold_high)
        )
        assert not self.df["hmpS_c"].isnull().any()

    def calc_traces_poles_prc(self) -> None:
        self.df.drop(
            ["poles_prc", "poles_prc_c"], axis=1, errors="ignore", inplace=True
        )

        # calc poles_prc
        def _calc_poles_prc(traces) -> float:
            return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)

        self.df["poles_prc"] = pd.Series(
            self.df["traces"].progress_apply(_calc_poles_prc)
        )

        # calc poles_prc_c
        threshold_medium, threshold_high = get_class_thresholds(self.df, "poles_prc")
        self.df["poles_prc_c"] = self.df["poles_prc"].progress_apply(
            get_class_name, args=(threshold_medium, threshold_high)
        )
        assert not self.df["poles_prc_c"].isna().any()

    def show_entropy_counts(self) -> None:
        log.info(count_entropy_str(self.df))
        log.info(f"df['actS'].max()={self.df['actS'].max()}")
        log.info(f"df['actS'].min()={self.df['actS'].min()}")

    def show_entropy_histogram(self, facet=None) -> None:
        px.histogram(
            self.df,
            x="actS",
            color="actS_c",
            facet_col=facet,
            color_discrete_map=ENTROPY_CLASS_COLORS,
            width=900,
        ).show()

    # -- tileset

    def calc_tileset_reqs_metrics(self, tileset_l: list[TileSet]) -> None:
        if len(self.df) >= 4:
            log.info("df.size >= 4, it will take for some time")

        def _trace_mestrics_np(trace, tileset) -> np.array:
            heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
            return np.array([np.sum(heatmap), vp_quality, area_out])

        def _traject_metrics_np(traces, tileset) -> np.array:
            return np.apply_along_axis(_trace_mestrics_np, 1, traces, tileset=tileset)

        for tileset in tileset_l:
            column_name = f"metrics_{tileset.title}"
            metrics_np = self.df["traces"].progress_apply(
                _traject_metrics_np, tileset=tileset
            )
            self.df[column_name] = metrics_np
            assert not self.df[column_name].empty

    def show_tileset_reqs_metrics(self) -> None:
        # check
        columns = [
            column for column in self.df.columns if column.startswith("metrics_")
        ]
        assert len(columns), "run calc_tileset_reqs_metrics"
        # create dftmp
        data = []
        for name in [
            column for column in self.df.columns if column.startswith("metrics_")
        ]:
            avg_reqs = float(
                self.df[name].apply(lambda traces: np.sum(traces[:, 0])).mean()
            )
            avg_qlt = self.df[name].apply(lambda traces: np.sum(traces[:, 1])).mean()
            avg_lost = self.df[name].apply(lambda traces: np.sum(traces[:, 2])).mean()
            score = avg_qlt / avg_lost
            data.append(
                (name.removeprefix("metrics_"), avg_reqs, avg_qlt, avg_lost, score)
            )
        assert len(data) > 0
        columns = ["tileset", "avg_reqs", "avg_qlt", "avg_lost", "score"]
        dftmp = pd.DataFrame(data, columns=columns)
        # show dftmp
        fig = make_subplots(
            rows=4, cols=1, subplot_titles=columns[1:], shared_yaxes=True
        )
        trace = go.Bar(y=dftmp["tileset"], x=dftmp["avg_reqs"], orientation="h")
        fig.add_trace(trace, row=1, col=1)
        trace = go.Bar(y=dftmp["tileset"], x=dftmp["avg_lost"], orientation="h")
        fig.add_trace(trace, row=2, col=1)
        trace = go.Bar(y=dftmp["tileset"], x=dftmp["avg_qlt"], orientation="h")
        fig.add_trace(trace, row=3, col=1)
        trace = go.Bar(y=dftmp["tileset"], x=dftmp["score"], orientation="h")
        fig.add_trace(trace, row=4, col=1)
        fig.update_layout(
            width=500,
            showlegend=False,
            barmode="stack",
        )
        fig.show()

    def partition(
        self, entropy_filter: str, train_size=0.8, val_size=0.25, test_size=0.2
    ) -> None:
        log.info(
            f"{train_size=}, {test_size=}. {val_size=} of train, so {train_size*val_size}"
        )
        # first split into x_train and x_test
        self.x_train, self.x_test = train_test_split(
            self.df,
            random_state=1,
            train_size=train_size,
            test_size=test_size,
            stratify=self.df["actS_c"],
        )
        # then split x_train in final x_train and x_val
        self.x_train, self.x_val = train_test_split(
            self.x_train,
            random_state=1,
            test_size=val_size,
            stratify=self.x_train["actS_c"],
        )
        log.info("trajectories x_train has " + count_entropy_str(self.x_train))
        log.info("trajectories x_val has " + count_entropy_str(self.x_val))
        log.info("trajectories x_test has " + count_entropy_str(self.x_test))

        if entropy_filter != "all":
            log.info("entropy_filter != all, so filtering x_train, x_val")
            self.x_train = filter_df_by_entropy(self.x_train, entropy_filter)
            self.x_val = filter_df_by_entropy(self.x_val, entropy_filter)
            log.info(
                "filtred trajectories x_train has " + count_entropy_str(self.x_train)
            )
            log.info("filtred trajectories x_val has " + count_entropy_str(self.x_val))
            log.info(
                "filtred trajectories x_test has " + count_entropy_str(self.x_test)
            )

    def create_wins(self, init_window: int, h_window: int) -> None:
        self.x_train_wins = [
            {"video": row[1]["video"], "user": row[1]["user"], "trace_id": trace_id}
            for row in self.x_train.iterrows()
            for trace_id in range(init_window, row[1]["traces"].shape[0] - h_window)
        ]
        self.x_val_wins = [
            {
                "video": row[1]["video"],
                "user": row[1]["user"],
                "trace_id": trace_id,
            }
            for row in self.x_val.iterrows()
            for trace_id in range(init_window, row[1]["traces"].shape[0] - h_window)
        ]
        self.x_test_wins = [
            {
                "video": row[1]["video"],
                "user": row[1]["user"],
                "trace_id": trace_id,
                "actS_c": row[1]["actS_c"],
            }
            for row in self.x_test.iterrows()
            for trace_id in range(init_window, row[1]["traces"].shape[0] - h_window)
        ]

    def show_train_test_split(self) -> None:
        self.x_train["partition"] = "train"
        self.x_test["partition"] = "test"
        df = pd.concat([self.x_train, self.x_test])
        df.show_histogram(facet="partition")
