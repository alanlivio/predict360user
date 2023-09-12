import logging
import os
import pickle
from os.path import basename, exists
from typing import Literal
from sklearn.utils import shuffle

from jenkspy import jenks_breaks
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from predict360user.plot360 import Plot360

from predict360user.tileset import TILESET_DEFAULT
from predict360user.utils import *

DATASETS = {
    "david": {"size": 1083},
    "fan": {"size": 300},
    "nguyen": {"size": 432},
    "xucvpr": {"size": 6654},
    "xupami": {"size": 4408},
}
log = logging.getLogger(basename(__file__))


def get_class_thresholds(df, col: str) -> tuple[float, float]:
    _, threshold_medium, threshold_high, _ = jenks_breaks(df[col], n_classes=3)
    return threshold_medium, threshold_high


def get_class_name(
    x: float, threshold_medium: float, threshold_high: float
) -> Literal["low", "medium", "high"]:
    return (
        "low" if x < threshold_medium else ("medium" if x < threshold_high else "high")
    )


def filter_by_entropy(
    df: pd.DataFrame, entropy: str, minsize: bool
) -> pd.DataFrame:
    assert entropy in ENTROPY_NAMES
    assert len(df["actS_c"].unique()) == 3 # low, medium, high
    if entropy == "all":
        df_filtred = df
    elif entropy == "nohigh":
        df_filtred = df[df["actS_c"] != "high"]
    elif entropy == "nolow":
        df_filtred = df[df["actS_c"] != "low"]
    else:
        df_filtred = df[df["actS_c"] == entropy]
    if minsize:
        minsize_among_classes = df["actS_c"].value_counts().min()
        final_n_classes = len(df_filtred["actS_c"].unique())
        n_sample_per_class = int(minsize_among_classes/final_n_classes)
        df_filtred = (
            df_filtred.groupby("actS_c")
            .apply(lambda x: x.sample(n=n_sample_per_class, random_state=1))
            .droplevel(0)  # undo groupby
        )
    return df_filtred


def count_entropy(df: pd.DataFrame) -> tuple[int, int, int, int]:
    a_len = len(df)
    l_len = len(df[df["actS_c"] == "low"])
    m_len = len(df[df["actS_c"] == "medium"])
    h_len = len(df[df["actS_c"] == "high"])
    return a_len, l_len, m_len, h_len


def count_entropy_str(df: pd.DataFrame) -> str:
    return "{}: {} low, {} medium, {} high".format(*count_entropy(df))


class Dataset:
    """:class:`Dataset` stores the original dataset in memory.
    It provides functions for data preprocessing, such user clustering by entropy, and analyses, such as tileset usage.
    Features are stored as :class:`pandas.DataFrame`.
    Attributes:
        df (str): pandas.DataFrame.
    """
    df: pd.DataFrame
    
    def __init__(self, dataset_name="all", savedir=DEFAULT_SAVEDIR) -> None:
        assert dataset_name in ["all"] + list(DATASETS.keys())
        self.dataset_name = dataset_name
        pickle_file = os.path.join(savedir, f"df_trajecs_{dataset_name}.pickle")
        if exists(pickle_file):
            with open(pickle_file, "rb") as f:
                log.info(f"loading df from {pickle_file}")
                self.df = pickle.load(f)
        else:
            log.info(f"there is no {pickle_file}")
            log.info(f"loading trajects from {HMDDIR}")
            self.df = self._load_df_trajecs_from_hmp()
            log.info(f"calculating entropy")
            self.calc_traces_entropy()
            log.info(f"saving trajects to {pickle_file} for fast loading")
            with open(pickle_file, "wb") as f:
                pickle.dump(self.df, f)

    def _load_df_trajecs_from_hmp(self) -> pd.DataFrame:
        # save cwd and move to head_motion_prediction for invoking funcs
        cwd = os.getcwd()
        os.chdir(HMDDIR)
        from .head_motion_prediction.David_MMSys_18 import Read_Dataset as david
        from .head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
        from .head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
        from .head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
        from .head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami

        DATASETS["david"]["pkg"] = david
        DATASETS["fan"]["pkg"] = fan
        DATASETS["nguyen"]["pkg"] = nguyen
        DATASETS["xucvpr"]["pkg"] = xucvpr
        DATASETS["xupami"]["pkg"] = xupami
        if self.dataset_name == "all":
            target = DATASETS
        else:
            target = {self.dataset_name: DATASETS[self.dataset_name]}
        n_traces = 100

        def _load_dataset_xyz(key, value) -> pd.DataFrame:
            # create_and_store_sampled_dataset()
            # stores csv at head_motion_prediction/<dataset>/sampled_dataset
            if len(os.listdir(value["pkg"].OUTPUT_FOLDER)) < 2:
                value["pkg"].create_and_store_sampled_dataset()
            # then call load_sample_dateset()
            # and convert csv as dict with
            # {<video1>:{
            #   <user1>:[time-stamp, x, y, z],
            #    ...
            #  },
            #  ...
            # }"
            dataset = value["pkg"].load_sampled_dataset()
            # time: np.around(dataset[user][video][:n_traces, 0], decimals=2)
            data = [
                (key, user, video, dataset[user][video][:n_traces, 1:])
                for user in dataset.keys()
                for video in dataset[user].keys()
            ]
            tmpdf = pd.DataFrame(
                data,
                columns=["ds", "user", "video", "traces"],
            )
            # assert and check
            assert len(tmpdf["ds"]) == value["size"]
            return tmpdf

        # create df for each dataset
        df = pd.concat(
            [_load_dataset_xyz(k, v) for k, v in target.items()]
        ).convert_dtypes()
        assert not df.empty
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
        df = df.set_index(["ds", "user", "video"])
        # back to cwd
        os.chdir(cwd)
        return df

    def sample_trace(self) -> np.array:
        traject_ar = self.df.sample(1).iloc[0]["traces"]
        trace = traject_ar[np.random.randint(len(traject_ar - 1))]
        return trace

    def calc_traces_entropy(self) -> None:
        self.df.drop(["actS", "actS_c"], axis=1, errors="ignore", inplace=True)
        tqdm.pandas(desc=f"calc actS")
        self.df["actS"] = (
            self.df["traces"].progress_apply(calc_actual_entropy).astype(float)
        )
        assert not self.df["actS"].isnull().any()
        threshold_medium, threshold_high = get_class_thresholds(self.df, "actS")
        self.df["actS_c"] = (
            self.df["actS"]
            .apply(get_class_name, args=(threshold_medium, threshold_high))
            .astype("string")
        )
        assert not self.df["actS_c"].isnull().any()

    def calc_traces_hmps(self) -> None:
        self.df.drop(["traces_hmps"], axis=1, errors="ignore", inplace=True)

        def _calc_traject_hmp(traces) -> np.array:
            return np.apply_along_axis(TILESET_DEFAULT.request, 1, traces)

        tqdm.pandas(desc=f"calc traces_hmps")
        np_hmps = self.df["traces"].progress_apply(_calc_traject_hmp)
        self.df["traces_hmps"] = pd.Series(np_hmps)
        assert not self.df["traces_hmps"].isnull().any()

    def calc_traces_poles_prc(self) -> None:
        def _calc_poles_prc(traces) -> float:
            return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)

        self.df.drop(
            ["poles_prc", "poles_prc_c"], axis=1, errors="ignore", inplace=True
        )
        tqdm.pandas(desc=f"calc poles_prc")
        self.df["poles_prc"] = pd.Series(
            self.df["traces"].progress_apply(_calc_poles_prc).astype(float)
        )
        threshold_medium, threshold_high = get_class_thresholds(self.df, "poles_prc")
        self.df["poles_prc_c"] = (
            self.df["poles_prc"]
            .apply(get_class_name, args=(threshold_medium, threshold_high))
            .astype("string")
        )
        assert not self.df["poles_prc_c"].isna().any()

    def partition(
        self,
        train_entropy="all",
        train_size=0.8,
        val_size=0.25,
        test_size=0.2,
        minsize=False,
    ) -> None:
        self.df.drop(["partition"], axis=1, errors="ignore", inplace=True)
        log.info(f"{train_size=} (with {val_size=}), {test_size=}")
        
        # split into train and test
        self.train, self.test = train_test_split(
            self.df,
            random_state=1,
            train_size=train_size,
            test_size=test_size,
            stratify=self.df["actS_c"],
        )
        log.info("train trajecs are " + count_entropy_str(self.train))
        
        # split train in train and val
        if train_entropy != "all" or minsize:
            log.info(f"{train_entropy=} and {minsize=}, so filtering train")
            self.train = filter_by_entropy(self.train, train_entropy, minsize)
            log.info("train trajecs (filtred) are " + count_entropy_str(self.train))
        self.train, self.val = train_test_split(
            self.train,
            random_state=1,
            test_size=val_size,
            stratify=self.train["actS_c"],
        )
        log.info("val trajecs are " + count_entropy_str(self.val))
        log.info("test trajecs are " + count_entropy_str(self.test))
        
        # save "partition" column for show_entropy_histogram_per_partition
        self.df.loc[self.train.index, "partition"] = "train"
        self.df.loc[self.val.index, "partition"] = "val"
        self.df.loc[self.test.index, "partition"] = "test"

    def create_wins(self, init_window: int, h_window: int) -> None:
        # create trace_id column
        f_list_trace_id = lambda traces: [
            trace_id for trace_id in range(init_window, traces.shape[0] - h_window)
        ]
        self.train["trace_id"] = self.train["traces"].apply(f_list_trace_id)
        self.val["trace_id"] = self.val["traces"].apply(f_list_trace_id)
        self.test["trace_id"] = self.test["traces"].apply(f_list_trace_id)

        # create train_wins, val_wins
        self.train_wins = self.train.explode("trace_id").reset_index()[
            ["ds", "user", "video", "trace_id"]
        ]
        self.train_wins.dropna(subset=["trace_id"], how="all", inplace=True)
        self.train_wins = shuffle(self.train_wins, random_state=1)

        self.val_wins = self.val.explode("trace_id").reset_index()[
            ["ds", "user", "video", "trace_id"]
        ]
        self.val_wins.dropna(subset=["trace_id"], how="all", inplace=True)
        self.val_wins = shuffle(self.val_wins, random_state=1)

        # create test_wins
        self.test_wins = self.test.explode("trace_id").reset_index()[
            ["ds", "user", "video", "trace_id", "actS_c"]
        ]
        self.test_wins.dropna(subset=["trace_id"], how="all", inplace=True)
        self.test = shuffle(self.test, random_state=1)

    def show_traject(self, row: pd.Series) -> None:
        assert "traces" in row.index
        traces = row["traces"]
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]]
        )

        # add traces
        plot = Plot360()
        plot.add_traces(traces)
        for d in plot.data:  # load all data from the self
            fig.append_trace(d, row=1, col=1)

        # add hmps_sum
        traces_hmps = np.apply_along_axis(TILESET_DEFAULT.request, 1, row["traces"])
        hmps_sum = np.sum(traces_hmps, axis=0)
        x = [str(x) for x in range(1, hmps_sum.shape[1] + 1)]
        y = [str(y) for y in range(1, hmps_sum.shape[0] + 1)]
        erp_heatmap = px.imshow(hmps_sum, text_auto=True, x=x, y=y)
        erp_heatmap.update_layout(width=100, height=100)

        # show fig
        fig.append_trace(erp_heatmap.data[0], row=1, col=2)
        # fix given phi 0 being the north pole at cartesian_to_eulerian
        fig.update_yaxes(autorange="reversed")
        title = f"trajec_{row.name}_[{TILESET_DEFAULT.prefix}]"
        fig.update_layout(width=800, showlegend=False, title_text=title)
        fig.show()

    def show_entropy_histogram(self) -> None:
        assert "actS" in self.df.columns
        px.histogram(
            self.df.dropna(),
            x="actS",
            color="actS_c",
            color_discrete_map=ENTROPY_CLASS_COLORS,
            width=900,
            category_orders={"actS": ["low", "medium", "high"]},
        ).show()

    def show_entropy_histogram_per_partition(self) -> None:
        assert "partition" in self.df.columns
        px.histogram(
            self.df.dropna(),
            x="actS",
            color="actS_c",
            facet_col="partition",
            color_discrete_map=ENTROPY_CLASS_COLORS,
            category_orders={
                "actS": ["low", "medium", "high"],
                "partition": ["train", "val", "test"],
            },
            width=900,
        ).show()
