import logging
import os
import pickle
from os.path import basename, exists
from typing import Literal

from jenkspy import jenks_breaks
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from predict360user.utils.utils import *

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


def count_entropy(df: pd.DataFrame) -> tuple[int, int, int, int]:
    a_len = len(df)
    l_len = len(df[df["actS_c"] == "low"])
    m_len = len(df[df["actS_c"] == "medium"])
    h_len = len(df[df["actS_c"] == "high"])
    return a_len, l_len, m_len, h_len


def count_entropy_str(df: pd.DataFrame) -> str:
    return "{}: {} low, {} medium, {} high".format(*count_entropy(df))


def load_df_trajecs(dataset_name="all") -> pd.DataFrame:
    assert dataset_name in ["all"] + list(DATASETS.keys())
    pickle_file = os.path.join(DEFAULT_SAVEDIR, f"df_trajecs_{dataset_name}.pickle")
    if exists(pickle_file):
        with open(pickle_file, "rb") as f:
            log.info(f"loading df from {pickle_file}")
            df = pickle.load(f)
    else:
        log.info(f"there is no {pickle_file}")
        log.info(f"loading trajects from {HMDDIR}")
        df = _load_df_trajecs_from_hmp(dataset_name)
        log.info(f"calculating entropy")
        df = _calc_traces_entropy(df)
        log.info(f"saving trajects to {pickle_file} for fast loading")
        with open(pickle_file, "wb") as f:
            pickle.dump(df, f)
    return df


def create_df_wins(df: pd.DataFrame, init_window=30, h_window=25) -> None:
    # "trace_id" as list
    create_trace_id_l = lambda traces: [
        trace_id for trace_id in range(init_window, traces.shape[0] - h_window)
    ]
    df_wins = df
    df_wins["trace_id"] = df["traces"].apply(create_trace_id_l)
    # explode the list but duplicating other columns
    df_wins = df_wins.drop(["traces"], axis=1)
    df_wins = df_wins.explode("trace_id", ignore_index=True)
    return df_wins


def load_df_wins(dataset_name="all", m_window=5, init_window=30, h_window=25) -> None:
    df_trajects = load_df_trajecs(dataset_name)
    return create_df_wins(df_trajects, m_window=5, init_window=30, h_window=25)


def _load_df_trajecs_from_hmp(dataset_name: str) -> pd.DataFrame:
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
    if dataset_name == "all":
        target = DATASETS
    else:
        target = {dataset_name: DATASETS[dataset_name]}
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
        [_load_dataset_xyz(k, v) for k, v in target.items()],
        ignore_index=True,  # otherwise the indexes are duplicated
    ).convert_dtypes()
    assert not df.empty
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
    os.chdir(cwd)
    return df


def _calc_traces_entropy(df) -> None:
    df.drop(["actS", "actS_c"], axis=1, errors="ignore", inplace=True)
    tqdm.pandas(desc=f"calc actS")
    df["actS"] = df["traces"].progress_apply(calc_actual_entropy).astype(float)
    assert not df["actS"].isnull().any()
    threshold_medium, threshold_high = get_class_thresholds(df, "actS")
    df["actS_c"] = (
        df["actS"]
        .apply(get_class_name, args=(threshold_medium, threshold_high))
        .astype("string")
    )
    assert not df["actS_c"].isnull().any()
    return df


def _filter_train(
    train: pd.DataFrame,
    train_entropy="all",
    train_minsize=False,
) -> None:
    # filter by given entropy
    if train_entropy == "all":
        filtered = train
    elif train_entropy == "nohigh":
        filtered = train[train["actS_c"] != "high"]
    elif train_entropy == "nolow":
        filtered = train[train["actS_c"] != "low"]
    else:
        filtered = train[train["actS_c"] == train_entropy]
    assert len(filtered)

    # filter for limiting to smallest class size in the train
    if train_minsize:
        target_size = train["actS_c"].value_counts().min()
        # stratify https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas
        n_current_classes = len(filtered["actS_c"].unique())
        n_sample_per_class = int(target_size / n_current_classes)
        filtered = filtered.groupby("actS_c", group_keys=False).apply(
            lambda x: x.sample(n=n_sample_per_class, random_state=1)
        )
    assert len(filtered)

    return filtered


def split(
    df: pd.DataFrame,
    train_size=0.8,
    val_size=0.25,
    test_size=0.2,
    train_entropy="",
    train_minsize=False,
) -> None:
    log.info(f"{train_size=} (with {val_size=}), {test_size=}")

    # split train and test
    train, test = train_test_split(
        df,
        random_state=1,
        train_size=train_size,
        test_size=test_size,
        stratify=df["actS_c"],
    )
    log.info("train trajecs are " + count_entropy_str(train))

    # filter train
    if train_entropy or train_minsize:
        train = _filter_train(train, train_entropy, train_minsize)
        log.info("train_filtred trajecs are " + count_entropy_str(train))

    # split train and val
    train_before_val_len = len(train)
    train, val = train_test_split(
        train,
        random_state=1,
        test_size=val_size,
        stratify=train["actS_c"],
    )
    log.info("train.val trajecs are " + count_entropy_str(val))
    log.info("test trajecs are " + count_entropy_str(test))

    # save partition as column
    df["partition"] = np.nan
    df.loc[train.index, "partition"] = "train"
    df.loc[val.index, "partition"] = "val"
    df.loc[test.index, "partition"] = "test"
    train_len = len(df[df["partition"] == "train"])
    val_len = len(df[df["partition"] == "val"])
    assert (train_len + val_len) == train_before_val_len

    return df
