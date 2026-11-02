from __future__ import annotations

import hashlib
import logging
import math
import os
import pathlib
import pickle
from os import mkdir
from os.path import exists
from typing import Any, Literal, Optional

import numpy as np
import pooch

try:
    import pandas as pd
    from jenkspy import jenks_breaks
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
except ImportError:
    pd = Any

from predict360user.run_config import RunConfig
from predict360user.utils.math360 import calc_actual_entropy

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DOT_DATA_DIR = _REPO_ROOT / ".data"
_LEGACY_DATA_DIR = _REPO_ROOT / "data"
DATADIR = f"{_DOT_DATA_DIR if _DOT_DATA_DIR.exists() else _LEGACY_DATA_DIR}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"

DATASET_ARCHIVES = {
    "david": "David_MMSys_18.zip",
    "fan": "Fan_NOSSDAV_17.zip",
    "nguyen": "Nguyen_MM_18.zip",
    "xucvpr": "Xu_CVPR_18.zip",
    "xupami": "Xu_PAMI_18.zip",
}

DATASET_SUBDIRS = {
    "david": "David_MMSys_18",
    "fan": "Fan_NOSSDAV_17",
    "nguyen": "Nguyen_MM_18",
    "xucvpr": "Xu_CVPR_18",
    "xupami": "Xu_PAMI_18",
}

POOCH = pooch.create(
    path=_DOT_DATA_DIR,
    base_url="",
    env="DATA_DIR",
    registry={
        "David_MMSys_18.zip": None,
        "Fan_NOSSDAV_17.zip": None,
        "Nguyen_MM_18.zip": None,
        "Xu_CVPR_18.zip": None,
        "Xu_PAMI_18.zip": None,
    },
    urls={
        "David_MMSys_18.zip": "doi:10.5281/zenodo.4673531/head-motion-prediction.zip",
        "Fan_NOSSDAV_17.zip": "https://gitlab.com/miguelfromeror/head-motion-prediction/-/archive/master/head-motion-prediction-master.zip",
        "Nguyen_MM_18.zip": "https://www.dropbox.com/s/r09s1yqr2qyluyr/salient_ds.zip?dl=1",
        "Xu_CVPR_18.zip": "https://gitlab.com/miguelfromeror/head-motion-prediction/-/archive/master/head-motion-prediction-master.zip",
        "Xu_PAMI_18.zip": "https://www.dropbox.com/s/swenk8b33vs6151/PVS-HM.tar.gz?dl=1",
    },
)

DATA_POOCH = POOCH
DATASET_REGISTRY = POOCH.registry
DATASET_URLS = POOCH.urls


def fetch_dataset(dataset: str = RunConfig.dataset, download: bool = False) -> dict[str, Any]:
    norm_name = dataset.lower().strip()
    fname = DATASET_ARCHIVES[norm_name]
    subdir = DATASET_SUBDIRS[norm_name]
    url = POOCH.urls[fname]

    dot_target = _DOT_DATA_DIR / "head_motion_prediction" / subdir
    legacy_target = pathlib.Path(HMDDIR) / subdir
    target_dir = dot_target if dot_target.exists() or not legacy_target.exists() else legacy_target
    target_dir.mkdir(parents=True, exist_ok=True)

    if download and fname in POOCH.registry:
        try:
            POOCH.fetch(fname, processor=pooch.Unzip(extract_dir=str(dot_target)))
            target_dir = dot_target
        except Exception:
            pass

    files = sorted(
        str(p.relative_to(target_dir))
        for p in target_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    h = hashlib.sha256()
    total_bytes = 0
    for rel in files:
        sz = (target_dir / rel).stat().st_size
        total_bytes += sz
        h.update(f"{rel}:{sz}\n".encode("utf-8"))

    return {
        "name": norm_name,
        "fname": fname,
        "url": url,
        "local_dir": str(target_dir),
        "num_files": len(files),
        "total_bytes": total_bytes,
        "content_hash": h.hexdigest()[:16] if files else "",
        "verified": len(files) > 0,
    }


class Trajectory360Dataset:
    def __init__(
        self,
        name: str = RunConfig.dataset,
        root: Optional[str] = None,
        split: str = "train",
        download: bool = False,
    ):
        self.name = name.lower().strip()
        self.root = root
        self.split = split
        self.manifest = fetch_dataset(self.name, download=download)
        local_dir = pathlib.Path(self.manifest["local_dir"])
        self.files = sorted(
            str(p) for p in local_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
        )

    def get_split_paths(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, list[str]]:
        n_total = len(self.files)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        return {
            "train": self.files[:n_train],
            "val": self.files[n_train : n_train + n_val],
            "test": self.files[n_train + n_val :] or self.files[n_train : n_train + n_val],
        }

    def __len__(self) -> int:
        if self.split in ("train", "val", "test"):
            return len(self.get_split_paths()[self.split])
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        if self.split in ("train", "val", "test"):
            return self.get_split_paths()[self.split][idx]
        return self.files[idx]


def resolve_dataset_manifest(dataset: str = RunConfig.dataset, download: bool = False) -> dict[str, Any]:
    if dataset == "all":
        manifests = {
            ds_key: fetch_dataset(ds_key, download=download)
            for ds_key in ("david", "fan", "nguyen", "xucvpr", "xupami")
        }
        return {"dataset": "all", "manifests": manifests}
    return fetch_dataset(dataset, download=download)
DATASETS = {
    "david": {"size": 1083},
    "fan": {"size": 300},
    "nguyen": {"size": 432},
    "xucvpr": {"size": 6654},
    "xupami": {"size": 4408},
}
DEFAULT_SAVEDIR = "./saved"
ENTROPY_NAMES = ["low", "medium", "high"]

log = logging.getLogger()


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


def _load_df_trajecs_from_hmp(dataset: str) -> pd.DataFrame:
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
    if dataset == "all":
        target = DATASETS
    else:
        target = {dataset: DATASETS[dataset]}
    n_traces = 100

    def _load_dataset_xyz(ds_name, ds_dict) -> pd.DataFrame:
        # create_and_store_sampled_dataset()
        # stores csv at head_motion_prediction/<dataset>/sampled_dataset
        if len(os.listdir(ds_dict["pkg"].OUTPUT_FOLDER)) < 2:
            ds_dict["pkg"].create_and_store_sampled_dataset()
        # then call load_sample_dateset()
        # and convert csv as dict with
        # {<video1>:{
        #   <user1>:[time-stamp, x, y, z],
        #    ...
        #  },
        #  ...
        # }"
        dataset = ds_dict["pkg"].load_sampled_dataset()
        # time: np.around(dataset[user][video][:n_traces, 0], decimals=2)
        data = [
            (ds_name, user, video, dataset[user][video][:n_traces, 1:])
            for user in dataset.keys()
            for video in dataset[user].keys()
        ]
        tmpdf = pd.DataFrame(
            data,
            columns=["ds", "user", "video", "traces"],
        )
        # assert and check
        assert len(tmpdf["ds"]) == ds_dict["size"]
        return tmpdf

    # create df for each dataset
    df = pd.concat(
        [_load_dataset_xyz(ds_name, ds_dict) for ds_name, ds_dict in target.items()],
        ignore_index=True,  # otherwise the indexes are duplicated
    ).convert_dtypes()
    assert not df.empty
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
    os.chdir(cwd)
    return df


def _calc_traces_entropy(df) -> pd.DataFrame:
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


def load_df_trajecs(dataset=RunConfig.dataset) -> pd.DataFrame:
    assert dataset in ["all"] + list(DATASETS.keys())
    pickle_file = os.path.join(DEFAULT_SAVEDIR, f"df_trajecs_{dataset}.pickle")
    df = None
    if exists(pickle_file):
        try:
            with open(pickle_file, "rb") as f:
                log.info(f"loading df from {pickle_file}")
                df = pickle.load(f)
        except Exception as e:
            log.warning(f"failed to unpickle {pickle_file} ({e}), regenerating...")
            df = None
    if df is None:
        log.info(f"loading trajects from {HMDDIR}")
        df = _load_df_trajecs_from_hmp(dataset)
        log.info(f"calculating entropy")
        df = _calc_traces_entropy(df)
        log.info(f"saving trajects to {pickle_file} for fast loading")
        if not exists(DEFAULT_SAVEDIR):
            mkdir(DEFAULT_SAVEDIR)
        with open(pickle_file, "wb") as f:
            pickle.dump(df, f)
    return df


def load_df_wins(
    dataset=RunConfig.dataset,
    m_window=RunConfig.m_window,
    init_window=RunConfig.init_window,
    h_window=RunConfig.h_window,
) -> pd.DataFrame:
    df_trajects = load_df_trajecs(dataset)

    df = df_trajects

    # create "trace_id" list as explode it duplicating other columns
    def _create_trace_id(traces) -> list[int]:
        return [trace_id for trace_id in range(init_window, traces.shape[0] - h_window)]

    df["trace_id"] = df_trajects["traces"].apply(_create_trace_id)
    df = df.explode("trace_id", ignore_index=True)
    df = df.dropna(subset=["trace_id"], how="all")

    # m_window and f_window
    def _create_m_window(row) -> np.ndarray:
        trace_id = row["trace_id"]
        return row["traces"][trace_id - m_window : trace_id]

    def _create_trace_pos(row) -> np.ndarray:
        trace_id = row["trace_id"]
        return row["traces"][trace_id : trace_id + 1]

    def _create_h_window(row) -> np.ndarray:
        trace_id = row["trace_id"]
        return row["traces"][trace_id + 1 : trace_id + h_window + 1]

    df["m_window"] = df.apply(_create_m_window, axis=1)
    df["trace"] = df.apply(_create_trace_pos, axis=1)
    df["h_window"] = df.apply(_create_h_window, axis=1)
    df = df.drop(["traces", "trace_id"], axis=1)

    del df_trajects
    return df


def split(
    df: pd.DataFrame,
    seed: int,
    train_size=RunConfig.train_size,
    test_size=RunConfig.test_size,
    val_size=0.25,
) -> pd.DataFrame:
    df["partition"] = "discarted"  # sanity check
    log.info(f"{train_size=} (with {val_size=}), {test_size=}")

    # split train and test
    train, test = train_test_split(
        df,
        random_state=seed,
        train_size=train_size,
        test_size=test_size,
        stratify=df["actS_c"],
    )
    log.info("samples/train are " + count_entropy_str(train))

    # split train and val
    train_before_val_split = len(train)
    train, val = train_test_split(
        train,
        random_state=seed,
        test_size=val_size,
        stratify=train["actS_c"],
    )
    log.info("samples/train/val are " + count_entropy_str(val))
    log.info("samples/test are " + count_entropy_str(test))

    # save partition as column
    df.loc[train.index, "partition"] = "train"
    df.loc[val.index, "partition"] = "val"
    df.loc[test.index, "partition"] = "test"
    train_len = len(df[df["partition"] == "train"])
    val_len = len(df[df["partition"] == "val"])
    assert (train_len + val_len) == train_before_val_split

    return df


def split_train_filtred(
    df: pd.DataFrame,
    train_entropy: str,
    seed: int = 1,
    train_size=RunConfig.train_size,
    test_size=RunConfig.test_size,
    val_size=0.25,
    train_minsize=False,
) -> pd.DataFrame:
    assert train_entropy in ENTROPY_NAMES
    df["partition"] = "discarted"

    train, test = train_test_split(
        df,
        random_state=seed,
        train_size=1 - test_size,
        test_size=test_size,
        stratify=df["actS_c"],
    )

    n_train = min(math.ceil(len(df) * train_size), len(train[train["actS_c"] == train_entropy]))
    train = train[train["actS_c"] == train_entropy].sample(n=n_train, random_state=seed)
    train, val = train_test_split(
        train,
        random_state=seed,
        test_size=val_size,
        stratify=train["actS_c"],
    )

    df.loc[train.index, "partition"] = "train"
    df.loc[val.index, "partition"] = "val"
    df.loc[test.index, "partition"] = "test"

    return df