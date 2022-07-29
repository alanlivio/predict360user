from __future__ import annotations
from .data import *
from .utils import *
from os.path import exists
import os
import pathlib
import pandas as pd
import numpy as np

class Data(Savable):
    # main dfs
    df_trajects = pd.DataFrame()
    df_users = pd.DataFrame()
    df_tileset_metrics = pd.DataFrame()

    # tileset processed data
    ts_polys = dict()
    ts_centers = dict()
    tsv_polys = dict()

    # singleton
    _instance = None

    @classmethod
    def singleton(cls) -> Data:
        if cls._instance is None:
            cls._instance = Data.load_or_create()
            if cls._instance.df_trajects.empty:
                _load_dataset(cls._instance)
        return cls._instance


def _load_dataset(instance):
    print('loading Trajectories from head_motion_prediction')
    # save cwd and move to head_motion_prediction
    project_path = "head_motion_prediction"
    cwd = os.getcwd()
    if os.path.basename(cwd) != project_path:
        os.chdir(pathlib.Path(__file__).parent.parent / 'head_motion_prediction')

    from head_motion_prediction.David_MMSys_18 import Read_Dataset as david
    from head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
    from head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
    from head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
    from head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami

    ds_names = ['David_MMSys_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'Xu_CVPR_18', 'Xu_PAMI_18']
    ds_sizes = [1083, 300, 432, 6654, 4408]
    ds_pkgs = [david, fan, nguyen, xucvpr, xupami][:1]
    ds_idxs = range(len(ds_pkgs))

    def load_dataset_xyz(idx, n_traces=100) -> pd.DataFrame:
        # create sampled
        if len(os.listdir(ds_pkgs[idx].OUTPUT_FOLDER)) < 2:
            ds_pkgs[idx].create_and_store_sampled_dataset()
        dataset = ds_pkgs[idx].load_sampled_dataset()

        # df with (dataset, user, video, times, traces)
        data = [(ds_names[idx],
                user,
                video,
                np.around(dataset[user][video][:n_traces, 0], decimals=2),
                dataset[user][video][:n_traces, 1:]
                 ) for user in dataset.keys() for video in dataset[user].keys()]
        tmpdf = pd.DataFrame(data, columns=['ds', 'ds_user', 'ds_video', 'times', 'traces'])
        # size check
        assert(tmpdf.ds.value_counts()[ds_names[idx]] == ds_sizes[idx])
        return tmpdf

    # create df_trajects for all dataset
    instance.df_trajects = pd.concat(map(load_dataset_xyz, ds_idxs), ignore_index=True)
    instance.df_trajects.insert(0, 'user', instance.df_trajects.groupby(['ds', 'ds_user']).ngroup())
    # create df_users
    instance.df_users = pd.DataFrame()
    instance.df_users['user'] = instance.df_trajects.groupby(['user']).ngroup().unique()

    # back to cwd
    os.chdir(cwd)
