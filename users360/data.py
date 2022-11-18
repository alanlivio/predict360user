from __future__ import annotations

import logging
import os
import pathlib
import pickle
from os.path import exists

import numpy as np
import pandas as pd

DATADIR = f"{pathlib.Path(__file__).parent.parent / 'data/'}"
HMDDIR = f"{pathlib.Path(__file__).parent / 'head_motion_prediction/'}"
DS_NAMES = ['david', 'fan', 'nguyen', 'xucvpr', 'xupami']

logging.basicConfig(level=logging.INFO, format='-- %(filename)s: %(message)s')

# Singleton following https://python-patterns.guide/gang-of-four/singleton/


class Data():

    # singleton and its pickle filename
    _instance = None
    _pickle_f = None

    # centralized data
    df_users: pd.Dataframe
    df_trajects: pd.Dataframe

    def __init__(self) -> None:
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls, pickle_sufix='') -> Data:
        if cls._instance is not None:
            return cls._instance

        filename = f'data_{pickle_sufix}.pickle' if pickle_sufix else 'data.pickle'
        cls._pickle_f = os.path.join(DATADIR, filename)
        if exists(cls._pickle_f):
            with open(cls._pickle_f, 'rb') as f:
                logging.info(f"loading data from {cls._pickle_f}")
                cls._instance: Data = pickle.load(f)
        else:
            cls._instance = cls.__new__(cls)
        if not hasattr(cls._instance, 'df_trajects'):
            cls._instance._load_data()
        return cls._instance

    def save(self) -> None:
        logging.info(f'saving data to {self._pickle_f}')
        with open(self._pickle_f, 'wb') as f:
            pickle.dump(self, f)

    def _load_data(self) -> None:
        logging.info('loading trajects from head_motion_prediction project')
        
        # save cwd and move to head_motion_prediction for invoking funcs
        cwd = os.getcwd()
        os.chdir(HMDDIR)
        from .head_motion_prediction.David_MMSys_18 import \
            Read_Dataset as david
        from .head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
        from .head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
        from .head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
        from .head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami
        DS_PKGS = [david, fan, nguyen, xucvpr, xupami]  # [:1]
        ds_sizes = [1083, 300, 432, 6654, 4408]
        ds_idxs = range(len(DS_PKGS))

        def _load_dataset_xyz(idx, n_traces=100) -> pd.DataFrame:
            # create sampled
            if len(os.listdir(DS_PKGS[idx].OUTPUT_FOLDER)) < 2:
                DS_PKGS[idx].create_and_store_sampled_dataset()
            # each load_sample_dateset return a dict with:
            # {<video1>:{<user1>:[time-stamp, x, y, z], ...}, ...}"
            dataset = DS_PKGS[idx].load_sampled_dataset()

            # df with (dataset, user, video, times, traces)
            # times has only time-stamps
            # traces has only x, y, z (in 3d coordinates)
            data = [(DS_NAMES[idx],
                    DS_NAMES[idx] + '_' + user,
                    DS_NAMES[idx] + '_' + video,
                    # np.around(dataset[user][video][:n_traces, 0], decimals=2),
                     dataset[user][video][:n_traces, 1:]
                     ) for user in dataset.keys() for video in dataset[user].keys()]
            tmpdf = pd.DataFrame(data, columns=[
                'ds', 'ds_user', 'ds_video',
                # 'times',
                'traces'])
            # size check
            assert (tmpdf['ds'].value_counts()[DS_NAMES[idx]] == ds_sizes[idx])
            return tmpdf

        # create df_trajects for each dataset
        self.df_trajects = pd.concat(map(_load_dataset_xyz, ds_idxs), ignore_index=True)
        assert (not self.df_trajects.empty)
        # back to cwd
        os.chdir(cwd)
        # create df_users
        self.df_users = pd.Dataframe()


def get_df_trajects() -> pd.DataFrame:
    return Data.instance().df_trajects


def get_one_trace() -> np.array:
    return Data.instance().df_trajects.iloc[0]['traces'][0]


def get_traces(video, user, ds='David_MMSys_18') -> np.array:
    # TODO: df indexed by (ds, ds_user, ds_video)
    if ds == 'all':
        row = Data.instance().df_trajects.query(f"ds_user=='{user}' and ds_video=='{video}'")
    else:
        row = Data.instance().df_trajects.query(f"ds=='{ds}' and ds_user=='{user}' and ds_video=='{video}'")
    assert (not row.empty)
    return row['traces'].iloc[0]


def get_video_ids(ds='David_MMSys_18') -> np.array:
    df = Data.instance().df_trajects
    return df.loc[df['ds'] == ds]['ds_video'].unique()


def get_user_ids(ds='David_MMSys_18') -> np.array:
    df = Data.instance().df_trajects
    return df.loc[df['ds'] == ds]['ds_user'].unique()
