from __future__ import annotations
from os.path import exists
import os
from .config import *
import pickle
import pandas as pd
import numpy as np
import logging

class Savable():

    @classmethod
    def _pickle_file(cls):
        return f'{DATADIR}/{cls.__name__}.pickle'

    def save(self):
        with open(self._pickle_file(), 'wb') as f:
            pickle.dump(self, f)

    def delete_saved(cls):
        if exists(cls._pickle_file()):
            logging.info(f"removing {cls._pickle_file()}")
            os.remove(cls._pickle_file())

    @classmethod
    def load_or_create(cls):
        if exists(cls._pickle_file()):
            with open(cls._pickle_file(), 'rb') as f:
                logging.info(f"loading {cls.__name__} from {cls._pickle_file()}")
                return pickle.load(f)
        else:
            return cls()


class Data(Savable):
    # trajects processed data
    df_trajects = pd.DataFrame()
    df_users = pd.DataFrame()

    # tileset processed data
    ts_polys = dict()
    ts_centers = dict()
    tsv_polys = dict()
    fov_points = dict()
    fov_polys = dict()
    df_tileset_metrics = pd.DataFrame()

    # singleton
    _instance = None

    @classmethod
    def singleton(cls) -> Data:
        if cls._instance is None:
            cls._instance = Data.load_or_create()
        return cls._instance

    def load_dataset(self):
        logging.info('loading trajects from head_motion_prediction project')
        # save cwd and move to head_motion_prediction
        cwd = os.getcwd()
        os.chdir(HMDDIR)

        from .head_motion_prediction.David_MMSys_18 import Read_Dataset as david
        from .head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
        from .head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
        from .head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
        from .head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami
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
        self.df_trajects = pd.concat(map(load_dataset_xyz, ds_idxs), ignore_index=True)
        self.df_trajects.insert(0, 'user', self.df_trajects.groupby(['ds', 'ds_user']).ngroup())
        # create df_users
        self.df_users = pd.DataFrame()
        self.df_users['user'] = self.df_trajects.groupby(['user']).ngroup().unique()

        # back to cwd
        os.chdir(cwd)
