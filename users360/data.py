from os.path import exists
import numpy as np
import os
import pathlib
import pickle
import pandas as pd
import scipy.stats
from .tileset import *
import swifter
ONE_VIDEO = '10_Cows'


class Data:
    # main data
    df_trajects = pd.DataFrame()
    df_users = pd.DataFrame()
    df_tileset_metrics = pd.DataFrame()

    # singleton
    _instance = None
    _instance_pickle = pathlib.Path(__file__).parent.parent / 'output/singleton.pickle'

    def __init__(self):
        if self.df_trajects.empty:
            print('loading Trajectories from head_motion_prediction')
            self._load_dataset()

    @classmethod
    def dump(cls):
        with open(cls._instance_pickle, 'wb') as f:
            pickle.dump(cls._instance, f)

    @classmethod
    def singleton(cls):
        if cls._instance is None:
            if exists(cls._instance_pickle):
                with open(cls._instance_pickle, 'rb') as f:
                    print(f"loading Trajectories from {cls._instance_pickle}")
                    cls._instance = pickle.load(f)
            else:
                cls._instance = Data()
        return cls._instance

    def _load_dataset(self):
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
        self.df_trajects = pd.concat(map(load_dataset_xyz, ds_idxs), ignore_index=True)
        self.df_trajects.insert(0, 'user', self.df_trajects.groupby(['ds', 'ds_user']).ngroup())
        # create df_users
        self.df_users = pd.DataFrame()
        self.df_users['user'] = self.df_trajects.groupby(['user']).ngroup().unique()

        # back to cwd
        os.chdir(cwd)

    def get_trajects(self, users=list[str], videos=list[str], perc=1.0):
        assert (perc <= 1.0 and perc >= 0.0)
        df = self.df_trajects
        if not users:
            df = df.loc[df.user.isin(users)]
        if not videos:
            df = df.loc[df.video.isin(videos)]
        if perc == 1.0:
            return df
        else:
            size = df.size
            step = int(size / (size * perc))
            return df[::step]

    def get_one_trace(self):
        return self.df_trajects.iloc[0]['traces'][0]

    def get_one_traject(self):
        return self.df_trajects.head(1)


def calc_poles_prc():
    df = Data.singleton().df_trajects
    f_traject = lambda traces: np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)
    df['poles_prc'] = pd.Series(df['traces'].apply(f_traject))
    idxs_sort = df['poles_prc'].argsort()
    trajects_len = len(df['poles_prc'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df['poles_prc'][idx_threshold_medium]
    threshold_hight = df['poles_prc'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df['poles_class'] = df['poles_prc'].apply(f_threshold)


def calc_hmps(tileset=TileSet.default()):
    df = Data.singleton().df_trajects
    f_trace = lambda trace: tileset.request(trace)
    f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
    df['hmps'] = pd.Series(df['traces'].swifter.apply(f_traject))


def calc_entropy():
    df = Data.singleton().df_trajects
    if 'hmps' not in df.columns:
        calc_hmps()
    # calc entropy
    f_entropy = lambda x: scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
    df['entropy'] = df['hmps'].swifter.apply(f_entropy, axis=1)
    # calc class
    idxs_sort = df['entropy'].argsort()
    trajects_len = len(df['entropy'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df['entropy'][idx_threshold_medium]
    threshold_hight = df['entropy'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df['entropy_class'] = df['entropy'].apply(f_threshold)


def calc_entropy_users():
    df = Data.singleton().df_trajects
    df_users = Data.singleton().df_users

    if 'hmps' not in df.columns:
        calc_hmps()

    # calc entropy
    def f_entropy_user(trajects):
        hmps_sum = np.sum(np.sum(trajects['hmps'].to_numpy(), axis=0), axis=0)
        return scipy.stats.entropy(hmps_sum.reshape((-1)))
    df_users['entropy'] = df.groupby(['user']).apply(f_entropy_user)

    # calc class
    idxs_sort = df_users['entropy'].argsort()
    trajects_len = len(df_users['entropy'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df_users['entropy'][idx_threshold_medium]
    threshold_hight = df_users['entropy'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df_users['entropy_class'] = df_users['entropy'].apply(f_threshold)


def calc_tileset_metrics(tileset_l: list[TileSetIF], n_trajects=None):
    assert(not Data.singleton().df_trajects.empty)
    if n_trajects:
        df = Data.singleton().df_trajects[:n_trajects]
    else:
        df = Data.singleton().df_trajects
    
    def create_tsdf(ts_idx):
        tileset = tileset_l[ts_idx]
        def f_trace(trace):
            heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
            return (int(np.sum(heatmap)), vp_quality, area_out)
        f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
        tmpdf = pd.DataFrame(df['traces'].swifter.apply(f_traject))
        tmpdf.columns = [tileset.title]
        return tmpdf
    df_tileset_metrics = pd.concat(map(create_tsdf, range(len(tileset_l))), axis=1)
    Data.singleton().df_tileset_metrics = df_tileset_metrics
