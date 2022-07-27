import head_motion_prediction
from head_motion_prediction.Utils import *
from numpy.typing import NDArray
from os.path import exists
from plotly.subplots import make_subplots
import numpy as np
import os
import pathlib
import pickle
import plotly.graph_objs as go
import scipy.stats
import pandas as pd
from .tileset import *
from .tileset_voro import *
import swifter

ONE_VIDEO = '10_Cows'
np.set_printoptions(suppress=True)
class_color_map = {"hight": "red", "medium": "green", "low": "blue"}


class Trajectories:
    df = pd.DataFrame()
    instance = None
    instance_pickle = pathlib.Path(__file__).parent.parent / 'output/singleton.pickle'

    def __init__(self):
        if self.df.empty:
            print('loading Trajectories from head_motion_prediction')
            self._load_dataset()

    @classmethod
    def dump(cls):
        with open(cls.instance_pickle, 'wb') as f:
            pickle.dump(cls.instance, f)

    @classmethod
    def singleton(cls):
        if cls.instance is None:
            if exists(cls.instance_pickle):
                with open(cls.instance_pickle, 'rb') as f:
                    print(f"loading Trajectories from {cls.instance_pickle}")
                    cls.instance = pickle.load(f)
            else:
                cls.instance = Trajectories()
        return cls.instance

    def _load_dataset(self) -> pd.DataFrame:
        # save cwd and move to head_motion_prediction
        project_path = "head_motion_prediction"
        cwd = os.getcwd()
        if os.path.basename(cwd) != project_path:
            os.chdir(pathlib.Path(__file__).parent.parent / 'head_motion_prediction')

        # create df for each dataset
        from head_motion_prediction.David_MMSys_18 import Read_Dataset as david
        from head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
        from head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
        from head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
        from head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami
        names = ['David_MMSys_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'Xu_CVPR_18', 'Xu_PAMI_18']
        sizes = [1083, 300, 432, 6654, 4408]
        pkgs = [david, fan, nguyen, xucvpr, xupami][:1]
        idxs = range(len(pkgs))
        def _df_xyz(idx, n_traces=100) -> pd.DataFrame:
            # create sampled
            if len(os.listdir(pkgs[idx].OUTPUT_FOLDER)) < 2:
                pkgs[idx].create_and_store_sampled_dataset()
            dataset = pkgs[idx].load_sampled_dataset()
            # df with (dataset, user, video, times, traces)
            data = [(names[idx],
                     user,
                     video,
                     np.around(dataset[user][video][:n_traces, 0], decimals=2),
                     dataset[user][video][:n_traces, 1:]
                     ) for user in dataset.keys() for video in dataset[user].keys()]
            tmpdf = pd.DataFrame(data, columns=['dataset', 'user', 'video', 'times', 'traces'])
            # size check
            assert(tmpdf.dataset.value_counts()[names[idx]] == sizes[idx])
            return tmpdf
        self.df = pd.concat(map(_df_xyz, idxs), ignore_index=True)

        # back to cwd
        os.chdir(cwd)

    def get_trajects(self, users=list[str], videos=list[str], perc=1.0):
        assert (perc <= 1.0 and perc >= 0.0)
        df = self.df
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
        return self.df.iloc[0]['traces'][0]

    def get_one_traject(self):
        return self.df.head(1)

    def get_one_video_trajects(self):
        return self.df.loc[self.df.video == ONE_VIDEO]

    def calc_poles_prc(self):
        # calc entropy
        f_traject = lambda traces: np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)
        self.df['poles_prc'] = pd.Series(self.df['traces'].apply(f_traject))

        idxs_sort = self.df['poles_prc'].argsort()
        trajects_len = len(self.df['poles_prc'])
        idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
        idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
        threshold_medium = self.df['poles_prc'][idx_threshold_medium]
        threshold_hight = self.df['poles_prc'][idx_threshold_hight]
        f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
        self.df['poles_class'] = self.df['poles_prc'].apply(f_threshold)

    def show_poles_prc(self):
        assert('poles_prc' in self.df.columns)
        px.scatter(self.df, x='user', y='poles_prc', color='poles_class',
                   color_discrete_map=class_color_map,
                   hover_data=[self.df.index], title='trajects poles_perc', width=600).show()

    def calc_hmps(self, tileset=TileSet.default()):
        if (len(self.df) > 2000):
            print('calc_hmps can last >10m when >2000 trajectories')
        f_trace = lambda trace: tileset.request(trace)
        f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
        self.df['hmps'] = pd.Series(self.df['traces'].swifter.apply(f_traject))

    def calc_entropy(self):
        if 'hmps' not in self.df.columns:
            self.calc_hmps()

        # calc entropy
        f_entropy = lambda x: scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
        self.df['entropy'] = self.df['hmps'].swifter.apply(f_entropy, axis=1)

        # calc class
        idxs_sort = self.df['entropy'].argsort()
        trajects_len = len(self.df['entropy'])
        idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
        idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
        threshold_medium = self.df['entropy'][idx_threshold_medium]
        threshold_hight = self.df['entropy'][idx_threshold_hight]
        f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
        self.df['entropy_class'] = self.df['entropy'].apply(f_threshold)

    def show_entropy(self):
        assert('entropy' in self.df.columns)
        px.scatter(self.df, x='user', y='entropy', color='entropy_class',
                   color_discrete_map=class_color_map, hover_data=[self.df.index], title='trajects entropy', width=600).show()

    def _create_tileset_df(self, tileset: TileSetIF, df: pd.DataFrame):
        tcols = [c for c in df.columns if c.startswith('t_')]
        def f_trace(trace):
            heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
            return [heatmap, int(np.sum(heatmap)), vp_quality, area_out]

        tmpdf = df[tcols].applymap(f_trace)
        def f_col(col):
            idxs = [f"{metric}_{col.replace('t_','')}" for metric in ['hm', 'reqs', 'lost', 'qlt']]
            f_expand = lambda x: pd.Series(x, index=idxs)
            return tmpdf[col].apply(f_expand)
        return pd.concat([f_col(col) for col in tcols], axis=1)

    def calc_tileset_metrics(self, tileset_l: list[TileSetIF], df: pd.DataFrame):
        # calc tileset reqs
        self.tileset_dict = {ts.title: self._create_tileset_df(ts, df) for ts in tileset_l}
        # calc tileset metrics
        data = {'scheme': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
        for tileset in tileset_l:
            _df = self.tileset_dict[tileset.title]
            data['scheme'].append(tileset.title)
            data['avg_reqs'].append(np.nanmean(_df.loc[:, _df.columns.str.endswith('reqs')]))
            data['avg_qlt'].append(np.nanmean(_df.loc[:, _df.columns.str.endswith('qlt')]))
            data['avg_lost'].append(np.nanmean(_df.loc[:, _df.columns.str.endswith('lost')]))
            data['score'] = data['avg_qlt'][-1] / data['avg_lost'][-1]
        self.tileset_metrics_df = pd.DataFrame(data)

    def show_tileset_metrics(self):
        fig = make_subplots(rows=1, cols=4, subplot_titles=("avg_reqs", "avg_lost",
                            "avg_qlt", "score=avg_qlt/avg_lost"), shared_yaxes=True)
        y = self.tileset_metrics_df['scheme']
        trace = go.Bar(y=y, x=self.tileset_metrics_df['avg_reqs'], orientation='h', width=0.3)
        fig.add_trace(trace, row=1, col=1)
        trace = go.Bar(y=y, x=self.tileset_metrics_df['avg_lost'], orientation='h', width=0.3)
        fig.add_trace(trace, row=1, col=2)
        trace = go.Bar(y=y, x=self.tileset_metrics_df['avg_qlt'], orientation='h', width=0.3)
        fig.add_trace(trace, row=1, col=3)
        trace = go.Bar(y=y, x=self.tileset_metrics_df['score'], orientation='h', width=0.3)
        fig.add_trace(trace, row=1, col=4)
        fig.update_layout(width=1500, showlegend=False, barmode="stack",)
        fig.show()
