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

ONE_VIDEO = '10_Cows'
np.set_printoptions(suppress=True)


class Trajectories:
    dataset = None
    dataset_pickle = pathlib.Path(__file__).parent.parent / 'output/david.pickle'
    instance = None
    instance_pickle = pathlib.Path(__file__).parent.parent / 'output/singleton.pickle'

    def __init__(self, dataset={}):
        if not dataset:
            # df
            self.dataset = self._load_dataset()
            # -- https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
            data = {(f'david_{user}', video): pd.Series({
                f"t_{time_trace[0]:.2f}": time_trace[1:]  # time : trace_xyz
                for time_trace in self.dataset[user][video]})
                for user in self.dataset.keys()
                for video in self.dataset[user].keys()}
            self.df = pd.DataFrame.from_dict(data, dtype=object, orient='index')
            self.df.index.names = ['user', 'video']
            self.df.reset_index(inplace=True)

    @classmethod
    def dump(cls):
        with open(cls.instance_pickle, 'wb') as f:
            pickle.dump(cls.instance, f)

    @classmethod
    def singleton(cls):
        if cls.instance is None:
            if exists(cls.instance_pickle):
                with open(cls.instance_pickle, 'rb') as f:
                    print(f"Trajectories from {cls.instance_pickle}")
                    cls.instance = pickle.load(f)
            else:
                cls.instance = Trajectories()
        return cls.instance

    def _load_dataset(self):
        if Trajectories.dataset is None:
            if not exists(Trajectories.dataset_pickle):
                project_path = "head_motion_prediction"
                cwd = os.getcwd()
                if os.path.basename(cwd) != project_path:
                    print(f"Trajectories loads {Trajectories.dataset_pickle}")
                    os.chdir(pathlib.Path(__file__).parent.parent /
                             'head_motion_prediction')
                    from head_motion_prediction.David_MMSys_18 import Read_Dataset as david
                    Trajectories.dataset = david.load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Trajectories.dataset_pickle, 'wb') as f:
                        pickle.dump(Trajectories.dataset, f)
            else:
                print(f"Trajectories loads {Trajectories.dataset_pickle}")
                with open(Trajectories.dataset_pickle, 'rb') as f:
                    Trajectories.dataset = pickle.load(f)
        return Trajectories.dataset

    def show_entropy(self):
        px.scatter(self.df, x='user', y='entropy', color='entropy_class',
                   symbol='entropy_class', width=600).show()

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
        return self.df.iloc[0]['t_0.00']

    def get_one_traject(self):
        return self.df.head(1)

    def get_one_video_trajects(self):
        return self.df.loc[self.df.video == ONE_VIDEO]

    # def traces_video_poles(self, users=[], video=ONE_VIDEO):
    #     count = 0
    #     if not users:
    #         users = self.dataset.keys()
    #     n_traces = len(self.dataset[ONE_USER][video][:, 1:])
    #     traces = np.zeros((len(users) * n_traces, 3), dtype=float)
    #     for user in users:
    #         for trace in self.dataset[user][video][:, 1:]:
    #             if abs(trace[2]) > 0.7:  # z-axis
    #                 traces.itemset((count, 0), trace[0])
    #                 traces.itemset((count, 1), trace[1])
    #                 traces.itemset((count, 2), trace[2])
    #                 count += 1
    #     return traces[:count]

    # def traces_video_equator(self, users=[], video=ONE_VIDEO):
    #     count = 0
    #     if not users:
    #         users = self.dataset.keys()
    #     n_traces = len(self.dataset[ONE_USER][video][:, 1:])
    #     traces = np.zeros((len(users) * n_traces, 3), dtype=float)
    #     for user in users:
    #         for trace in self.dataset[user][video][:, 1:]:
    #             if abs(trace[2]) < 0.7:  # z-axis
    #                 traces.itemset((count, 0), trace[0])
    #                 traces.itemset((count, 1), trace[1])
    #                 traces.itemset((count, 2), trace[2])
    #                 count += 1
    #     return traces[:count]

    def calc_entropy(self, tileset=TileSet.default()):
        tcols = [col for col in self.df.columns if col.startswith('t_')]
        hcols = [col.replace('t_', 'hm_') for col in tcols]

        # calc heatmaps
        f_request_hmp = lambda trace: tileset.request(trace)[0]
        dftmp = self.df[tcols].applymap(f_request_hmp)
        dftmp.columns = hcols
        self.df = pd.concat([self.df, dftmp], axis=1)

        # calc entropy
        f_entropy = lambda x: scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
        self.df['entropy'] = self.df[hcols].apply(f_entropy, axis=1)

        # calc class
        p_sort = self.df['entropy'].argsort()
        users_len = len(self.df['entropy'])
        idx_threshold_medium = p_sort[int(users_len * .60)]
        idx_threshold_hight = p_sort[int(users_len * .80)]
        threshold_medium = self.df['entropy'][idx_threshold_medium]
        threshold_hight = self.df['entropy'][idx_threshold_hight]
        f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
        self.df['entropy_class'] = self.df['entropy'].apply(f_threshold)

    def _create_tileset_df(self, tileset: TileSetIF, df: pd.DataFrame):
        tcols = [c for c in df.columns if c.startswith('t_')]
        def f_request(trace):
            heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
            return [heatmap, int(np.sum(heatmap)), vp_quality, area_out]
        tmpdf = df[tcols].applymap(f_request)
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
