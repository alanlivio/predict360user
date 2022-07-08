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
from .tiles import *
from .tiles_voro import *

ONE_VIDEO = '10_Cows'
np.set_printoptions(suppress=True)


class Users360:
    dataset = None
    dataset_pickle = pathlib.Path(__file__).parent.parent / 'output/david.pickle'
    instance = None
    instance_pickle = pathlib.Path(__file__).parent.parent / 'output/singleton.pickle'

    def __init__(self, dataset={}):
        if not dataset:
            # df
            self.dataset = self.load_dataset()
            # -- https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
            data = {(f'david_{user}', video): pd.Series({
                f"t_{time_trace[0]:.2f}": time_trace[1:]  # time : trace_xyz
                for time_trace in self.dataset[user][video]})
                for user in self.dataset.keys()
                for video in self.dataset[user].keys()}
            self.df = pd.DataFrame.from_dict(data, dtype=object, orient='index')
            self.df.index.names = ['user', 'video']
            self.df.reset_index(inplace=True)

            # users_df
            self.users_id = [key for key in self.dataset.keys()]
            self.users_id.sort(key=int)
            self.users_len = len(self.users_id)
            self.users_df = pd.DataFrame()
            self.users_df['user_id'] = self.users_id

    @classmethod
    def dump(cls):
        with open(cls.instance_pickle, 'wb') as f:
            pickle.dump(cls.instance, f)

    @classmethod
    def singleton(cls):
        if cls.instance is None:
            if exists(cls.instance_pickle):
                with open(cls.instance_pickle, 'rb') as f:
                    print(f"Users360.instance from {cls.instance_pickle}")
                    cls.instance = pickle.load(f)
            else:
                cls.instance = Users360()
        return cls.instance

    def load_dataset(self):
        print("loading dataset")
        if Users360.dataset is None:
            if not exists(Users360.dataset_pickle):
                project_path = "head_motion_prediction"
                cwd = os.getcwd()
                if os.path.basename(cwd) != project_path:
                    print(f"Users360.dataset from {Users360.dataset_pickle}")
                    os.chdir(pathlib.Path(__file__).parent.parent /
                             'head_motion_prediction')
                    from head_motion_prediction.David_MMSys_18 import Read_Dataset as david
                    Users360.dataset = david.load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Users360.dataset_pickle, 'wb') as f:
                        pickle.dump(Users360.dataset, f)
            else:
                print(f"Users360.dataset from {Users360.dataset_pickle}")
                with open(Users360.dataset_pickle, 'rb') as f:
                    Users360.dataset = pickle.load(f)
        return Users360.dataset

    def calc_users_entropy(self, tiles: TilesIF = Tiles.default(), recalculate_requests=False):
        df = self.df
        cols = [c for c in df.columns if c.startswith('h_')]

        # calculate entropy
        if not 'entropy' in df or recalculate_requests:
            df.drop([c for c in df.columns if c.startswith('h_')], axis=1, inplace=True)
            for t in [c for c in df.columns if c.startswith('t_')]:
                df[t.replace('t', 'h')] = df[t].apply(lambda trace: tiles.request(trace)[0])
        def f(x):
            sum = np.sum(x, axis=0).reshape((-1))
            return scipy.stats.entropy(sum)
        df['entropy'] = df[cols].apply(f, axis=1)

        # calculate class
        p_sort = df['entropy'].argsort()
        users_len = len(df['entropy'])
        idx_threshold_medium = p_sort[int(users_len * .60)]
        idx_threshold_hight = p_sort[int(users_len * .80)]
        threshold_medium = df['entropy'][idx_threshold_medium]
        threshold_hight = df['entropy'][idx_threshold_hight]
        df['entropy_class'] = df['entropy'].apply(
            lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight'))

    def show_users_entropy(self):
        px.scatter(self.df, x='user', y='entropy', color='entropy_class',
                   symbol='entropy_class', width=600).show()

    def trajectories(self, users=list[str], videos=list[str], perc=1.0):
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

    def example_one_trace(self):
        return self.df.iloc[0]['t_0.00']

    def example_one_trajectory(self):
        return self.df.head(1)

    def example_one_video_trajectories(self):
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

    def _df_tilescheme(self, tscheme: TilesIF, df: pd.DataFrame):
        cols = [c for c in df.columns if c.startswith('t_')]
        def f_request(trace):
            heatmap, vp_quality, area_out = tscheme.request(trace, return_metrics=True)
            ret = [heatmap, int(np.sum(heatmap)), vp_quality, area_out]
            return ret
        tmpdf = df[cols].applymap(f_request)
        def f_col(col):
            idxs = [f"{col.replace('t_','')}_{metric}" for metric in ['hmp', 'reqs', 'lost', 'qlt']]
            f = lambda x: pd.Series(x, index=idxs)
            return tmpdf[col].apply(f)
        return pd.concat([f_col(col) for col in cols], axis=1)

    def calc_tileschemes_metrics(self, tilescheme_l: list[TilesIF], df):
        # req for each tscheme
        self.tilescheme_dict = {ts.title: self._df_tilescheme(ts, df) for ts in tilescheme_l}
        # metrics for plot
        data = {'scheme': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
        for tscheme in tilescheme_l:
            _df = self.tilescheme_dict[tscheme.title]
            data['scheme'].append(tscheme.title)
            data['avg_reqs'].append(np.nanmean(_df.loc[:, _df.columns.str.endswith('reqs')]))
            data['avg_qlt'].append(np.nanmean(_df.loc[:, _df.columns.str.endswith('qlt')]))
            data['avg_lost'].append(np.nanmean(_df.loc[:, _df.columns.str.endswith('lost')]))
            data['score'] = data['avg_qlt'][-1] / data['avg_lost'][-1]
        self.tilescheme_metrics_df = pd.DataFrame(data)

    def show_tileschemes_metrics(self):
        fig = make_subplots(rows=1, cols=4, subplot_titles=("avg_reqs", "avg_lost",
                            "avg_qlt", "score=avg_qlt/avg_lost"), shared_yaxes=True)
        y = self.tilescheme_metrics_df['scheme']
        trace = go.Bar(y=y, x=self.tilescheme_metrics_df['avg_reqs'], orientation='h')
        fig.add_trace(trace, row=1, col=1)
        trace = go.Bar(y=y, x=self.tilescheme_metrics_df['avg_lost'], orientation='h')
        fig.add_trace(trace, row=1, col=2)
        trace = go.Bar(y=y, x=self.tilescheme_metrics_df['avg_qlt'], orientation='h')
        fig.add_trace(trace, row=1, col=3)
        trace = go.Bar(y=y, x=self.tilescheme_metrics_df['score'], orientation='h')
        fig.add_trace(trace, row=1, col=4)
        fig.update_layout(width=1500, showlegend=False, barmode="stack")
        fig.show()
