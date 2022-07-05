import head_motion_prediction
from head_motion_prediction.Utils import *
from numpy.typing import NDArray
from os.path import exists
from plotly.subplots import make_subplots
from typing import Iterable
import numpy as np
import os
import pathlib
import pickle
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats
import pandas as pd
from .tiles import *
from .tiles_voro import *

ONE_DATASET = 'david'
ONE_USER = '0'
ONE_ID = 'david_0'
ONE_VIDEO = '10_Cows'
np.set_printoptions(suppress=True)


class Dataset:
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
                    print(f"Dataset.instance from {cls.instance_pickle}")
                    cls.instance = pickle.load(f)
            else:
                cls.instance = Dataset()
        return cls.instance

    def load_dataset(self):
        print("loading dataset")
        if Dataset.dataset is None:
            if not exists(Dataset.dataset_pickle):
                project_path = "head_motion_prediction"
                cwd = os.getcwd()
                if os.path.basename(cwd) != project_path:
                    print(f"Dataset.dataset from {Dataset.dataset_pickle}")
                    os.chdir(pathlib.Path(__file__).parent.parent /
                             'head_motion_prediction')
                    from head_motion_prediction.David_MMSys_18 import Read_Dataset as david
                    Dataset.dataset = david.load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Dataset.dataset_pickle, 'wb') as f:
                        pickle.dump(Dataset.dataset, f)
            else:
                print(f"Dataset.dataset from {Dataset.dataset_pickle}")
                with open(Dataset.dataset_pickle, 'rb') as f:
                    Dataset.dataset = pickle.load(f)
        return Dataset.dataset

    def calc_users_entropy(self, tiles: TilesIF = Tiles.default(), recalculate_requests=False):
        df = Dataset.singleton().df
        if recalculate_requests:
            df.drop([c for c in df.columns if c.startswith('h_')], axis=1, inplace=True)
            for t in [c for c in df.columns if c.startswith('t_')]:
                df[t.replace('t', 'h')] = df[t].apply(lambda trace: tiles.request(trace)[0])
        cols = [c for c in df.columns if c.startswith('h_')]
        def f(x):
            sum = np.sum(x, axis=0).reshape((-1))
            return scipy.stats.entropy(sum)
        df['entropy'] = df[cols].apply(f, axis=1)

        # TODO
        p_sort = df['entropy'].argsort()
        users_len = len(df['entropy'])
        idx_threshold_medium = p_sort[int(users_len * .60)]
        idx_threshold_hight = p_sort[int(users_len * .80)]
        threshold_medium = df['entropy'][idx_threshold_medium]
        threshold_hight = df['entropy'][idx_threshold_hight]
        df['entropy_class'] = df['entropy'].apply(
            lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight'))

    def show_users_entropy(self):
        px.scatter(self.users_df, x='user_id', y='entropy', color='entropy_class',
                   symbol='entropy_class', width=600, category_orders={"user_id": self.users_id}).show()

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

    # def traces_video(self, users=[], video=ONE_VIDEO):
    #     count = 0
    #     if not users:
    #         users = self.dataset.keys()
    #     n_traces = len(self.dataset[ONE_USER][video][:, 1:])
    #     traces = np.zeros((len(users) * n_traces, 3), dtype=float)
    #     for user in users:
    #         for trace in self.dataset[user][video][:, 1:]:
    #             traces.itemset((count, 0), trace[0])
    #             traces.itemset((count, 1), trace[1])
    #             traces.itemset((count, 2), trace[2])
    #             count += 1
    #     return traces

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

    def calc_metrics_tiles(self, tiles_l: list[TilesIF], users=[], video=ONE_VIDEO, perc=1.0):
        if not users:
            users = self.dataset.keys()
        # 3 metrics for each user/tiles: avg_reqs, avg avg_lost, avg_qlt
        metrics_request = np.empty((len(tiles_l), len(users), 3))
        for indexvp, tiles in enumerate(tiles_l):
            for indexuser, user in enumerate(users):
                traces = self.trajectories(user=user, video=video, perc=perc)
                def fn(trace):
                    heatmap, vp_quality, area_out = tiles.request(
                        trace, return_metrics=True)
                    return np.hstack([np.sum(heatmap), vp_quality, area_out])
                traces_res = np.apply_along_axis(fn, 1, traces)
                metrics_request[indexvp][indexuser][0] = np.average(traces_res[:, 0])
                metrics_request[indexvp][indexuser][1] = np.average(traces_res[:, 1])
                metrics_request[indexvp][indexuser][2] = np.sum(traces_res[:, 2])
        self.tiles_df = pd.DataFrame()
        self.tiles_df['tiles'] = [ttype.title for ttype in tiles_l]
        self.tiles_df['avg_reqs'] = np.average(metrics_request[:, :, 0], axis=1)
        self.tiles_df['avg_qlt'] = np.average(metrics_request[:, :, 1], axis=1)
        self.tiles_df['avg_lost'] = np.average(metrics_request[:, :, 2], axis=1)
        self.tiles_df['score'] = self.tiles_df['avg_qlt'] / self.tiles_df['avg_lost']

    def show_metrics_tiles(self):
        fig = make_subplots(rows=1, cols=4, subplot_titles=("avg_reqs", "avg_lost",
                            "avg_qlt", "score=avg_qlt/avg_lost"), shared_yaxes=True)
        fig.add_trace(go.Bar(y=self.tiles_df['tiles'], x=self.tiles_df['avg_reqs'], orientation='h'), row=1, col=1)
        fig.add_trace(go.Bar(y=self.tiles_df['tiles'], x=self.tiles_df['avg_lost'], orientation='h'), row=1, col=2)
        fig.add_trace(go.Bar(y=self.tiles_df['tiles'], x=self.tiles_df['avg_qlt'], orientation='h'), row=1, col=3)
        fig.add_trace(go.Bar(y=self.tiles_df['tiles'], x=self.tiles_df['score'], orientation='h'), row=1, col=4)
        fig.update_layout(width=1500, showlegend=False, barmode="stack")
        fig.show()
