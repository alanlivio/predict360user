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

ONE_USER = '0'
ONE_VIDEO = '10_Cows'
np.set_printoptions(suppress=True)

class Dataset:
    dataset = None
    dataset_pickle = pathlib.Path(__file__).parent.parent/'output/david.pickle'
    instance = None
    instance_pickle = pathlib.Path(__file__).parent.parent/'output/singleton.pickle'

    def __init__(self, dataset={}):
        if not dataset:
            self.dataset = self.load_dataset()
            # self.users_id = np.array([key for key in self.dataset.keys()])
            self.users_id = [key for key in self.dataset.keys()]
            self.users_id.sort(key=int)
            self.users_len = len(self.users_id)
            self.users_df = pd.DataFrame(index=self.users_id)
            self.users_df['users_id'] = self.users_id

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
                    from .head_motion_prediction.David_MMSys_18 import Read_Dataset as david
                    Dataset.dataset = david.load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Dataset.dataset_pickle, 'wb') as f:
                        pickle.dump(Dataset.dataset, f)
            else:
                print(f"Dataset.dataset from {Dataset.dataset_pickle}")
                with open(Dataset.dataset_pickle, 'rb') as f:
                    Dataset.dataset = pickle.load(f)
        return Dataset.dataset

    def calc_users_entropy(self, tiles: TilesIF):
        # fill entropy
        entropy = np.zeros(self.users_len, dtype=float)
        for user in self.users_id:
            heatmaps = []
            for trace in self.dataset[user][ONE_VIDEO][:, 1:]:
                heatmap, _, _ = tiles.request(trace)
                heatmaps.append(heatmap)
            sum = np.sum(heatmaps, axis=0).reshape((-1))
            # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
            entropy[int(user)] = scipy.stats.entropy(sum)

        # define class threshold
        p_sort = entropy.argsort()
        threshold_medium = int(self.users_len * .60)
        threshold_hight = int(self.users_len * .80)
        self.users_low = [str(x) for x in p_sort[:threshold_medium]]
        self.users_medium = [str(x) for x in p_sort[threshold_medium:threshold_hight]]
        self.users_hight = [str(x) for x in p_sort[threshold_hight:]]

        # save on df
        self.users_df['entropy'] = entropy  # this consider an orted index
        self.users_df.loc[self.users_df['users_id'].isin(self.users_low), 'entropy_class'] = 'low'
        self.users_df.loc[self.users_df['users_id'].isin(self.users_medium), 'entropy_class'] = 'medium'
        self.users_df.loc[self.users_df['users_id'].isin(self.users_hight), 'entropy_class'] = 'hight'

    def plot_users(self):
        px.scatter(self.users_df, x='users_id', y='entropy', color='entropy_class',
                   symbol='entropy_class', width=600, category_orders={"users_id": self.users_id}).show()

    def one_trace(self, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        return self.dataset[user][video][:, 1:][0]

    def traces_video_user(self, perc_traces=1.0, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        assert (perc_traces <= 1.0 and perc_traces >= 0.0)
        if perc_traces == 1.0:
            return self.dataset[user][video][:, 1:]
        else:
            on_user = self.dataset[user][video][:, 1:]
            one_user_len = len(on_user)
            step = int(one_user_len/(one_user_len*perc_traces))
            return on_user[::step]

    def traces_video(self, users=[], video=ONE_VIDEO) -> NDArray:
        count = 0
        if not users:
            users = self.dataset.keys()
        n_traces = len(self.dataset[ONE_USER][video][:, 1:])
        traces = np.zeros((len(users)*n_traces, 3), dtype=float)
        for user in users:
            for trace in self.dataset[user][video][:, 1:]:
                traces.itemset((count, 0), trace[0])
                traces.itemset((count, 1), trace[1])
                traces.itemset((count, 2), trace[2])
                count += 1
        return traces

    def traces_video_poles(self, users=[], video=ONE_VIDEO) -> NDArray:
        count = 0
        if not users:
            users = self.dataset.keys()
        n_traces = len(self.dataset[ONE_USER][video][:, 1:])
        traces = np.zeros((len(users)*n_traces, 3), dtype=float)
        for user in users:
            for trace in self.dataset[user][video][:, 1:]:
                if abs(trace[2]) > 0.7:  # z-axis
                    traces.itemset((count, 0), trace[0])
                    traces.itemset((count, 1), trace[1])
                    traces.itemset((count, 2), trace[2])
                    count += 1
        return traces[:count]

    def traces_video_equator(self, users=[], video=ONE_VIDEO) -> NDArray:
        count = 0
        if not users:
            users = self.dataset.keys()
        n_traces = len(self.dataset[ONE_USER][video][:, 1:])
        traces = np.zeros((len(users)*n_traces, 3), dtype=float)
        for user in users:
            for trace in self.dataset[user][video][:, 1:]:
                if abs(trace[2]) < 0.7:  # z-axis
                    traces.itemset((count, 0), trace[0])
                    traces.itemset((count, 1), trace[1])
                    traces.itemset((count, 2), trace[2])
                    count += 1
        return traces[:count]

    def metrics_tiles_video(self, tiles_l, users=[], video=ONE_VIDEO, perc_traces=1.0, show_plot=True):
        if not users:
            users = self.dataset.keys()
        # 3 metrics for each user/tiles = reqs, avg area, avg quality
        metrics_request = np.empty((len(tiles_l), len(users), 3))
        for indexvp, tiles in enumerate(tiles_l):
            for indexuser, user in enumerate(users):
                traces = self.traces_video_user(
                    user=user, video=video, perc_traces=perc_traces)

                def fn(trace):
                    heatmap, vp_quality, area_out = tiles.request(
                        trace, return_metrics=True)
                    return np.hstack([np.sum(heatmap), vp_quality, area_out])
                traces_res = np.apply_along_axis(fn, 1, traces)

                metrics_request[indexvp][indexuser][0] = np.average(
                    traces_res[:, 0])
                metrics_request[indexvp][indexuser][1] = np.average(
                    traces_res[:, 1])
                metrics_request[indexvp][indexuser][2] = np.sum(
                    traces_res[:, 2])
        trace_len = len(self.traces_video_user(
            user=ONE_USER, video=ONE_VIDEO, perc_traces=perc_traces))
        print(
            f"Dataset.metrics_request.shape={metrics_request.shape} traces {trace_len}")
        tiles_reqs = np.average(metrics_request[:, :, 0], axis=1)
        tiles_vp_quality = np.average(metrics_request[:, :, 1], axis=1)
        tiles_area_out = np.average(metrics_request[:, :, 2], axis=1)

        # figs from metrics
        if (show_plot):
            tiles_names = [str(tiles.title) for tiles in tiles_l]
            fig_bar = make_subplots(rows=1, cols=4,  subplot_titles=("n_reqs_avg_user", "area_out_avg_user",
                                    "vp_quality_avg_user", "score=vp_quality_avg_user/area_out_avg_user"), shared_yaxes=True)
            fig_bar.add_trace(
                go.Bar(y=tiles_names, x=tiles_reqs, orientation='h'), row=1, col=1)
            fig_bar.add_trace(
                go.Bar(y=tiles_names, x=tiles_area_out, orientation='h'), row=1, col=2)
            fig_bar.add_trace(
                go.Bar(y=tiles_names, x=tiles_vp_quality, orientation='h'), row=1, col=3)
            tiles_score = [tiles_vp_quality[i] / tiles_area_out[i]
                           for i, _ in enumerate(tiles_reqs)]
            fig_bar.add_trace(
                go.Bar(y=tiles_names, x=tiles_score, orientation='h'), row=1, col=4)
            fig_bar.update_layout(
                width=1500, showlegend=False, barmode="stack", title_text="metrics_request")
            fig_bar.show()

    def metrics_tiles_video_old(self, tiles_l: Iterable[TilesIF], users=[], video=ONE_VIDEO, perc_traces=1.0):
        # LAYOUT = go.Layout(width=600)
        if not users:
            users = self.dataset.keys()
        # fig_reqs = go.Figure(layout=LAYOUT)
        # fig_areas = go.Figure(layout=LAYOUT)
        # fig_quality = go.Figure(layout=LAYOUT)
        tiles_reqs = []
        tiles_area_out = []
        tiles_vp_quality = []
        for tiles in tiles_l:
            # call func per trace
            users_reqs = []
            # users_reqs_use = []
            users_reqs_use = []
            # users_heatmaps = []
            users_quality = []
            for user in users:
                user_reqs = []
                user_reqs_use = []
                user_quality = []
                # user_reqs_use = []
                # user_heatmaps = []
                for trace in self.traces_video_user(user=user, video=video, perc_traces=perc_traces):
                    heatmap, vp_quality, areas_in = tiles.request(
                        trace, return_metrics=True)
                    user_reqs.append(np.sum(heatmap))
                    user_reqs_use.append(areas_in/np.sum(heatmap))
                    user_quality.append(vp_quality)
                    # user_heatmaps.append(heatmap)
                    # user_reqs_use.append(areas_in)
                users_reqs.append(np.average(user_reqs))
                users_reqs_use.append(np.average(user_reqs_use))
                users_quality.append(np.average(user_quality))
            if not len(users_reqs):
                continue
            # # line reqs
            # fig_reqs.add_trace(go.Scatter(y=traces_reqs, mode='lines', name=f"{tiles.title}"))
            # # line areas
            # fig_areas.add_trace(go.Scatter(y=traces_areas_avg, mode='lines', name=f"{tiles.title}"))
            # # line quality
            # fig_quality.add_trace(go.Scatter(y=traces_vp_quality, mode='lines', name=f"{tiles.title}"))
            # # heatmap
            # if(plot_heatmaps and len(traces_heatmaps)):
            #     fig_heatmap = px.imshow(
            #         np.sum(traces_heatmaps, axis=0).reshape(tiles.shape),
            #         title=f"{tiles.title_with_sum_heatmaps(traces_heatmaps)}",
            #         labels=dict(x="longitude", y="latitude", color="VP_Extracts"))
            #     fig_heatmap.update_layout(LAYOUT)
            #     fig_heatmap.show()
            # sum
            tiles_reqs.append(np.average(users_reqs))
            tiles_area_out.append(np.average(users_reqs_use))
            tiles_vp_quality.append(np.average(users_quality))
        # print(tiles_reqs)
        # print(tiles_area_out)
        # print(tiles_vp_quality)
        # line fig reqs areas
        # if(plot_traces):
        #     fig_reqs.update_layout(xaxis_title="user trace", title="req_tiles").show()
        #     fig_areas.update_layout(xaxis_title="user trace",
        #                             title="avg req_tiles view_ratio").show()
        #     fig_quality.update_layout(xaxis_title="user trace", title="avg quality ratio").show()

        # bar fig tiles_reqs tiles_area_out
        tiles_names = [str(tiles.title) for tiles in tiles_l]
        fig_bar = make_subplots(rows=1, cols=4,  subplot_titles=(
            "avg tiles_reqs", "avg reqs_use", "avg vp_quality", "score=vp_quality/(tiles_reqs*(1-reqs_use))"), shared_yaxes=True)
        fig_bar.add_trace(go.Bar(y=tiles_names, x=tiles_reqs,
                          orientation='h'), row=1, col=1)
        fig_bar.add_trace(
            go.Bar(y=tiles_names, x=tiles_area_out, orientation='h'), row=1, col=2)
        fig_bar.add_trace(
            go.Bar(y=tiles_names, x=tiles_vp_quality, orientation='h'), row=1, col=3)
        tiles_score = [tiles_vp_quality[i] / tiles_area_out[i]
                       for i, _ in enumerate(tiles_reqs)]
        fig_bar.add_trace(go.Bar(y=tiles_names, x=tiles_score,
                          orientation='h'), row=1, col=4)
        fig_bar.update_layout(width=1500, showlegend=False,
                              title_text="metrics_tiles_video")
        fig_bar.update_layout(barmode="stack")
        fig_bar.show()
