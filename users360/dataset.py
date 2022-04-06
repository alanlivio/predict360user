# %%
from .vpextract import *
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

ONE_USER = '0'
ONE_VIDEO = '10_Cows'


class Dataset:
    sample_dataset = None
    sample_dataset_pickle = pathlib.Path(__file__).parent.parent/'output/david.pickle'
    _singleton = None
    _singleton_pickle = pathlib.Path(__file__).parent.parent/'output/singleton.pickle'

    def __init__(self, dataset={}):
        if not dataset:
            self.dataset = self._sample_dataset()
            self.users_id = np.array([key for key in self.dataset.keys()])
            self.users_len = len(self.users_id)

    @classmethod
    def singleton_dump(cls):
        with open(cls._singleton_pickle, 'wb') as f:
            pickle.dump(cls._singleton, f)

    @classmethod
    def singleton(cls):
        if cls._singleton is None:
            if exists(cls._singleton_pickle):
                with open(cls._singleton_pickle, 'rb') as f:
                    cls._singleton = pickle.load(f)
            else:
                cls._singleton = Dataset()
        return cls._singleton

    # -- dataset

    def _sample_dataset(self):
        if Dataset.sample_dataset is None:
            if not exists(Dataset.sample_dataset_pickle):
                project_path = "head_motion_prediction"
                cwd = os.getcwd()
                if os.path.basename(cwd) != project_path:
                    print(f"Dataset.sample_dataset from {project_path}")
                    os.chdir(pathlib.Path(__file__).parent.parent/'head_motion_prediction')
                    from .head_motion_prediction.David_MMSys_18 import Read_Dataset as david
                    Dataset.sample_dataset = david.load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Dataset.sample_dataset_pickle, 'wb') as f:
                        pickle.dump(Dataset.sample_dataset, f)
            else:
                print(f"Dataset.sample_dataset from {Dataset.sample_dataset_pickle}")
                with open(Dataset.sample_dataset_pickle, 'rb') as f:
                    Dataset.sample_dataset = pickle.load(f)
        return Dataset.sample_dataset

    # -- cluster

    def users_entropy(self, vpextract, plot_scatter=False):
        # fill users_entropy
        users_entropy = np.ndarray(self.users_len)
        for user in self.users_id:
            heatmaps = []
            for trace in self.dataset[user][ONE_VIDEO][:, 1:]:
                heatmap, _, _ = vpextract.request(trace)
                heatmaps.append(heatmap)
            sum = np.sum(heatmaps, axis=0).reshape((-1))
            # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
            users_entropy[int(user)] = scipy.stats.entropy(sum)  # type: ignore
        # define class threshold
        if plot_scatter:
            px.scatter(y=users_entropy, labels={"y": "entropy"}, width=600).show()
        p_sort = users_entropy.argsort()
        threshold_medium = int(self.users_len * .60)
        threshold_hight = int(self.users_len * .80)
        self.users_low = [str(x) for x in p_sort[:threshold_medium]]
        self.users_medium = [str(x) for x in p_sort[threshold_medium:threshold_hight]]
        self.users_hight = [str(x) for x in p_sort[threshold_hight:]]

    # -- traces

    def one_trace(self, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        return self.dataset[user][video][:, 1:][: 1]

    def traces_video_user(self, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        return self.dataset[user][video][:, 1:]

    def traces_video_user_perc(self, perc_traces: float, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        one_user = self.dataset[user][video][:, 1:]
        one_user_len = len(self.dataset[user][video][:, 1:])
        step = int(one_user_len/(one_user_len*perc_traces))
        return one_user[::step]

    def traces_video(self, users=[], video=ONE_VIDEO) -> NDArray:
        count = 0
        if not users:
            users = self.dataset.keys()
        n_traces = len(self.dataset[ONE_USER][video][:, 1:])
        traces = np.ndarray((len(users)*n_traces, 3))
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
        traces = np.ndarray((len(users)*n_traces, 3))
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
        traces = np.ndarray((len(users)*n_traces, 3))
        for user in users:
            for trace in self.dataset[user][video][:, 1:]:
                if abs(trace[2]) < 0.7:  # z-axis
                    traces.itemset((count, 0), trace[0])
                    traces.itemset((count, 1), trace[1])
                    traces.itemset((count, 2), trace[2])
                    count += 1
        return traces[:count]

    # -- vpextract

    def metrics_vpextract_video(self, vpextract_l: Iterable[VPExtract], users=[], video=ONE_VIDEO, perc_traces=1.0, plot_traces=False, plot_heatmaps=False):
        if not users:
            users = self.dataset.keys()
        # fig_reqs = go.Figure(layout=LAYOUT)
        # fig_areas = go.Figure(layout=LAYOUT)
        # fig_quality = go.Figure(layout=LAYOUT)
        vpextract_reqs = []
        vpextract_reqs_use = []
        vpextract_vp_quality = []
        for vpextract in vpextract_l:
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
                for trace in self.traces_video_user_perc(user=user, video=video, perc_traces=perc_traces):
                    try:
                        heatmap, vp_quality, areas_in = vpextract.request(trace, return_metrics=True)
                    except:
                        continue
                    user_reqs.append(np.sum(heatmap))
                    user_reqs_use.append(areas_in)
                    user_quality.append(vp_quality)
                    # user_heatmaps.append(heatmap)
                    # user_reqs_use.append(areas_in)
                users_reqs.append(np.average(user_reqs))
                users_reqs_use.append(np.average(user_reqs_use))
                users_quality.append(np.average(user_quality))
            if not len(users_reqs):
                continue
            # # line reqs
            # fig_reqs.add_trace(go.Scatter(y=traces_reqs, mode='lines', name=f"{vpextract.title}"))
            # # line areas
            # fig_areas.add_trace(go.Scatter(y=traces_areas_avg, mode='lines', name=f"{vpextract.title}"))
            # # line quality
            # fig_quality.add_trace(go.Scatter(y=traces_vp_quality, mode='lines', name=f"{vpextract.title}"))
            # # heatmap
            # if(plot_heatmaps and len(traces_heatmaps)):
            #     fig_heatmap = px.imshow(
            #         np.sum(traces_heatmaps, axis=0).reshape(vpextract.shape),
            #         title=f"{vpextract.title_with_sum_heatmaps(traces_heatmaps)}",
            #         labels=dict(x="longitude", y="latitude", color="VP_Extracts"))
            #     fig_heatmap.update_layout(LAYOUT)
            #     fig_heatmap.show()
            # sum
            vpextract_reqs.append(np.average(users_reqs))
            vpextract_reqs_use.append(np.average(users_reqs_use))
            vpextract_vp_quality.append(np.average(users_quality))
        print(vpextract_reqs)
        print(vpextract_reqs_use)
        print(vpextract_vp_quality)
        # line fig reqs areas
        # if(plot_traces):
        #     fig_reqs.update_layout(xaxis_title="user trace", title="req_tiles").show()
        #     fig_areas.update_layout(xaxis_title="user trace",
        #                             title="avg req_tiles view_ratio").show()
        #     fig_quality.update_layout(xaxis_title="user trace", title="avg quality ratio").show()

        # bar fig vpextract_reqs vpextract_reqs_use
        vpextract_names = [str(vpextract.title) for vpextract in vpextract_l]
        fig_bar = make_subplots(rows=1, cols=4,  subplot_titles=(
            "avg tiles_reqs", "avg reqs_use", "avg vp_quality", "score=vp_quality/(tiles_reqs*(1-reqs_use))"), shared_yaxes=True)
        fig_bar.add_trace(go.Bar(y=vpextract_names, x=vpextract_reqs, orientation='h'), row=1, col=1)
        fig_bar.add_trace(go.Bar(y=vpextract_names, x=vpextract_reqs_use, orientation='h'), row=1, col=2)
        fig_bar.add_trace(go.Bar(y=vpextract_names, x=vpextract_vp_quality, orientation='h'), row=1, col=3)
        vpextract_score = [vpextract_vp_quality[i] / (vpextract_reqs[i] * (1 - vpextract_reqs_use[i]))
                           for i, _ in enumerate(vpextract_reqs)]
        fig_bar.add_trace(go.Bar(y=vpextract_names, x=vpextract_score, orientation='h'), row=1, col=4)
        fig_bar.update_layout(width=1500, showlegend=False, title_text="metrics_vpextract_video")
        fig_bar.update_layout(barmode="stack")
        fig_bar.show()
# %%
