# %%
from cmath import cos
import os
import pathlib
from head_motion_prediction.Utils import *
from .vpextract import *
from scipy.stats import entropy
from scipy.spatial.transform import Rotation
from typing import Tuple, Iterable
from spherical_geometry import polygon
from abc import ABC
import numpy as np
from numpy.typing import NDArray
from os.path import exists
import pickle
import plotly
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

    # -- dataset funcs

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

    # -- cluster funcs

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

    def one_trace(self, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        return self.dataset[user][video][:, 1:][: 1]

    def traces_one_video_one_user(self, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        return self.dataset[user][video][:, 1:]

    def traces_one_video(self, users=None, video=ONE_VIDEO) -> NDArray:
        count = 0
        if users is None:
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

    def traces_one_user_steped(self, num, user=ONE_USER, video=ONE_VIDEO) -> NDArray:
        one_user = self.traces_one_video_one_user(user)
        step = int(len(one_user)/num)
        return one_user[:: step]

    def traces_on_poles(self, users=None, video=ONE_VIDEO) -> NDArray:
        count = 0
        if users is None:
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

    def traces_on_equator(self, users=None, video=ONE_VIDEO) -> NDArray:
        count = 0
        if users is None:
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
# %%
