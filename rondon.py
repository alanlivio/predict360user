from head_motion_prediction.Utils import *
import plotly.graph_objs as go
import plotly.express as px
import pickle
from os.path import exists

SAMPLE_DATASET = None
SAMPLE_DATASET_PICKLE = 'SAMPLE_DATASET.pickle'
ONE_USER = '0'
ONE_VIDEO = '10_Cows'
LAYOUT = go.Layout(width=600)
TILES_H6, TILES_V4 = 6, 4


def layout_with_title(title):
    return go.Layout(width=800, title=title)


def get_sample_dataset(load=False):
    global SAMPLE_DATASET
    if SAMPLE_DATASET is None:
        if load or not exists(SAMPLE_DATASET_PICKLE):
            import sys
            import os
            sys.path.append('head_motion_prediction')
            from head_motion_prediction.David_MMSys_18.Read_Dataset import load_sampled_dataset
            import head_motion_prediction.Utils as Utis
            project_path = "head_motion_prediction"
            cwd = os.getcwd()
            if os.path.basename(cwd) != project_path:
                print(f"-- get SAMPLE_DATASET from {project_path}")
                os.chdir(project_path)
                SAMPLE_DATASET = load_sampled_dataset()
                os.chdir(cwd)
                with open(SAMPLE_DATASET_PICKLE, 'wb') as f:
                    pickle.dump(SAMPLE_DATASET, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f"-- get SAMPLE_DATASET from {SAMPLE_DATASET_PICKLE}")
            with open(SAMPLE_DATASET_PICKLE, 'rb') as f:
                SAMPLE_DATASET = pickle.load(f)
    return SAMPLE_DATASET


SAMPLE_DATASET = get_sample_dataset()


def get_one_trace():
    return get_sample_dataset()[ONE_USER][ONE_VIDEO][0, 1:]


def get_one_trace_eulerian():
    trace_cartesian = get_one_trace()
    return cartesian_to_eulerian(trace_cartesian[0], trace_cartesian[1], trace_cartesian[2])


def get_traces_one_video_one_user():
    return get_sample_dataset()[ONE_USER][ONE_VIDEO][:, 1:]


def get_traces_one_video_all_users():
    dataset = get_sample_dataset()
    n_traces = len(dataset[ONE_USER][ONE_VIDEO][:, 1:])
    traces = np.ndarray((len(dataset.keys())*n_traces, 3))
    count = 0
    for user in dataset.keys():
        for i in dataset[user][ONE_VIDEO][:, 1:]:
            traces.itemset((count, 0), i[0])
            traces.itemset((count, 1), i[1])
            traces.itemset((count, 2), i[2])
            count += 1
    return traces


def erp_tiles_heatmap(traces):
    heatmap = []
    for i in traces:
        heatmap.append(from_position_to_tile(eulerian_in_range(
            *cartesian_to_eulerian(i[0], i[1], i[2])), TILES_H6, TILES_V4))
    fig = px.imshow(np.sum(heatmap, axis=0), labels=dict(
        x="longitude", y="latitude", color="requests"), title=f"reqs={str(np.sum(heatmap))}")
    fig.update_layout(LAYOUT)
    fig.show()
