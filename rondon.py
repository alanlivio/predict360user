from head_motion_prediction.Utils import *
import plotly.graph_objs as go
import plotly.express as px

SAMPLE_DATASET = None
ONE_USER = '0'
ONE_VIDEO = '10_Cows'
LAYOUT = go.Layout(width=800,
                   margin={'l': 0, 'r': 0, 'b': 0, 't': 40})
TILES_H6, TILES_V4 = 6, 4


def get_sample_dataset():
    import sys
    import os
    global SAMPLE_DATASET
    sys.path.append('head_motion_prediction')
    from head_motion_prediction.David_MMSys_18.Read_Dataset import load_sampled_dataset
    import head_motion_prediction.Utils as Utis
    project_path = "head_motion_prediction"
    cwd = os.getcwd()
    if SAMPLE_DATASET is None:
        if os.path.basename(cwd) != project_path:
            print(f"running get_sample_dataset on {project_path}")
            os.chdir(project_path)
            SAMPLE_DATASET = load_sampled_dataset()
            os.chdir(cwd)
    return SAMPLE_DATASET


SAMPLE_DATASET = get_sample_dataset()


def get_traces_one_video_one_user():
    return get_sample_dataset()[ONE_USER][ONE_VIDEO][:, 1:]


def get_traces_one_video_all_users():
    dataset = get_sample_dataset()
    traces = []
    for user in dataset.keys():
        for i in dataset[user][ONE_VIDEO][:, 1:]:
            traces.append(i)
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
