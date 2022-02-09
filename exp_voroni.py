import math
import numpy as np
import plotly
import plotly.graph_objs as go
from scipy.spatial import SphericalVoronoi, geometric_slerp


# Trinity paches
TRINITY_NPATCHS = 14
TRINITY_POINTS = np.empty((0, 3))
for i in range(0, TRINITY_NPATCHS-1):
    zi = (1 - 1.0/TRINITY_NPATCHS) * (1 - 2.0*i / (TRINITY_NPATCHS - 1))
    di = math.sqrt(1 - math.pow(zi, 2))
    alphai = i * math.pi * (3 - math.sqrt(5))
    xi = di * math.cos(alphai)
    yi = di * math.sin(alphai)
    new_point = np.array([[xi, yi, zi]])
    TRINITY_POINTS = np.append(TRINITY_POINTS, new_point, axis=0)

# head_motion_predction sample dataset
SAMPLE_DATASET = None
ONE_USER = '0'
ONE_VIDEO = '10_Cows'


def get_sample_dataset():
    import sys
    import os
    global SAMPLE_DATASET
    sys.path.append('head_motion_prediction')
    from head_motion_prediction.David_MMSys_18.Read_Dataset import load_sampled_dataset
    project_path = "head_motion_prediction"
    cwd = os.getcwd()
    if SAMPLE_DATASET is None:
        if os.path.basename(cwd) != project_path:
            print(f"running get_sample_dataset on {project_path}")
            os.chdir(project_path)
            SAMPLE_DATASET = load_sampled_dataset()
            os.chdir(cwd)
    return SAMPLE_DATASET


def get_sample_one_user_one_video():
    return get_sample_dataset()[ONE_USER][ONE_VIDEO][:, 1:]


# plot functions

def plot_voroni_one_user_one_video_with_matplot():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) *
                    np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(TRINITY_POINTS, radius, center)
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    # plot generator TRINITY_POINTS
    ax.scatter(TRINITY_POINTS[:, 0], TRINITY_POINTS[:,
               1], TRINITY_POINTS[:, 2], c='b')
    # plot Voronoi vertices
    ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
               c='g')
    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            ax.plot(result[..., 0],
                    result[..., 1],
                    result[..., 2],
                    c='k')

    trajectory = get_sample_dataset()[ONE_USER][ONE_VIDEO][:, 1:]
    ax.plot(trajectory[:, 0], trajectory[:, 1],
            trajectory[:, 2], label='parametric curve')
    plt.show()


def plotly_data_voroni():
    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(TRINITY_POINTS, radius, center)
    sv.sort_vertices_of_regions()
    data = []

    # -- add generator points
    # gens = go.Scatter3d(x=TRINITY_POINTS[:, 0], y=TRINITY_POINTS[:, 1], z=TRINITY_POINTS[:, 2], mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='Generator TRINITY_POINTS')
    # data.append(gens)

    # -- add vortonoi vertices
    vrts = go.Scatter3d(x=sv.vertices[:, 0], y=sv.vertices[:, 1], z=sv.vertices[:, 2],
                        mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'green'}, name='Voronoi Vertices')
    data.append(vrts)

    # -- add vortonoi edges
    for region in sv.regions:
        n = len(region)
        rgb = np.random.rand(3,)
        t = np.linspace(0, 1, 100)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 1, 'color': [rgb]}, name='region edge', showlegend=False)
            data.append(edge)
    return data


def plotly_data_append_user_traces(data, dataset, user, video):
    trajc = go.Scatter3d(x=dataset[user][video][:, 1:][:, 0],
                         y=dataset[user][video][:, 1:][:, 1],
                         z=dataset[user][video][:, 1:][:, 2],
                         mode='lines',
                         line={'width': 1, 'color': 'blue'},
                         name='trajectory', showlegend=False)
    data.append(trajc)


def plot_voroni_one_video_one_user(to_html=False):
    data = plotly_data_voroni()
    dataset = get_sample_dataset()
    plotly_data_append_user_traces(data, dataset, ONE_USER, ONE_VIDEO)
    if to_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        go.Figure(data=data, layout=layout, layout_width=800).show()


def plot_voroni_one_video_one_all_user(to_html=False):
    data = plotly_data_voroni()
    dataset = get_sample_dataset()
    for user in dataset.keys():
        plotly_data_append_user_traces(data, dataset, user, ONE_VIDEO)
    if to_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        go.Figure(data=data, layout=layout, layout_width=800).show()
