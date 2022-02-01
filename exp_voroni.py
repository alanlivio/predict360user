# Trinity paches
import math
import numpy as np
trinity_npatchs = 14
trinity_points = np.empty((0, 3))
for i in range(0, trinity_npatchs-1):
    zi = (1 - 1.0/trinity_npatchs) * (1 - 2.0*i / (trinity_npatchs - 1))
    di = math.sqrt(1 - math.pow(zi, 2))
    alphai = i * math.pi * (3 - math.sqrt(5))
    xi = di * math.cos(alphai)
    yi = di * math.sin(alphai)
    new_point = np.array([[xi, yi, zi]])
    trinity_points = np.append(trinity_points, new_point, axis=0)
# trinity_points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]])


def vis_voroni_plotly(trajectory=None, save_html=False):
    import numpy as np
    import pandas as pd
    import plotly
    import plotly.graph_objs as go
    from scipy.spatial import SphericalVoronoi, geometric_slerp
    import numpy as np
    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(trinity_points, radius, center)
    sv.sort_vertices_of_regions()

    # -- plot the voronoi tessellation
    plotly.offline.init_notebook_mode()
    data = []

    # -- plot generator trinity_points
    # gens = go.Scatter3d(x=trinity_points[:, 0], y=trinity_points[:, 1], z=trinity_points[:, 2],
    #                     mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='Generator trinity_Points')
    # data.append(gens)

    # -- plot vortonoi vertices
    vrts = go.Scatter3d(x=sv.vertices[:, 0], y=sv.vertices[:, 1], z=sv.vertices[:, 2],
                        mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'green'}, name='Voronoi Vertices')
    data.append(vrts)

    for region in sv.regions:
        n = len(region)
        # -- get color for region
        rgb = np.random.rand(3,)
        t = np.linspace(0, 1, 100)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 1, 'color': [rgb]}, name='region edge', showlegend=False)
            data.append(edge)

    if trajectory is not None:
        if isinstance(trajectory, np.ndarray):
            trajc = go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                                 mode='lines', line={'width': 1, 'color': 'blue'}, name='trajectory', showlegend=False)
            data.append(trajc)
        else:
            for user in trajectory.keys():
                for video in trajectory[user].keys():
                    trajc = go.Scatter3d(x=trajectory[user][video][:, 0], y=trajectory[user][video][:, 1],
                                         z=trajectory[user][video][:, 2], mode='lines', 
                                         line={'width': 1, 'color': 'blue'}, 
                                         name='trajectory',showlegend=False)
                    data.append(trajc)

    # -- configure layout
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, title={
        'text': "Trinity Voroni with trajectory",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    plot_figure = go.Figure(data=data, layout=layout)

    # -- render plot

    if save_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        plotly.offline.iplot(plot_figure)


def vis_voroni_matplot(trajectory=None):
    import numpy as np
    import pandas as pd
    from head_motion_prediction.Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees, compute_orthodromic_distance, store_dict_as_csv
    from pyquaternion import Quaternion
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.spatial import SphericalVoronoi, geometric_slerp
    from mpl_toolkits.mplot3d import proj3d
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
    sv = SphericalVoronoi(trinity_points, radius, center)
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    # plot generator trinity_points
    ax.scatter(trinity_points[:, 0], trinity_points[:,
               1], trinity_points[:, 2], c='b')
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
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                trajectory[:, 2], label='parametric curve')
    plt.show()


SAMPLED_DATASET = None


def get_sample_dataset():
    import sys
    import os
    global SAMPLED_DATASET
    sys.path.append('head_motion_prediction')
    from head_motion_prediction.David_MMSys_18.Read_Dataset import load_sampled_dataset
    project_path = "head_motion_prediction"
    cwd = os.getcwd()
    if SAMPLED_DATASET is None:
        if os.path.basename(cwd) != project_path:
            print(f"running load_sampled_dataset on {project_path}")
            os.chdir(project_path)
            SAMPLED_DATASET = load_sampled_dataset()
            os.chdir(cwd)
    return SAMPLED_DATASET


def get_sample_user_trace():
    user = '0'
    video = '10_Cows'
    return get_sample_dataset()[user][video][:, 1:]
