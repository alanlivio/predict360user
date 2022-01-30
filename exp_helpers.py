# Trinity paches
import math
import numpy as np
npatchs = 14
points = np.empty((0, 3))
for i in range(0, npatchs-1):
    zi = (1 - 1.0/npatchs) * (1 - 2.0*i / (npatchs - 1))
    di = math.sqrt(1 - math.pow(zi, 2))
    alphai = i * math.pi * (3 - math.sqrt(5))
    xi = di * math.cos(alphai)
    yi = di * math.sin(alphai)
    new_point = np.array([[xi, yi, zi]])
    points = np.append(points, new_point, axis=0)
# points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]])


def vis_voroni_plotly(trajectory=None):
    import numpy as np
    import pandas as pd
    import plotly
    import plotly.graph_objs as go
    from scipy.spatial import SphericalVoronoi, geometric_slerp
    import numpy as np
    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(points, radius, center)
    sv.sort_vertices_of_regions()

    # -- plot the voronoi tessellation
    plotly.offline.init_notebook_mode()

    # -- plot generator points
    gens = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker={
                        'size': 5, 'opacity': 1.0, 'color': 'blue'}, name='Generator Points')

    # -- plot vortonoi vertices
    vrts = go.Scatter3d(x=sv.vertices[:, 0], y=sv.vertices[:, 1], z=sv.vertices[:, 2], mode='markers', marker={
                        'size': 3, 'opacity': 1.0, 'color': 'red'}, name='Voronoi Vertices')

    data = [gens, vrts]

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

    if trajectory:
        trajc = go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2], mode='markers', marker={
                             'size': 5, 'opacity': 1.0, 'color': 'yeallow'}, name='trajectory')
        data.append(trajc)

    # -- configure layout
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, title={
        'text': "Voronoi Tessellation with a random set of generators",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    plot_figure = go.Figure(data=data, layout=layout)
    # -- render plot
    plotly.offline.iplot(plot_figure)


def vis_voroni_matplot():
    import numpy as np
    import pandas as pd
    from head_motion_prediction.Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees, compute_orthodromic_distance, store_dict_as_csv
    from pyquaternion import Quaternion
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.spatial import SphericalVoronoi, geometric_slerp
    from mpl_toolkits.mplot3d import proj3d

    radius = 1
    center = np.array([0, 0, 0])
    sv = SphericalVoronoi(points, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 2000)
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    # plot generator points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
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
    ax.azim = 10
    ax.elev = 40
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_zticks([])
    fig.set_size_inches(4, 4)
    plt.show()
