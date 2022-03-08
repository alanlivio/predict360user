import math
import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.express as px
from scipy.spatial import SphericalVoronoi, geometric_slerp
from head_motion_prediction.Utils import *
from VRClient.src.help_functions import *
from spherical_geometry import polygon
from plotly.subplots import make_subplots

# voroni sphere


def create_voroni_sphere(npatchs) -> tuple[np.ndarray, SphericalVoronoi]:
    points = np.empty((0, 3))
    for i in range(0, npatchs):
        zi = (1 - 1.0/npatchs) * (1 - 2.0*i / (npatchs - 1))
        di = math.sqrt(1 - math.pow(zi, 2))
        alphai = i * math.pi * (3 - math.sqrt(5))
        xi = di * math.cos(alphai)
        yi = di * math.sin(alphai)
        new_point = np.array([[xi, yi, zi]])
        points = np.append(points, new_point, axis=0)
    sv = SphericalVoronoi(points, 1, np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()
    return points, sv


TRINITY_NPATCHS = 14
VORONOI_CPOINTS_14P, VORONOI_SPHERE_14P = create_voroni_sphere(TRINITY_NPATCHS)
VORONOI_CPOINTS_24P, VORONOI_SPHERE_24P = create_voroni_sphere(24)

# dataset
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


def get_traces_one_video_one_user():
    return get_sample_dataset()[ONE_USER][ONE_VIDEO][:, 1:]


def get_traces_one_video_all_users():
    dataset = get_sample_dataset()
    traces = []
    for user in dataset.keys():
        for i in dataset[user][ONE_VIDEO][:, 1:]:
            traces.append(i)
    return traces


# plot funcs
TILES_WIDTH, TILES_HEIGHT = 6, 4
LAYOUT = go.Layout(width=800,
                   margin={'l': 0, 'r': 0, 'b': 0, 't': 40})


def plot_voro14_one_user_one_video_matplot():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0: 2 * np.pi: 20j, 0: np.pi: 10j]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) *
                    np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    t_vals = np.linspace(0, 1, 2000)
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    # plot generator VORONOI_CPOINTS_14P
    ax.scatter(VORONOI_CPOINTS_14P[:, 0], VORONOI_CPOINTS_14P[:,
               1], VORONOI_CPOINTS_14P[:, 2], c='b')
    # plot Voronoi vertices
    ax.scatter(VORONOI_SPHERE_14P.vertices[:, 0], VORONOI_SPHERE_14P.vertices[:, 1], VORONOI_SPHERE_14P.vertices[:, 2],
               c='g')
    # indicate Voronoi regions (as Euclidean polygons)
    for region in VORONOI_SPHERE_14P.regions:
        n = len(region)
        for i in range(n):
            start = VORONOI_SPHERE_14P.vertices[region][i]
            end = VORONOI_SPHERE_14P.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            ax.plot(result[..., 0],
                    result[..., 1],
                    result[..., 2],
                    c='k')

    trajectory = get_sample_dataset()[ONE_USER][ONE_VIDEO][:, 1:]
    ax.plot(trajectory[:, 0], trajectory[:, 1],
            trajectory[:, 2], label='parametric curve')
    plt.show()


def plotly_data_voro14():
    data = []

    # -- add generator points
    # gens = go.Scatter3d(x=VORONOI_CPOINTS_14P[:, 0], y=VORONOI_CPOINTS_14P[:, 1], z=VORONOI_CPOINTS_14P[:, 2], mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='Generator VORONOI_CPOINTS_14P')
    # data.append(gens)

    # -- add vortonoi vertices
    vrts = go.Scatter3d(x=VORONOI_SPHERE_14P.vertices[:, 0], y=VORONOI_SPHERE_14P.vertices[:, 1], z=VORONOI_SPHERE_14P.vertices[:, 2],
                        mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'green'}, name='Voronoi Vertices')
    data.append(vrts)

    # -- add vortonoi edges
    for region in VORONOI_SPHERE_14P.regions:
        n = len(region)
        rgb = np.random.rand(3,)
        t = np.linspace(0, 1, 100)
        for i in range(n):
            start = VORONOI_SPHERE_14P.vertices[region][i]
            end = VORONOI_SPHERE_14P.vertices[region][(i + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 1, 'color': [rgb]}, name='region edge', showlegend=False)
            data.append(edge)
    return data


def plotly_scatter3d_add_user_traces(data, dataset, user, video):
    trajc = go.Scatter3d(x=dataset[user][video][:, 1:][:, 0],
                         y=dataset[user][video][:, 1:][:, 1],
                         z=dataset[user][video][:, 1:][:, 2],
                         mode='lines',
                         line={'width': 1, 'color': 'blue'},
                         name='trajectory', showlegend=False)
    data.append(trajc)


def plot_rec_tiles_heatmap(traces):
    heatmap = []
    for i in traces:
        heatmap.append(from_position_to_tile(eulerian_in_range(
            *cartesian_to_eulerian(i[0], i[1], i[2])), TILES_WIDTH, TILES_HEIGHT))
    fig = px.imshow(np.sum(heatmap, axis=0), labels=dict(
        x="longitude", y="latitude", color="requests"), title=f"reqs={str(np.sum(heatmap))}")
    fig.update_layout(LAYOUT)
    fig.show()


def plot_voro14_one_video_one_user(to_html=False):
    data = plotly_data_voro14()
    dataset = get_sample_dataset()
    plotly_scatter3d_add_user_traces(data, dataset, ONE_USER, ONE_VIDEO)
    if to_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        go.Figure(data=data, layout=LAYOUT).show()


def plot_voro14_one_video_all_users(to_html=False):
    data = plotly_data_voro14()
    dataset = get_sample_dataset()
    for user in dataset.keys():
        plotly_scatter3d_add_user_traces(data, dataset, user, ONE_VIDEO)
    if to_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        go.Figure(data=data, layout=LAYOUT).show()


def plot_reqs_per_func(traces, func_list, plot_heatmaps=False):
    fig_reqs = go.Figure(layout=LAYOUT)
    fig_areas = go.Figure(layout=LAYOUT)
    fig_sum = make_subplots(rows=1, cols=2,  subplot_titles=("reqs sum", "areas sum"))

    reqs_all = []
    areas_all = []
    for func in func_list:
        reqs = []
        areas = []
        heatmaps = []
        # call func per trace
        for t in traces:
            reqs_in, areas_in, heatmap_in = func(*cartesian_to_eulerian(t[0], t[1], t[2]))
            reqs.append(reqs_in)
            areas.append(np.average(areas_in))
            if heatmap_in is not None:
                heatmaps.append(heatmap_in)
        # line reqs
        fig_reqs.add_trace(go.Scatter(y=reqs, mode='lines', name=f"reqs {func.__name__}"))
        # line areas
        fig_areas.add_trace(go.Scatter(y=areas, mode='lines', name=f"areas {func.__name__}"))
        # heatmap
        if(plot_heatmaps and len(heatmaps)):
            fig_heatmap = px.imshow(np.sum(heatmaps, axis=0), title=f"heatmap {func.__name__}",
                                    labels=dict(x="longitude", y="latitude", color="requests"))
            fig_heatmap.update_layout(LAYOUT)
            fig_heatmap.show()
        # sum
        reqs_all.append(np.sum(reqs))
        areas_all.append(np.sum(areas))

    # line fig reqs areas
    fig_reqs.show()
    fig_areas.show()

    # bar fig reqs_all areas_all
    funcs_names = [str(func.__name__) for func in func_list]
    fig_sum.add_trace(go.Bar(x=funcs_names, y=reqs_all), row=1, col=1)
    fig_sum.add_trace(go.Bar(x=funcs_names, y=areas_all), row=1, col=2)
    fig_sum.update_layout(width=800, showlegend=False)
    fig_sum.show()

# tiles funcs


def polygon_fov_cartesian(phi_vp, theta_vp):
    # https://daglar-cizmeci.com/how-does-virtual-reality-work/
    margin_lateral = np.deg2rad(90/2)
    margin_ab = np.deg2rad(110/2)
    polygon_fov = np.array([
        eulerian_to_cartesian(phi_vp-margin_ab, theta_vp+margin_lateral),
        eulerian_to_cartesian(phi_vp+margin_ab, theta_vp+margin_lateral),
        eulerian_to_cartesian(phi_vp+margin_ab, theta_vp-margin_lateral),
        eulerian_to_cartesian(phi_vp-margin_ab, theta_vp-margin_lateral)])
    return polygon_fov


def tiles_rectan_fov_110radius_cover_center(phi_vp, theta_vp):
    t_hor, t_vert = TILES_WIDTH, TILES_HEIGHT
    vp_110d = 110
    vp_110_rad = vp_110d * np.pi / 180
    d_hor = 2 * np.pi / t_hor
    d_vert = np.pi / t_vert
    projection = np.ndarray((t_vert, t_hor))
    view_areas = []
    for i in range(t_vert):
        for j in range(t_hor):
            # reqs, heatmap
            phi_c = d_hor * (j + 0.5)
            theta_c = d_vert * (i + 0.5)
            dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
            projection[i][j] = 1 if dist <= vp_110_rad / 2 else 0
            # # area
            # voroni_patch_polygon = polygon.SphericalPolygon(VORONOI_SPHERE_24P.vertices[region])
            # fov_polygon = polygon.SphericalPolygon(fov_ar_cartesian(phi_vp, theta_vp))
            # view_area = voroni_patch_polygon.overlap(fov_polygon)
            # if view_area > 0:
            #     view_areas.append(view_area)
    reqs = np.sum(projection)
    heatmap = projection
    return reqs, None, heatmap


def tiles_voro24_fov_110radius_cover_center(phi_vp, theta_vp):
    t_hor, t_vert = TILES_WIDTH, TILES_HEIGHT
    vp_110d = 110
    vp_110_rad = vp_110d * np.pi / 180
    reqs = 0
    view_areas = []

    # reqs, area
    for index, region in enumerate(VORONOI_SPHERE_24P.regions):
        phi_c, theta_c = cart_to_spher(*VORONOI_SPHERE_24P.points[index])
        dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
        if dist <= vp_110_rad / 2:
            reqs += 1
            voroni_patch_polygon = polygon.SphericalPolygon(VORONOI_SPHERE_24P.vertices[region])
            fov_polygon = polygon.SphericalPolygon(polygon_fov_cartesian(phi_vp, theta_vp))
            view_area = voroni_patch_polygon.overlap(fov_polygon)
            if view_area > 0:
                view_areas.append(view_area)

    # heatmap
    projection = np.ndarray((t_vert, t_hor))
    for i in range(t_vert):
        for j in range(t_hor):
            index = i * t_hor + j - 1
            phi_c, theta_c = cart_to_spher(*VORONOI_CPOINTS_24P[index])
            dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
            projection[i][j] = 1 if dist <= vp_110_rad / 2 else 0
    heatmap = projection
    return reqs, view_areas, heatmap


def tiles_voro14_fov_110x90_cover_any(phi_vp, theta_vp):
    reqs = 0
    view_areas = []
    for region in VORONOI_SPHERE_14P.regions:
        voroni_patch_polygon = polygon.SphericalPolygon(VORONOI_SPHERE_14P.vertices[region])
        fov_polygon = polygon.SphericalPolygon(polygon_fov_cartesian(phi_vp, theta_vp))
        view_area = voroni_patch_polygon.overlap(fov_polygon)
        if view_area > 0:
            reqs += 1
            view_areas.append(view_area)
    return reqs, view_areas, None


def tiles_voro24_fov_110x90_cover_any(phi_vp, theta_vp):
    reqs = 0
    view_areas = []
    for region in VORONOI_SPHERE_24P.regions:
        voroni_patch_polygon = polygon.SphericalPolygon(VORONOI_SPHERE_24P.vertices[region])
        fov_polygon = polygon.SphericalPolygon(polygon_fov_cartesian(phi_vp, theta_vp))
        view_area = voroni_patch_polygon.overlap(fov_polygon)
        if view_area > 0:
            reqs += 1
            view_areas.append(view_area)
    return reqs, view_areas, None
