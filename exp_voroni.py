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
from typing import List, Callable, Tuple, Any


# -- dataset

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


# -- sphere

TRINITY_NPATCHS = 14
LAYOUT = go.Layout(width=800,
                   margin={'l': 0, 'r': 0, 'b': 0, 't': 40})


def sphere_voroni_points(npatchs) -> tuple[np.ndarray, SphericalVoronoi]:
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


VORONOI_CPOINTS_14P, VORONOI_SPHERE_14P = sphere_voroni_points(TRINITY_NPATCHS)
VORONOI_CPOINTS_24P, VORONOI_SPHERE_24P = sphere_voroni_points(24)


def sphere_plot_voro14_one_user_one_video_matplot():
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


def sphere_data_voro():
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


def sphere_data_add_user_traces(data, dataset, user, video):
    trajc = go.Scatter3d(x=dataset[user][video][:, 1:][:, 0],
                         y=dataset[user][video][:, 1:][:, 1],
                         z=dataset[user][video][:, 1:][:, 2],
                         mode='lines',
                         line={'width': 1, 'color': 'blue'},
                         name='trajectory', showlegend=False)
    data.append(trajc)


def sphere_plot_voro14_one_video_one_user(to_html=False):
    data = sphere_data_voro()
    dataset = get_sample_dataset()
    sphere_data_add_user_traces(data, dataset, ONE_USER, ONE_VIDEO)
    if to_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        go.Figure(data=data, layout=LAYOUT).show()


def sphere_plot_voro14_one_video_all_users(to_html=False):
    data = sphere_data_voro()
    dataset = get_sample_dataset()
    for user in dataset.keys():
        sphere_data_add_user_traces(data, dataset, user, ONE_VIDEO)
    if to_html:
        plotly.offline.plot(data, filename=f'{__file__}.html', auto_open=False)
    else:
        go.Figure(data=data, layout=LAYOUT).show()


# --  erp

TILES_WIDTH, TILES_HEIGHT = 6, 4


def erp_tiles_heatmap(traces):
    heatmap = []
    for i in traces:
        heatmap.append(from_position_to_tile(eulerian_in_range(
            *cartesian_to_eulerian(i[0], i[1], i[2])), TILES_WIDTH, TILES_HEIGHT))
    fig = px.imshow(np.sum(heatmap, axis=0), labels=dict(
        x="longitude", y="latitude", color="requests"), title=f"reqs={str(np.sum(heatmap))}")
    fig.update_layout(LAYOUT)
    fig.show()

# -- points funcs


def points_rectan_tile_cartesian(i, j) -> Tuple[np.ndarray, float, float]:
    t_hor, t_vert = TILES_WIDTH, TILES_HEIGHT
    d_hor = np.deg2rad(360/t_hor)
    d_vert = np.deg2rad(180/t_vert)
    phi_c = d_hor * (j + 0.5)
    theta_c = d_vert * (i + 0.5)
    margin_lateral = np.deg2rad(360/t_hor/2)
    margin_ab = np.deg2rad(180/t_vert/2)
    polygon_rectan_tile = np.array([
        eulerian_to_cartesian(phi_c-margin_ab, theta_c+margin_lateral),
        eulerian_to_cartesian(phi_c+margin_ab, theta_c+margin_lateral),
        eulerian_to_cartesian(phi_c+margin_ab, theta_c-margin_lateral),
        eulerian_to_cartesian(phi_c-margin_ab, theta_c-margin_lateral)])
    return polygon_rectan_tile, phi_c, theta_c


def points_fov_cartesian(phi_vp, theta_vp) -> np.ndarray:
    # https://daglar-cizmeci.com/how-does-virtual-reality-work/
    margin_lateral = np.deg2rad(90/2)
    margin_ab = np.deg2rad(110/2)
    polygon_fov = np.array([
        eulerian_to_cartesian(phi_vp-margin_ab, theta_vp+margin_lateral),
        eulerian_to_cartesian(phi_vp+margin_ab, theta_vp+margin_lateral),
        eulerian_to_cartesian(phi_vp+margin_ab, theta_vp-margin_lateral),
        eulerian_to_cartesian(phi_vp-margin_ab, theta_vp-margin_lateral)])
    return polygon_fov

# -- tiles funcs


def req_plot_per_func(traces,
                      func_list: List[Callable[[float, float], tuple[int, float, List[float], Any]]],
                      plot_lines=False, plot_heatmaps=False):
    fig_reqs = go.Figure(layout=LAYOUT)
    fig_areas = go.Figure(layout=LAYOUT)
    funcs_n_reqs = []
    funcs_avg_area = []
    funcs_vp_quality = []
    for func in func_list:
        traces_n_reqs = []
        traces_avg_areas = []
        traces_heatmaps = []
        traces_vp_quality = []
        # func_funcs_avg_area = [] # to calc avg funcs_avg_area
        # call func per trace
        for t in traces:
            req_in, quality_in, areas_in, heatmap_in = func(*cartesian_to_eulerian(t[0], t[1], t[2]))
            # print(areas_in)
            traces_n_reqs.append(req_in)
            traces_avg_areas.append(np.average(areas_in))
            traces_vp_quality.append(quality_in)
            if heatmap_in is not None:
                traces_heatmaps.append(heatmap_in)
            # func_funcs_avg_area.append(areas_in)
        # line reqs
        fig_reqs.add_trace(go.Scatter(y=traces_n_reqs, mode='lines', name=f"{func.__name__}"))
        # line areas
        fig_areas.add_trace(go.Scatter(y=traces_avg_areas, mode='lines', name=f"{func.__name__}"))
        # line quality
        fig_areas.add_trace(go.Scatter(y=traces_vp_quality, mode='lines', name=f"{func.__name__}"))
        # heatmap
        if(plot_heatmaps):
            fig_heatmap = px.imshow(np.sum(traces_heatmaps, axis=0), title=f"{func.__name__}",
                                    labels=dict(x="longitude", y="latitude", color="requests"))
            fig_heatmap.update_layout(LAYOUT)
            fig_heatmap.show()
        # sum
        funcs_n_reqs.append(np.sum(traces_n_reqs))
        funcs_avg_area.append(np.average(traces_avg_areas))
        funcs_vp_quality.append(np.average(traces_vp_quality))

    # line fig reqs areas
    if(plot_lines):
        fig_reqs.update_layout(xaxis_title="user position", yaxis_title="req_tiles",)
        fig_reqs.show()
        fig_areas.update_layout(xaxis_title="user position", yaxis_title="avg req_tiles view_ratio",)
        fig_areas.show()

    # bar fig funcs_n_reqs funcs_avg_area
    funcs_names = [str(func.__name__) for func in func_list]
    fig_bar = make_subplots(rows=1, cols=4,  subplot_titles=(
        "req_tiles", "avg req_tiles view_ratio", "avg VP quality_ratio", "score=quality_ratio/(req_tiles*(1-view_ratio)))"), shared_yaxes=True)
    fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_n_reqs, orientation='h'), row=1, col=1)
    fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_avg_area, orientation='h'), row=1, col=2)
    fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_vp_quality, orientation='h'), row=1, col=3)
    funcs_score = [funcs_vp_quality[i] * (1 / (funcs_n_reqs[i] * (1 - funcs_avg_area[i])))
                   for i, _ in enumerate(funcs_n_reqs)]
    fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_score, orientation='h'), row=1, col=4)
    fig_bar.update_layout(width=1500, showlegend=False)
    fig_bar.update_layout(barmode="stack")
    fig_bar.show()


def req_tiles_rectan_fov_required_intersec(phi_vp, theta_vp, required_intersec):
    t_hor, t_vert = TILES_WIDTH, TILES_HEIGHT
    projection = np.ndarray((t_vert, t_hor))
    view_areas = []
    vp_quality = 0.0
    for i in range(t_vert):
        for j in range(t_hor):
            rectan_tile_points, phi_c, theta_c = points_rectan_tile_cartesian(i, j)
            projection[i][j] = 0
            rectan_tile_polygon = polygon.SphericalPolygon(rectan_tile_points)
            fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
            view_area = rectan_tile_polygon.overlap(fov_polygon)
            if view_area > required_intersec and view_area <= 1:  # TODO: reivew this
                projection[i][j] = 1
                view_area = 1 if view_area > 1 else view_area  # TODO: reivew this
                view_areas.append(view_area)
                vp_quality += fov_polygon.overlap(rectan_tile_polygon)
    reqs = np.sum(projection)
    heatmap = projection
    return reqs, vp_quality, view_areas, heatmap


def req_tiles_rectan_fov_110radius_cover_center(phi_vp, theta_vp):
    t_hor, t_vert = TILES_WIDTH, TILES_HEIGHT
    projection = np.ndarray((t_vert, t_hor))
    # vp_110d = 110
    # vp_110_rad = vp_110d * np.pi / 180
    vp_110_rad_half = np.deg2rad(110/2)
    view_areas = []
    vp_quality = 0.0
    for i in range(t_vert):
        for j in range(t_hor):
            rectan_tile_points, phi_c, theta_c = points_rectan_tile_cartesian(i, j)
            dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
            projection[i][j] = 0
            if dist <= vp_110_rad_half:
                projection[i][j] = 1
                rectan_tile_polygon = polygon.SphericalPolygon(rectan_tile_points)
                fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
                view_area = rectan_tile_polygon.overlap(fov_polygon)
                view_areas.append(view_area)
                vp_quality += fov_polygon.overlap(rectan_tile_polygon)
    reqs = np.sum(projection)
    heatmap = projection
    return reqs, vp_quality, view_areas, heatmap


def req_tiles_voro_fov_110radius_cover_center(phi_vp, theta_vp, spherical_voronoi: SphericalVoronoi):
    t_hor, t_vert = TILES_WIDTH, TILES_HEIGHT
    # vp_110d = 110
    # vp_110_rad = vp_110d * np.pi / 180
    vp_110_rad_half = np.deg2rad(110/2)
    reqs = 0
    view_areas = []
    vp_quality = 0.0

    # reqs, area
    for index, region in enumerate(spherical_voronoi.regions):
        phi_c, theta_c = cart_to_spher(*spherical_voronoi.points[index])
        dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
        if dist <= vp_110_rad_half:
            reqs += 1
            voroni_tile_polygon = polygon.SphericalPolygon(spherical_voronoi.vertices[region])
            fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
            view_area = voroni_tile_polygon.overlap(fov_polygon)
            view_areas.append(view_area)
            vp_quality += fov_polygon.overlap(voroni_tile_polygon)

    # heatmap
    projection = np.ndarray((t_vert, t_hor))
    for i in range(t_vert):
        for j in range(t_hor):
            index = i * t_hor + j - 1
            phi_c, theta_c = cart_to_spher(*VORONOI_CPOINTS_24P[index])
            dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
            projection[i][j] = 1 if dist <= vp_110_rad_half else 0
    heatmap = projection
    return reqs, vp_quality, view_areas, heatmap


def req_tiles_voro_fov_110x90_cover_any(phi_vp, theta_vp, spherical_voronoi: SphericalVoronoi):
    reqs = 0
    view_areas = []
    vp_quality = 0.0
    for region in spherical_voronoi.regions:
        voroni_tile_polygon = polygon.SphericalPolygon(spherical_voronoi.vertices[region])
        fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
        view_area = voroni_tile_polygon.overlap(fov_polygon)
        if view_area > 0:
            reqs += 1
            view_areas.append(view_area)
            vp_quality += fov_polygon.overlap(voroni_tile_polygon)
    return reqs, vp_quality, view_areas, None


def req_tiles_voro_fov_required_intersec(phi_vp, theta_vp, spherical_voronoi: SphericalVoronoi, required_intersec):
    reqs = 0
    view_areas = []
    vp_quality = 0.0
    for region in spherical_voronoi.regions:
        voroni_tile_polygon = polygon.SphericalPolygon(spherical_voronoi.vertices[region])
        fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
        view_area = voroni_tile_polygon.overlap(fov_polygon)
        if view_area > required_intersec:
            reqs += 1
            view_area = 1 if view_area > 1 else view_area  # TODO: reivew this
            view_areas.append(view_area)
            vp_quality += fov_polygon.overlap(voroni_tile_polygon)
    return reqs, vp_quality, view_areas, None


def req_tiles_rectan_fov_20perc_cover(phi_vp, theta_vp):
    return req_tiles_rectan_fov_required_intersec(phi_vp, theta_vp, 0.2)


def req_tiles_rectan_fov_33perc_cover(phi_vp, theta_vp):
    return req_tiles_rectan_fov_required_intersec(phi_vp, theta_vp, 0.33)


def req_tiles_voro14_fov_110radius_cover_center(phi_vp, theta_vp):
    return req_tiles_voro_fov_110radius_cover_center(phi_vp, theta_vp, VORONOI_SPHERE_14P)


def req_tiles_voro24_fov_110radius_cover_center(phi_vp, theta_vp):
    return req_tiles_voro_fov_110radius_cover_center(phi_vp, theta_vp, VORONOI_SPHERE_24P)


def req_tiles_voro14_fov_110x90_cover_any(phi_vp, theta_vp):
    return req_tiles_voro_fov_110x90_cover_any(phi_vp, theta_vp, VORONOI_SPHERE_14P)


def req_tiles_voro24_fov_110x90_cover_any(phi_vp, theta_vp):
    return req_tiles_voro_fov_110x90_cover_any(phi_vp, theta_vp, VORONOI_SPHERE_24P)


def req_tiles_voro14_fov_20perc_cover(phi_vp, theta_vp):
    return req_tiles_voro_fov_required_intersec(phi_vp, theta_vp, VORONOI_SPHERE_14P, 0.2)


def req_tiles_voro14_fov_33perc_cover(phi_vp, theta_vp):
    return req_tiles_voro_fov_required_intersec(phi_vp, theta_vp, VORONOI_SPHERE_14P, 0.33)


def req_tiles_voro24_fov_20perc_cover(phi_vp, theta_vp):
    return req_tiles_voro_fov_required_intersec(phi_vp, theta_vp, VORONOI_SPHERE_24P, 0.2)


def req_tiles_voro24_fov_33perc_cover(phi_vp, theta_vp):
    return req_tiles_voro_fov_required_intersec(phi_vp, theta_vp, VORONOI_SPHERE_24P, 0.33)
