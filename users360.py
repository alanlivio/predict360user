# %%
from users360 import *
from enum import Enum, auto
from plotly.subplots import make_subplots
from head_motion_prediction.Utils import *
from os.path import exists
from scipy.spatial import SphericalVoronoi, geometric_slerp
from spherical_geometry import polygon
from typing import List, Callable, Tuple, Any
import math
from scipy.stats import entropy
import numpy as np
import os
import pickle
import plotly
import plotly.express as px
import plotly.graph_objs as go
import sys

ONE_USER = '0'
ONE_VIDEO = '10_Cows'
LAYOUT = go.Layout(width=600)
TILES_H6, TILES_V4 = 6, 4
TRINITY_NPATCHS = 14


def layout_with_title(title):
    return go.Layout(width=800, title=title)


def points_voroni(npatchs) -> SphericalVoronoi:
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
    return sv


VORONOI_14P = points_voroni(TRINITY_NPATCHS)
VORONOI_24P = points_voroni(24)


def cart_to_spher(x, y, z):
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    theta = np.arccos(z)
    return phi, theta


def spher_to_cart(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def arc_dist(phi_1, theta_1, phi_2, theta_2):
    t_1 = np.cos(theta_1) * np.cos(theta_2)
    t_2 = np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2)
    return np.arccos(t_1 + t_2)


def points_rectan_tile_cartesian(i, j, t_hor, t_vert) -> Tuple[np.ndarray, float, float]:
    d_hor = np.deg2rad(360/t_hor)
    d_vert = np.deg2rad(180/t_vert)
    phi_c = d_hor * (j + 0.5)
    theta_c = d_vert * (i + 0.5)
    polygon_rectan_tile = np.array([
        eulerian_to_cartesian(d_hor * j, d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * i),
        eulerian_to_cartesian(d_hor * j, d_vert * i)])
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


class Users360:
    sample_dataset = None
    sample_dataset_pickle = 'users360.pickle'

    def __init__(self, dataset=None):
        if dataset is None:
            self.dataset = self._get_sample_dataset()

    # -- dataset funcs

    def _get_sample_dataset(self, load=False):
        if Users360.sample_dataset is None:
            if load or not exists(Users360.sample_dataset_pickle):
                sys.path.append('head_motion_prediction')
                from head_motion_prediction.David_MMSys_18.Read_Dataset import load_sampled_dataset
                project_path = "head_motion_prediction"
                cwd = os.getcwd()
                if os.path.basename(cwd) != project_path:
                    print(f"-- get Users360.sample_dataset from {project_path}")
                    os.chdir(project_path)
                    Users360.sample_dataset = load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Users360.sample_dataset_pickle, 'wb') as f:
                        pickle.dump(Users360.sample_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f"-- get Users360.sample_dataset from {Users360.sample_dataset_pickle}")
                with open(Users360.sample_dataset_pickle, 'rb') as f:
                    Users360.sample_dataset = pickle.load(f)
        return Users360.sample_dataset

    # -- traces funcs

    def get_one_trace(self):
        return self.dataset[ONE_USER][ONE_VIDEO][:, 1:][:1]

    def get_one_trace_eulerian(self):
        trace_cartesian = self.get_one_trace()[0]
        return cartesian_to_eulerian(trace_cartesian[0], trace_cartesian[1], trace_cartesian[2])

    def get_traces_one_video_one_user(self):
        return self.dataset[ONE_USER][ONE_VIDEO][:, 1:]

    def get_traces_one_video_all_users(self):
        n_traces = len(self.dataset[ONE_USER][ONE_VIDEO][:, 1:])
        traces = np.ndarray((len(self.dataset.keys())*n_traces, 3))
        count = 0
        for user in self.dataset.keys():
            for i in self.dataset[user][ONE_VIDEO][:, 1:]:
                traces.itemset((count, 0), i[0])
                traces.itemset((count, 1), i[1])
                traces.itemset((count, 2), i[2])
                count += 1
        return traces

    def get_traces_random_one_user(self, num):
        one_user = self.get_traces_one_video_one_user()
        step = int(len(one_user)/num)
        return one_user[::step]


class Traces360:
    def __init__(self, traces: np.ndarray, title_sufix="", verbose=False):
        assert traces.shape[1] == 3  # check cartesian ndarray
        self.verbose = verbose
        self.traces = traces
        self.title_sufix = str(len(traces)) + "_traces" if not title_sufix else title_sufix
        if self.verbose:
            print("Users360.traces.shape is " + str(traces.shape))

    # -- sphere funcs

    def _sphere_data_voro(self, spherical_voronoi: SphericalVoronoi):
        data = []

        # add generator points
        # gens = go.Scatter3d(x=spherical_voronoi.points[:, 0], y=spherical_voronoi.points[:, 1], z=spherical_voronoi.points[:, 2], mode='markers', marker={
        #                     'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='voronoi center')
        # data.append(gens)

        # -- add vortonoi vertices
        # vrts = go.Scatter3d(x=spherical_voronoi.vertices[:, 0],
        #                     y=spherical_voronoi.vertices[:, 1],
        #                     z=spherical_voronoi.vertices[:, 2],
        #                     mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'green'}, name='voronoi Vertices')
        # data.append(vrts)

        # add vortonoi edges
        for region in spherical_voronoi.regions:
            n = len(region)
            t = np.linspace(0, 1, 100)
            for i in range(n):
                start = spherical_voronoi.vertices[region][i]
                end = spherical_voronoi.vertices[region][(i + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                    'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
                data.append(edge)
        return data

    def _sphere_data_add_user(self, data):
        trajc = go.Scatter3d(x=self.traces[:, 0],
                             y=self.traces[:, 1],
                             z=self.traces[:, 2],
                             mode='lines',
                             line={'width': 1, 'color': 'blue'},
                             name='trajectory', showlegend=False)
        data.append(trajc)

    def _sphere_data_rectan_tiles(self, t_hor, t_vert):
        data = []
        for i in range(t_hor+1):
            for j in range(t_vert+1):
                # -- add rectan tiles edges
                rectan_tile_points, phi_c, theta_c = points_rectan_tile_cartesian(i, j, t_hor, t_vert)
                n = len(rectan_tile_points)
                t = np.linspace(0, 1, 100)
                for index in range(n):
                    start = rectan_tile_points[index]
                    end = rectan_tile_points[(index + 1) % n]
                    result = np.array(geometric_slerp(start, end, t))
                    edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                        'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
                    data.append(edge)
                # x, y, z = eulerian_to_cartesian(phi_c, theta_c)
                # corners = go.Scatter3d(x=rectan_tile_points[:, 0], y=rectan_tile_points[:, 1], z=rectan_tile_points[:, 2], mode='markers', marker={
                #     'size': 2, 'opacity': 1.0, 'color': 'blue'}, name='tile corners', showlegend=False)
                # data.append(corners)
                # centers = go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker={
                #     'size': 2, 'opacity': 1.0, 'color': 'red'}, name='tile center', showlegend=False)
                # data.append(centers)
        return data

    def plot_sphere_voro_matplot(self, spherical_voronoi: SphericalVoronoi = VORONOI_14P):
        """
        Example:
            plot_sphere_voro_matplot(get_traces_one_video_one_user())
        """
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
        ax.scatter(spherical_voronoi.points[:, 0], spherical_voronoi.points[:,
                                                                            1], spherical_voronoi.points[:, 2], c='b')
        # plot voronoi vertices
        ax.scatter(spherical_voronoi.vertices[:, 0], spherical_voronoi.vertices[:, 1], spherical_voronoi.vertices[:, 2],
                   c='g')
        # indicate voronoi regions (as Euclidean polygons)
        for region in spherical_voronoi.regions:
            n = len(region)
            for i in range(n):
                start = spherical_voronoi.vertices[region][i]
                end = spherical_voronoi.vertices[region][(i + 1) % n]
                result = geometric_slerp(start, end, t_vals)
                ax.plot(result[..., 0],
                        result[..., 1],
                        result[..., 2],
                        c='k')

        # trajectory = get_traces_one_video_one_user()
        ax.plot(self.traces[:, 0], self.traces[:, 1],
                self.traces[:, 2], label='parametric curve')
        plt.show()

    def plot_sphere_voro(self, spherical_voronoi: SphericalVoronoi, to_html=False):
        data = self._sphere_data_voro(spherical_voronoi)
        self._sphere_data_add_user(data)
        title = f"traces_voro{len(spherical_voronoi.points)}_" + self.title_sufix
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{title}.html', auto_open=False)
        else:
            fig.show()

    def plot_sphere_voro_with_vp(self, spherical_voronoi: SphericalVoronoi, to_html=False):
        data = self._sphere_data_voro(spherical_voronoi)
        for trace in self.traces:
            phi_vp, theta_vp = cartesian_to_eulerian(trace[0], trace[1], trace[2])
            fov_polygon = points_fov_cartesian(phi_vp, theta_vp)
            n = len(fov_polygon)
            t = np.linspace(0, 1, 100)
            for index in range(n):
                start = fov_polygon[index]
                end = fov_polygon[(index + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                    'width': 5, 'color': 'red'}, name='vp edge', showlegend=False)
                data.append(edge)
        title = f"FoV_voro{len(spherical_voronoi.points)}_" + self.title_sufix
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{title}.html', auto_open=False)
        else:
            fig.show()

    def plot_sphere_rectan_with_vp(self, t_hor, t_vert, to_html=False):
        data = self._sphere_data_rectan_tiles(t_hor, t_vert)
        for trace in self.traces:
            phi_vp, theta_vp = cartesian_to_eulerian(trace[0], trace[1], trace[2])
            fov_polygon = points_fov_cartesian(phi_vp, theta_vp)
            n = len(fov_polygon)
            t = np.linspace(0, 1, 100)
            for index in range(n):
                start = fov_polygon[index]
                end = fov_polygon[(index + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                    'width': 5, 'color': 'red'}, name='vp edge', showlegend=False)
                data.append(edge)
        title = f"FoV_rectan{t_hor}x{t_vert}_" + self.title_sufix
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{title}.html', auto_open=False)
        else:
            fig.show()

    def plot_sphere_rectan(self, t_hor, t_vert, to_html=False):
        data = self._sphere_data_rectan_tiles(t_hor, t_vert)
        self._sphere_data_add_user(data)
        title = f"traces_rectan{t_hor}x{t_vert}_" + self.title_sufix
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{title}.html', auto_open=False)
        else:
            fig.show()

    # -- erp funcs

    def plot_erp_heatmap(self):
        heatmaps = []
        fov = FoV360TilesRect(6, 4, FoV360.Cover.CENTER)
        for i in self.traces:
            _, _, _, heatmap = fov.request(*cartesian_to_eulerian(i[0], i[1], i[2]))
            heatmaps.append(heatmap)
        fig_heatmap = px.imshow(np.sum(heatmaps, axis=0), labels=dict(
            x="longitude", y="latitude", color="requests", title=f"reqs={str(np.sum(heatmaps))}"))
        fig_heatmap.update_layout(LAYOUT)
        fig_heatmap.show()

    # -- reqs funcs

    def plot_reqs_per_func(self, func_list,
                           plot_lines=False, plot_heatmaps=False):
        fig_reqs = go.Figure(layout=LAYOUT)
        fig_areas = go.Figure(layout=LAYOUT)
        fig_quality = go.Figure(layout=LAYOUT)
        funcs_n_reqs = []
        funcs_avg_area = []
        funcs_vp_quality = []
        for func in func_list:
            traces_n_reqs = []
            traces_areas = []
            traces_areas_svg = []
            traces_heatmaps = []
            traces_vp_quality = []
            # func_funcs_avg_area = [] # to calc avg funcs_avg_area
            # call func per trace
            for t in self.traces:
                req_in, quality_in, areas_in, heatmap_in = func.request(*cartesian_to_eulerian(t[0], t[1], t[2]))
                # print(areas_in)
                traces_n_reqs.append(req_in)
                traces_areas.append(areas_in)
                traces_areas_svg.append(np.average(areas_in))
                traces_vp_quality.append(quality_in)
                if heatmap_in is not None:
                    traces_heatmaps.append(heatmap_in)
                # func_funcs_avg_area.append(areas_in)
            # line reqs
            fig_reqs.add_trace(go.Scatter(y=traces_n_reqs, mode='lines', name=f"{func.title()}"))
            # line areas
            fig_areas.add_trace(go.Scatter(y=traces_areas_svg, mode='lines', name=f"{func.title()}"))
            # line quality
            fig_quality.add_trace(go.Scatter(y=traces_vp_quality, mode='lines', name=f"{func.title()}"))
            # heatmap
            if(plot_heatmaps and len(traces_heatmaps)):
                fig_heatmap = px.imshow(np.sum(traces_heatmaps, axis=0), title=f"{func.title()}",
                                        labels=dict(x="longitude", y="latitude", color="FoV360s"))
                fig_heatmap.update_layout(LAYOUT)
                fig_heatmap.show()
            # sum
            funcs_n_reqs.append(np.sum(traces_n_reqs))
            funcs_avg_area.append(np.average(traces_areas_svg))
            funcs_vp_quality.append(np.average(traces_vp_quality))

        # line fig reqs areas
        if(plot_lines):
            fig_reqs.update_layout(xaxis_title="user trace", title="req_tiles " + self.title_sufix).show()
            fig_areas.update_layout(xaxis_title="user trace",
                                    title="avg req_tiles view_ratio " + self.title_sufix).show()
            fig_quality.update_layout(xaxis_title="user trace", title="avg quality ratio " + self.title_sufix).show()

        # bar fig funcs_n_reqs funcs_avg_area
        funcs_names = [str(func.title()) for func in func_list]
        fig_bar = make_subplots(rows=1, cols=4,  subplot_titles=(
            "req_tiles", "avg req_tiles view_ratio", "avg VP quality_ratio", "score=quality_ratio/(req_tiles*(1-view_ratio)))"), shared_yaxes=True)
        fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_n_reqs, orientation='h'), row=1, col=1)
        fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_avg_area, orientation='h'), row=1, col=2)
        fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_vp_quality, orientation='h'), row=1, col=3)
        funcs_score = [funcs_vp_quality[i] * (1 / (funcs_n_reqs[i] * (1 - funcs_avg_area[i])))
                       for i, _ in enumerate(funcs_n_reqs)]
        fig_bar.add_trace(go.Bar(y=funcs_names, x=funcs_score, orientation='h'), row=1, col=4)
        fig_bar.update_layout(width=1500, showlegend=False, title_text=self.title_sufix)
        fig_bar.update_layout(barmode="stack")
        fig_bar.show()


class FoV360:
    class Cover(Enum):
        ANY = auto()
        CENTER = auto()
        ONLY20PERC = auto()
        ONLY33PERC = auto()

    def request(self, phi_vp, theta_vp):
        pass


class FoV360TilesRect(FoV360):
    def __init__(self, t_hor, t_vert, cover: FoV360.Cover):
        self.t_hor, self.t_vert = t_hor, t_vert
        self.cover = cover

    def title(self):
        match self.cover:
            case FoV360.Cover.ANY:
                return f'rect_tiles_{self.t_hor}x{self.t_vert}_any'
            case FoV360.Cover.CENTER:
                return f'rect_tiles_{self.t_hor}x{self.t_vert}_center'
            case FoV360.Cover.ONLY20PERC:
                return f'rect_tiles_{self.t_hor}x{self.t_vert}_20perc'
            case FoV360.Cover.ONLY33PERC:
                return f'rect_tiles_{self.t_hor}x{self.t_vert}_33perc'

    def request(self, phi_vp, theta_vp):
        match self.cover:
            case FoV360.Cover.ANY:
                raise NotImplementedError()
            case FoV360.Cover.CENTER:
                return self.request_110radius_center(phi_vp, theta_vp)
            case FoV360.Cover.ONLY20PERC:
                return self.request_only(phi_vp, theta_vp, 0.2)
            case FoV360.Cover.ONLY33PERC:
                return self.request_only(phi_vp, theta_vp, 0.33)

    def request_only(self, phi_vp, theta_vp, required_intersec):
        # t_hor, t_vert = TILES_H6, TILES_V4
        projection = np.ndarray((self.t_vert, self.t_hor))
        areas_in = []
        vp_quality = 0.0
        vp_threshold = np.deg2rad(110/2)
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                rectan_tile_points, phi_c, theta_c = points_rectan_tile_cartesian(i, j, self.t_hor, self.t_vert)
                dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
                if dist <= vp_threshold:
                    rectan_tile_polygon = polygon.SphericalPolygon(rectan_tile_points)
                    fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
                    view_area = rectan_tile_polygon.overlap(fov_polygon)
                    if view_area > required_intersec:
                        projection[i][j] = 1
                        areas_in.append(view_area)
                        vp_quality += fov_polygon.overlap(rectan_tile_polygon)
                else:
                    projection[i][j] = 0
        reqs = np.sum(projection)
        heatmap = projection
        return reqs, vp_quality, areas_in, heatmap

    def request_110radius_center(self, phi_vp, theta_vp):
        # t_hor, t_vert = TILES_H6, TILES_V4
        projection = np.ndarray((self.t_vert, self.t_hor))
        # vp_110d = 110
        # vp_110_rad = vp_110d * np.pi / 180
        vp_110_rad_half = np.deg2rad(110/2)
        areas_in = []
        vp_quality = 0.0
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                rectan_tile_points, phi_c, theta_c = points_rectan_tile_cartesian(i, j, self.t_hor, self.t_vert)
                dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
                if dist <= vp_110_rad_half:
                    projection[i][j] = 1
                    rectan_tile_polygon = polygon.SphericalPolygon(rectan_tile_points)
                    fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
                    view_area = rectan_tile_polygon.overlap(fov_polygon)
                    areas_in.append(view_area)
                    vp_quality += fov_polygon.overlap(rectan_tile_polygon)
                else:
                    projection[i][j] = 0
        reqs = np.sum(projection)
        heatmap = projection
        return reqs, vp_quality, areas_in, heatmap


class FoV360TilesVoro(FoV360):
    def __init__(self, spherical_voronoi: SphericalVoronoi, cover: FoV360.Cover):
        self.spherical_voronoi = spherical_voronoi
        self.cover = cover

    def title(self):
        match self.cover:
            case FoV360.Cover.ANY:
                return f'voroni_tiles_{len(self.spherical_voronoi.points)}_any'
            case FoV360.Cover.CENTER:
                return f'voroni_tiles_{len(self.spherical_voronoi.points)}_center'
            case FoV360.Cover.ONLY20PERC:
                return f'voroni_tiles_{len(self.spherical_voronoi.points)}_20perc'
            case FoV360.Cover.ONLY33PERC:
                return f'voroni_tiles_{len(self.spherical_voronoi.points)}_33perc'

    def request(self, phi_vp, theta_vp):
        match self.cover:
            case FoV360.Cover.ANY:
                return self.request_110x90_any(phi_vp, theta_vp)
            case FoV360.Cover.CENTER:
                return self.request_110radius_center(phi_vp, theta_vp)
            case FoV360.Cover.ONLY20PERC:
                return self.request_required_intersec(phi_vp, theta_vp, 0.2)
            case FoV360.Cover.ONLY33PERC:
                return self.request_required_intersec(phi_vp, theta_vp, 0.33)

    def request_110radius_center(self, phi_vp, theta_vp):
        # t_hor, t_vert = TILES_H6, TILES_V4
        # vp_110d = 110
        # vp_110_rad = vp_110d * np.pi / 180
        vp_110_rad_half = np.deg2rad(110/2)
        reqs = 0
        areas_in = []
        vp_quality = 0.0

        # reqs, area
        for index, region in enumerate(self.spherical_voronoi.regions):
            phi_c, theta_c = cart_to_spher(*self.spherical_voronoi.points[index])
            dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
            if dist <= vp_110_rad_half:
                reqs += 1
                voroni_tile_polygon = polygon.SphericalPolygon(self.spherical_voronoi.vertices[region])
                fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
                view_area = voroni_tile_polygon.overlap(fov_polygon)
                areas_in.append(view_area)
                vp_quality += fov_polygon.overlap(voroni_tile_polygon)

        # heatmap # TODO review
        # projection = np.ndarray((t_vert, t_hor))
        # for i in range(t_vert):
        #     for j in range(t_hor):
        #         index = i * t_hor + j
        #         phi_c, theta_c = cart_to_spher(*spherical_voronoi.points[index])
        #         dist = arc_dist(phi_vp, theta_vp, phi_c, theta_c)
        #         projection[i][j] = 1 if dist <= vp_110_rad_half else 0
        # heatmap = projection
        return reqs, vp_quality, areas_in, None

    def request_110x90_any(self, phi_vp, theta_vp):
        reqs = 0
        areas_in = []
        vp_quality = 0.0
        for region in self.spherical_voronoi.regions:
            voroni_tile_polygon = polygon.SphericalPolygon(self.spherical_voronoi.vertices[region])
            fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
            view_area = voroni_tile_polygon.overlap(fov_polygon)
            if view_area > 0:
                reqs += 1
                areas_in.append(view_area)
                vp_quality += fov_polygon.overlap(voroni_tile_polygon)
        return reqs, vp_quality, areas_in, None

    def request_required_intersec(self, phi_vp, theta_vp, required_intersec):
        reqs = 0
        areas_in = []
        vp_quality = 0.0
        for region in self.spherical_voronoi.regions:
            voroni_tile_polygon = polygon.SphericalPolygon(self.spherical_voronoi.vertices[region])
            fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(phi_vp, theta_vp))
            view_area = voroni_tile_polygon.overlap(fov_polygon)
            if view_area > required_intersec:
                reqs += 1
                view_area = 1 if view_area > 1 else view_area  # TODO: reivew this
                areas_in.append(view_area)
                vp_quality += fov_polygon.overlap(voroni_tile_polygon)
        return reqs, vp_quality, areas_in, None

# %%
