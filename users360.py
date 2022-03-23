# %%
from enum import Enum, auto
from head_motion_prediction.Utils import *
from os.path import exists
from plotly.subplots import make_subplots
from scipy.spatial import SphericalVoronoi, geometric_slerp
from scipy.stats import entropy
from spherical_geometry import polygon
from typing import List, Callable, Tuple, Any
import math
import numpy as np
import os
import pickle
import plotly
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats
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


def points_rectan_tile_cartesian(i, j, t_hor, t_vert) -> Tuple[np.ndarray, np.ndarray]:
    d_hor = np.deg2rad(360/t_hor)
    d_vert = np.deg2rad(180/t_vert)
    theta_c = d_hor * (j + 0.5)
    phi_c = d_vert * (i + 0.5)
    polygon_rectan_tile = np.array([
        eulerian_to_cartesian(d_hor * j, d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * (i+1)),
        eulerian_to_cartesian(d_hor * (j+1), d_vert * i),
        eulerian_to_cartesian(d_hor * j, d_vert * i)])
    return polygon_rectan_tile, eulerian_to_cartesian(theta_c, phi_c)


def points_fov_cartesian(x, y, z) -> np.ndarray:
    # https://daglar-cizmeci.com/how-does-virtual-reality-work/
    margin_lateral = np.deg2rad(90/2)
    margin_ab = np.deg2rad(110/2)
    theta_vp, phi_vp = cartesian_to_eulerian(x, y, z)
    polygon_fov = np.array([
        eulerian_to_cartesian(theta_vp+margin_lateral, phi_vp-margin_ab),
        eulerian_to_cartesian(theta_vp+margin_lateral, phi_vp+margin_ab),
        eulerian_to_cartesian(theta_vp-margin_lateral, phi_vp+margin_ab),
        eulerian_to_cartesian(theta_vp-margin_lateral, phi_vp-margin_ab)])
    return polygon_fov


class Dataset:
    sample_dataset = None
    sample_dataset_pickle = 'users360.pickle'

    def __init__(self, dataset=None):
        if dataset is None:
            self.dataset = self._get_sample_dataset()
            self.users_id = [key for key in self.dataset.keys()]

    # -- dataset funcs

    def _get_sample_dataset(self, load=False):
        if Dataset.sample_dataset is None:
            if load or not exists(Dataset.sample_dataset_pickle):
                sys.path.append('head_motion_prediction')
                from head_motion_prediction.David_MMSys_18.Read_Dataset import load_sampled_dataset
                project_path = "head_motion_prediction"
                cwd = os.getcwd()
                if os.path.basename(cwd) != project_path:
                    print(f"-- get Dataset.sample_dataset from {project_path}")
                    os.chdir(project_path)
                    Dataset.sample_dataset = load_sampled_dataset()
                    os.chdir(cwd)
                    with open(Dataset.sample_dataset_pickle, 'wb') as f:
                        pickle.dump(Dataset.sample_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f"-- get Dataset.sample_dataset from {Dataset.sample_dataset_pickle}")
                with open(Dataset.sample_dataset_pickle, 'rb') as f:
                    Dataset.sample_dataset = pickle.load(f)
        return Dataset.sample_dataset

    # -- cluster funcs

    def get_cluster_entropy_by_vpextract(self, vp=None):
        if vp is None:
            vp = VPExtractTilesRect(6, 4, cover=VPExtract.Cover.CENTER)
        users_entropy = np.ndarray(len(self.users_id))
        # fill users_entropy
        for index, user in enumerate(self.users_id[:1]):
            heatmaps = []
            for trace in self.dataset[user][ONE_VIDEO][:, 1:]:
                _, _, _, heatmap = vp.request(*trace)
                heatmaps.append(heatmap)
            sum = np.sum(heatmaps, axis=0).reshape((-1))
            print(sum)
            # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
            users_entropy[index] = scipy.stats.entropy(sum)
            # print(users_entropy[index])
        # define class threshold
        # https://www.askpython.com/python/normal-distribution
        # https://matthew-brett.github.io/teaching/random_fields.html
        return users_entropy

    def get_one_trace(self) -> np.ndarray:
        return self.dataset[ONE_USER][ONE_VIDEO][:, 1:][:1]

    def get_traces_one_video_one_user(self) -> np.ndarray:
        return self.dataset[ONE_USER][ONE_VIDEO][:, 1:]

    def get_traces_one_video_all_users(self) -> np.ndarray:
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

    def get_traces_random_one_user(self, num) -> np.ndarray:
        one_user = self.get_traces_one_video_one_user()
        step = int(len(one_user)/num)
        return one_user[::step]


class Plot:
    def __init__(self, traces: np.ndarray, title_sufix="", verbose=False):
        assert traces.shape[1] == 3  # check if cartesian
        self.verbose = verbose
        self.traces = traces
        self.title_sufix = str(len(traces)) + "_traces" if not title_sufix else title_sufix
        if self.verbose:
            print("Dataset.traces.shape is " + str(traces.shape))

    # -- sphere funcs

    def _sphere_data_voro(self, spherical_voronoi: SphericalVoronoi, with_generators=False):
        data = []

        # add generator points
        if with_generators:
            gens = go.Scatter3d(x=spherical_voronoi.points[:, 0], y=spherical_voronoi.points[:, 1], z=spherical_voronoi.points[:, 2], mode='markers', marker={
                                'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='voronoi center')
            data.append(gens)

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
                rectan_tile_points, _ = points_rectan_tile_cartesian(i, j, t_hor, t_vert)
                n = len(rectan_tile_points)
                t = np.linspace(0, 1, 100)
                for index in range(n):
                    start = rectan_tile_points[index]
                    end = rectan_tile_points[(index + 1) % n]
                    result = np.array(geometric_slerp(start, end, t))
                    edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                        'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
                    data.append(edge)
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
            plotly.offline.plot(fig, filename=f'./plot_figs/{title}.html', auto_open=False)
        else:
            fig.show()

    def plot_sphere_voro_with_vp(self, spherical_voronoi: SphericalVoronoi, to_html=False):
        data = self._sphere_data_voro(spherical_voronoi)
        for trace in self.traces:
            fov_polygon = points_fov_cartesian(trace[0], trace[1], trace[2])
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
            plotly.offline.plot(fig, filename=f'./plot_figs/{title}.html', auto_open=False)
        else:
            fig.show()

    def plot_sphere_rectan_with_vp(self, t_hor, t_vert, to_html=False):
        data = self._sphere_data_rectan_tiles(t_hor, t_vert)
        for trace in self.traces:
            fov_polygon = points_fov_cartesian(trace[0], trace[1], trace[2])
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
            plotly.offline.plot(fig, filename=f'./plot_figs/{title}.html', auto_open=False)
        else:
            fig.show()

    def plot_sphere_rectan(self, t_hor, t_vert, to_html=False):
        data = self._sphere_data_rectan_tiles(t_hor, t_vert)
        self._sphere_data_add_user(data)
        title = f"traces_rectan{t_hor}x{t_vert}_" + self.title_sufix
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'./plot_figs/{title}.html', auto_open=False)
        else:
            fig.show()

    # -- erp funcs

    def plot_erp_heatmap(self):
        heatmaps = []
        fov = VPExtractTilesRect(6, 4, VPExtract.Cover.CENTER)
        for i in self.traces:
            _, _, _, heatmap = fov.request(i[0], i[1], i[2])
            heatmaps.append(heatmap)
        fig_heatmap = px.imshow(np.sum(heatmaps, axis=0), labels=dict(
            x="longitude", y="latitude", color="requests", title=f"reqs={str(np.sum(heatmaps))}"))
        fig_heatmap.update_layout(LAYOUT)
        fig_heatmap.show()

    # -- reqs funcs

    def plot_vp_extractions(self, vp_extrac_list, plot_bars=True,
                            plot_lines=False, plot_heatmaps=False):
        fig_reqs = go.Figure(layout=LAYOUT)
        fig_areas = go.Figure(layout=LAYOUT)
        fig_quality = go.Figure(layout=LAYOUT)
        vp_extract_n_reqs = []
        vp_extract_avg_area = []
        vp_extract_quality = []
        for vp_extract in vp_extrac_list:
            traces_n_reqs = []
            traces_areas = []
            traces_areas_svg = []
            traces_heatmaps = []
            traces_vp_quality = []
            # call func per trace
            for t in self.traces:
                req_in, quality_in, areas_in, heatmap_in = vp_extract.request(t[0], t[1], t[2])
                traces_n_reqs.append(req_in)
                traces_areas.append(areas_in)
                traces_areas_svg.append(np.average(areas_in))
                traces_vp_quality.append(quality_in)
                if heatmap_in is not None:
                    traces_heatmaps.append(heatmap_in)
            # line reqs
            fig_reqs.add_trace(go.Scatter(y=traces_n_reqs, mode='lines', name=f"{vp_extract.title()}"))
            # line areas
            fig_areas.add_trace(go.Scatter(y=traces_areas_svg, mode='lines', name=f"{vp_extract.title()}"))
            # line quality
            fig_quality.add_trace(go.Scatter(y=traces_vp_quality, mode='lines', name=f"{vp_extract.title()}"))
            # heatmap
            if(plot_heatmaps and len(traces_heatmaps)):
                fig_heatmap = px.imshow(np.sum(traces_heatmaps, axis=0).reshape(
                    vp_extract.shape), title=f"{vp_extract.title()}",
                    labels=dict(x="longitude", y="latitude", color="VP_Extracts"))
                fig_heatmap.update_layout(LAYOUT)
                fig_heatmap.show()
            # sum
            vp_extract_n_reqs.append(np.sum(traces_n_reqs))
            vp_extract_avg_area.append(np.average(traces_areas_svg))
            vp_extract_quality.append(np.average(traces_vp_quality))

        # line fig reqs areas
        if(plot_lines):
            fig_reqs.update_layout(xaxis_title="user trace", title="req_tiles " + self.title_sufix).show()
            fig_areas.update_layout(xaxis_title="user trace",
                                    title="avg req_tiles view_ratio " + self.title_sufix).show()
            fig_quality.update_layout(xaxis_title="user trace", title="avg quality ratio " + self.title_sufix).show()

        # bar fig vp_extract_n_reqs vp_extract_avg_area
        vp_extract_names = [str(func.title()) for func in vp_extrac_list]
        fig_bar = make_subplots(rows=1, cols=4,  subplot_titles=(
            "req_tiles", "avg req_tiles view_ratio", "avg VP quality_ratio", "score=quality_ratio/(req_tiles*(1-view_ratio)))"), shared_yaxes=True)
        fig_bar.add_trace(go.Bar(y=vp_extract_names, x=vp_extract_n_reqs, orientation='h'), row=1, col=1)
        fig_bar.add_trace(go.Bar(y=vp_extract_names, x=vp_extract_avg_area, orientation='h'), row=1, col=2)
        fig_bar.add_trace(go.Bar(y=vp_extract_names, x=vp_extract_quality, orientation='h'), row=1, col=3)
        vp_extract_score = [vp_extract_quality[i] * (1 / (vp_extract_n_reqs[i] * (1 - vp_extract_avg_area[i])))
                            for i, _ in enumerate(vp_extract_n_reqs)]
        fig_bar.add_trace(go.Bar(y=vp_extract_names, x=vp_extract_score, orientation='h'), row=1, col=4)
        fig_bar.update_layout(width=1500, showlegend=False, title_text=self.title_sufix)
        fig_bar.update_layout(barmode="stack")
        if(plot_bars):
            fig_bar.show()


class VPExtract():
    class Cover(Enum):
        ANY = auto()
        CENTER = auto()
        ONLY20PERC = auto()
        ONLY33PERC = auto()

    def request(self, x, y, z):
        pass

    def title(self):
        pass


class VPExtractTilesRect(VPExtract):
    def __init__(self, t_hor, t_vert, cover: VPExtract.Cover):
        self.t_hor, self.t_vert = t_hor, t_vert
        self.cover = cover
        self.shape = (self.t_vert, self.t_hor)

    def title(self):
        prefix = f'rect_tiles_{self.t_hor}x{self.t_vert}'
        match self.cover:
            case VPExtract.Cover.ANY:
                return f'{prefix}_any'
            case VPExtract.Cover.CENTER:
                return f'{prefix}_center'
            case VPExtract.Cover.ONLY20PERC:
                return f'{prefix}_20perc'
            case VPExtract.Cover.ONLY33PERC:
                return f'{prefix}_33perc'

    def request(self, x, y, z):
        match self.cover:
            case VPExtract.Cover.CENTER:
                return self._request_110radius_center(x, y, z)
            case VPExtract.Cover.ANY:
                return self._request_min_cover(x, y, z, 0.0)
            case VPExtract.Cover.ONLY20PERC:
                return self._request_min_cover(x, y, z, 0.2)
            case VPExtract.Cover.ONLY33PERC:
                return self._request_min_cover(x, y, z, 0.33)

    def _request_min_cover(self, x, y, z, required_cover: float):
        heatmap = np.zeros((self.t_vert, self.t_hor))
        areas_in = []
        vp_quality = 0.0
        vp_threshold = np.deg2rad(110/2)
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                rectan_tile_points, tile_center = points_rectan_tile_cartesian(i, j, self.t_hor, self.t_vert)
                dist = compute_orthodromic_distance([x, y, z], tile_center)
                if dist <= vp_threshold:
                    rectan_tile_polygon = polygon.SphericalPolygon(rectan_tile_points)
                    fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(x, y, z))
                    view_area = rectan_tile_polygon.overlap(fov_polygon)
                    if view_area > required_cover:
                        heatmap[i][j] = 1
                        areas_in.append(view_area)
                        vp_quality += fov_polygon.overlap(rectan_tile_polygon)
        reqs = np.sum(heatmap)
        return reqs, vp_quality, areas_in, heatmap

    def _request_110radius_center(self, x, y, z):
        heatmap = np.zeros((self.t_vert, self.t_hor))
        vp_110_rad_half = np.deg2rad(110/2)
        areas_in = []
        vp_quality = 0.0
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                rectan_tile_points, tile_center = points_rectan_tile_cartesian(i, j, self.t_hor, self.t_vert)
                dist = compute_orthodromic_distance([x, y, z], tile_center)
                if dist <= vp_110_rad_half:
                    heatmap[i][j] = 1
                    rectan_tile_polygon = polygon.SphericalPolygon(rectan_tile_points)
                    fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(x, y, z))
                    view_area = rectan_tile_polygon.overlap(fov_polygon)
                    areas_in.append(view_area)
                    vp_quality += fov_polygon.overlap(rectan_tile_polygon)
        reqs = np.sum(heatmap)
        return reqs, vp_quality, areas_in, heatmap


class VPExtractTilesVoro(VPExtract):
    def __init__(self, spherical_voronoi: SphericalVoronoi, cover: VPExtract.Cover):
        self.spherical_voronoi = spherical_voronoi
        self.cover = cover
        self.shape = (len(spherical_voronoi.points)//6, -1)

    def title(self):
        prefix = f'voroni_tiles_{len(self.spherical_voronoi.points)}'
        match self.cover:
            case VPExtract.Cover.ANY:
                return f'{prefix}_any'
            case VPExtract.Cover.CENTER:
                return f'{prefix}_center'
            case VPExtract.Cover.ONLY20PERC:
                return f'{prefix}_20perc'
            case VPExtract.Cover.ONLY33PERC:
                return f'{prefix}_33perc'

    def request(self, x, y, z):
        match self.cover:
            case VPExtract.Cover.CENTER:
                return self._request_110radius_center(x, y, z)
            case VPExtract.Cover.ANY:
                return self._request_min_cover(x, y, z, 0)
            case VPExtract.Cover.ONLY20PERC:
                return self._request_min_cover(x, y, z, 0.2)
            case VPExtract.Cover.ONLY33PERC:
                return self._request_min_cover(x, y, z, 0.33)

    def _request_110radius_center(self, x, y, z):
        vp_110_rad_half = np.deg2rad(110/2)
        reqs = 0
        areas_in = []
        vp_quality = 0.0
        heatmap = np.zeros(len(self.spherical_voronoi.points))
        for index, region in enumerate(self.spherical_voronoi.regions):
            dist = compute_orthodromic_distance([x, y, z], self.spherical_voronoi.points[index])
            if dist <= vp_110_rad_half:
                reqs += 1
                heatmap[index] += 1
                voroni_tile_polygon = polygon.SphericalPolygon(self.spherical_voronoi.vertices[region])
                fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(x, y, z))
                view_area = voroni_tile_polygon.overlap(fov_polygon)
                areas_in.append(view_area)
                vp_quality += fov_polygon.overlap(voroni_tile_polygon)
        return reqs, vp_quality, areas_in, heatmap

    def _request_min_cover(self, x, y, z, required_cover: float):
        reqs = 0
        areas_in = []
        vp_quality = 0.0
        heatmap = np.zeros(len(self.spherical_voronoi.points))
        for index, region in enumerate(self.spherical_voronoi.regions):
            voroni_tile_polygon = polygon.SphericalPolygon(self.spherical_voronoi.vertices[region])
            fov_polygon = polygon.SphericalPolygon(points_fov_cartesian(x, y, z))
            view_area = voroni_tile_polygon.overlap(fov_polygon)
            if view_area > required_cover:
                reqs += 1
                heatmap[index] += 1
                # view_area = 1 if view_area > 1 else view_area  # fixed with compute_orthodromic_distance
                areas_in.append(view_area)
                vp_quality += fov_polygon.overlap(voroni_tile_polygon)
        return reqs, vp_quality, areas_in, heatmap


VP_EXTRACTS_VORO = [
    VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.CENTER),
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.CENTER),
    VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ANY),
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ANY),
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ONLY20PERC),
    VPExtractTilesVoro(VORONOI_14P, VPExtract.Cover.ONLY33PERC),
    VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ONLY20PERC),
    VPExtractTilesVoro(VORONOI_24P, VPExtract.Cover.ONLY33PERC),
]
VP_EXTRACTS_RECT = [
    VPExtractTilesRect(6, 4, VPExtract.Cover.ONLY33PERC),
    VPExtractTilesRect(6, 4, VPExtract.Cover.ONLY20PERC),
    VPExtractTilesRect(6, 4, VPExtract.Cover.CENTER),
]

VPEXTRACT_METHODS = [* VP_EXTRACTS_VORO, *VP_EXTRACTS_RECT]
# %%
