import pathlib
from .vpextract import *
from head_motion_prediction.Utils import *
from scipy.spatial import SphericalVoronoi, geometric_slerp
import numpy as np
from numpy.typing import NDArray
import plotly
import plotly.express as px
import plotly.graph_objs as go


def _sphere_data_surface():
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(theta), np.sin(phi))*0.98
    y = np.outer(np.sin(theta), np.sin(phi))*0.98
    z = np.outer(np.ones(100), np.cos(phi))*0.98
    # https://community.plotly.com/t/3d-surface-bug-a-custom-colorscale-defined-with-rgba-values-ignores-the-alpha-specification/30809
    colorscale = [[0, "rgba(200, 0, 0, 0.1)"], [1.0, "rgba(255, 0, 0, 0.1)"]]
    return [go.Surface(x=x, y=y, z=z, colorscale=colorscale, showlegend=False, showscale=False)]


def _sphere_data_voro(sphere_voro: SphericalVoronoi, with_generators=False):
    data = _sphere_data_surface()
    # generator points
    if with_generators:
        gens = go.Scatter3d(x=sphere_voro.points[:, 0], y=sphere_voro.points[:, 1], z=sphere_voro.points[:, 2],
                            mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='voron center')
        data.append(gens)
    # edges
    for region in sphere_voro.regions:
        n = len(region)
        t = np.linspace(0, 1, 100)
        for i in range(n):
            start = sphere_voro.vertices[region][i]
            end = sphere_voro.vertices[region][(i + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
            data.append(edge)
    return data


def _sphere_data_rect_tiles(t_hor, t_vert):
    data = _sphere_data_surface()
    for i in range(t_hor+1):
        for j in range(t_vert+1):
            # -- add rect tiles edges
            rect_tile_points, _ = points_rect_tile_cartesian(i, j, t_hor, t_vert)
            n = len(rect_tile_points)
            t = np.linspace(0, 1, 100)
            for index in range(n):
                start = rect_tile_points[index]
                end = rect_tile_points[(index + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                    'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
                data.append(edge)
    return data


class PlotVP():

    def __init__(self, trace, title_sufix=""):
        assert len(trace) == 3  # cartesian
        self.title = f"1_trace_{title_sufix}"
        self.trace = trace
        self.output_folder = pathlib.Path(__file__).parent.parent/'output'

    def sphere_vp_trace(self, vpextract: VPExtract, to_html=False):
        data, title = [], ""
        if isinstance(vpextract, VPExtractTilesRect):
            data = _sphere_data_rect_tiles(vpextract.t_hor, vpextract.t_vert)
        elif isinstance(vpextract, VPExtractTilesVoro):
            data = _sphere_data_voro(vpextract.sphere_voro)
        # trace
        data.append(go.Scatter3d(x=[self.trace[0]], y=[self.trace[1]], z=[self.trace[2]],
                                 mode='markers', marker={'size': 5, 'opacity': 1.0, 'color': 'red'}, name='center'))
        # vp
        fov_polygon = points_fov_cartesian(self.trace)
        n = len(fov_polygon)
        gens = go.Scatter3d(x=fov_polygon[:, 0], y=fov_polygon[:, 1], z=fov_polygon[:, 2],
                            mode='markers', marker={'size': 5, 'opacity': 1.0,
                                                    'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(fov_polygon))]},
                            name='fov corner', showlegend=False)
        data.append(gens)
        t = np.linspace(0, 1, 100)
        for index in range(n):
            start = fov_polygon[index]
            end = fov_polygon[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 5, 'color': 'blue'}, name='vp edge', showlegend=False)
            data.append(edge)
        heatmap, _, _ = vpextract.request(self.trace)
        title = f"{self.title} {vpextract.title_with_sum_heatmaps([heatmap])}"
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{self.output_folder}/{title}.html', auto_open=False)
        else:
            fig.show()

    def erp_heatmap(self, vpextract: VPExtract, to_html=False):
        heatmap, _, _, = vpextract.request(self.trace)
        if isinstance(vpextract, VPExtractTilesVoro):
            heatmap = np.reshape(heatmap, vpextract.shape)
        fig = px.imshow(heatmap, labels=dict(x="longitude", y="latitude", color="requests"))
        title = f"{self.title} {vpextract.title_with_sum_heatmaps([heatmap])}"
        fig.update_layout(layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{self.output_folder}/{title}.html', auto_open=False)
        else:
            fig.show()


class PlotTraces():

    def __init__(self, traces: NDArray, title_sufix=""):
        assert traces.shape[1] == 3  # check if cartesian
        self.traces = traces
        print("PlotTraces.shape is " + str(traces.shape))
        self.title = f"{str(len(traces))}_traces_{title_sufix}"
        self.output_folder = pathlib.Path(__file__).parent.parent/'output'

    def _sphere_data_add_user(self, data, vpinfo):
        trajc = go.Scatter3d(x=self.traces[:, 0],
                             y=self.traces[:, 1],
                             z=self.traces[:, 2],
                             hovertemplate="<b>requested tiles=%{text}</b>",
                             text=vpinfo,
                             mode='lines', line={'width': 3, 'color': 'blue'}, name='trajectory', showlegend=False)
        data.append(trajc)

    def sphere(self, vpextract: VPExtract, to_html=False):
        data, title = [], ""
        if isinstance(vpextract, VPExtractTilesRect):
            data = _sphere_data_rect_tiles(vpextract.t_hor, vpextract.t_vert)
            title = f"{self.title} rect{vpextract.t_hor}x{vpextract.t_vert}"
        elif isinstance(vpextract, VPExtractTilesVoro):
            data = _sphere_data_voro(vpextract.sphere_voro)
            title = f"{self.title} voro{len(vpextract.sphere_voro.points)}"
        vpinfo = [np.sum(vpextract.request(trace)[0]) for trace in self.traces]
        self._sphere_data_add_user(data, vpinfo)
        fig = go.Figure(data=data, layout=layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{self.output_folder}/{title}.html', auto_open=False)
        else:
            fig.show()

    def erp_heatmap(self, vpextract: VPExtract, to_html=False):
        heatmaps = []
        for trace in self.traces:
            heatmap, _, _, = vpextract.request(trace)
            heatmaps.append(heatmap)
        if isinstance(vpextract, VPExtractTilesVoro):
            heatmap = np.reshape(heatmaps, vpextract.shape)
        fig = px.imshow(np.sum(heatmaps, axis=0), labels=dict(x="longitude", y="latitude", color="requests"))
        title = f"{self.title} {vpextract.title_with_sum_heatmaps(heatmaps)}"
        fig.update_layout(layout_with_title(title))
        if to_html:
            plotly.offline.plot(fig, filename=f'{self.output_folder}/{title}.html', auto_open=False)
        else:
            fig.show()
