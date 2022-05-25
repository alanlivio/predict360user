import pathlib
from .tiles import *
from .tiles_voro import *
from head_motion_prediction.Utils import *
from scipy.spatial import SphericalVoronoi, geometric_slerp
import numpy as np
from numpy.typing import NDArray
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def _sphere_data_init() -> list:
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(theta), np.sin(phi)) * 0.98
    y = np.outer(np.sin(theta), np.sin(phi)) * 0.98
    z = np.outer(np.ones(100), np.cos(phi)) * 0.98
    # https://community.plotly.com/t/3d-surface-bug-a-custom-colorscale-defined-with-rgba-values-ignores-the-alpha-specification/30809
    colorscale = [[0, "rgba(200, 0, 0, 0.1)"], [1.0, "rgba(255, 0, 0, 0.1)"]]
    return [go.Surface(x=x, y=y, z=z, colorscale=colorscale, showlegend=False, showscale=False)]


def _sphere_data_voro(sphere_voro: SphericalVoronoi, with_generators=False) -> list:
    data = _sphere_data_init()
    # generator points
    if with_generators:
        gens = go.Scatter3d(x=sphere_voro.points[:, 0], y=sphere_voro.points[:, 1], z=sphere_voro.points[:, 2],
                            mode='markers', marker={'size': 1, 'opacity': 1.0, 'color': 'blue'}, name='voron center')
        data.append(gens)
    # edges
    for region in sphere_voro.regions:
        n = len(region)
        t = np.linspace(0, 1, 100)
        for index in range(n):
            start = sphere_voro.vertices[region][index]
            end = sphere_voro.vertices[region][(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
            data.append(edge)
    return data


def _sphere_data_tiles(t_ver, t_hor) -> list:
    data = _sphere_data_init()
    for row in range(t_ver):
        for col in range(t_hor):
            # -- add tiles edges
            tile_points = Tiles.tile_points(t_ver, t_hor, row, col)
            n = len(tile_points)
            t = np.linspace(0, 1, 100)
            for index in range(n):
                start = tile_points[index]
                end = tile_points[(index + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                    'width': 5, 'color': 'black'}, name='region edge', showlegend=False)
                data.append(edge)
    return data


class ProjectPolys():

    def __init__(self, tiles=None):
        if isinstance(tiles, Tiles):
            self.data = _sphere_data_tiles(tiles.t_ver, tiles.t_hor)
        elif isinstance(tiles, TilesVoro):
            self.data = _sphere_data_voro(tiles.voro)
        else:
            self.data = _sphere_data_init()
        self.title = f"polygon"

    def add_polygon(self, polygon: polygon.SphericalPolygon):
        self._add_points([point for point in polygon.points])  # polygon.points is a generator

    def add_polygon_as_points(self, points):
        assert points.shape[1] == 3  # check if cartesian
        self._add_points(points)

    def add_polygon_as_row_col_tile(self, t_ver, t_hor, row, col):
        points = Tiles.tile_points(t_ver, t_hor, row, col)
        self._add_points(points)

    def add_polygon_from_trace(self, trace):
        points = FOV.points(trace)
        self._add_points(points)

    def _add_points(self, points):
        # points
        n = len(points)
        t = np.linspace(0, 1, 100)
        gens = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', showlegend=False,
                            marker={'size': 5, 'opacity': 1.0, 'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(points))]})
        self.data.append(gens)
        for index in range(n):
            start = points[index]
            end = points[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 5, 'color': 'blue'}, name='vp edge', showlegend=False)
            self.data.append(edge)

    def show(self):
        fig = go.Figure(data=self.data)
        fig.update_layout(width=800, showlegend=False, title_text=self.title)
        fig.show()


class ProjectFOV():

    def __init__(self, trace, tiles=Tiles.default(), title_sufix=""):
        assert len(trace) == 3  # cartesian
        self.title = f"1_trace_{title_sufix}"
        self.trace = trace
        self.output_folder = pathlib.Path(__file__).parent.parent / 'output'
        self.data = []
        self.tiles = tiles
        if isinstance(self.tiles, Tiles):
            self.data = _sphere_data_tiles(self.tiles.t_ver, tiles.t_hor)
        elif isinstance(self.tiles, TilesVoro):
            self.data = _sphere_data_voro(self.tiles.voro)

    def show(self, to_html=False):
        # trace
        self.data.append(go.Scatter3d(x=[self.trace[0]], y=[self.trace[1]], z=[self.trace[2]],
                         mode='markers', marker={'size': 5, 'opacity': 1.0, 'color': 'red'}))

        # vp
        points_fov = FOV.points(self.trace)
        n = len(points_fov)
        gens = go.Scatter3d(x=points_fov[:, 0], y=points_fov[:, 1], z=points_fov[:, 2], mode='markers',
                            marker={'size': 5, 'opacity': 1.0, 'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(points_fov))]}, name='fov corner', showlegend=False)
        self.data.append(gens)
        t = np.linspace(0, 1, 100)
        for index in range(n):
            start = points_fov[index]
            end = points_fov[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 5, 'color': 'blue'}, name='vp edge', showlegend=False)
            self.data.append(edge)

        # erp heatmap
        heatmap, _, _ = self.tiles.request(self.trace)

        # subplot two figures https://stackoverflow.com/questions/67291178/how-to-create-subplots-using-plotly-express
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])
        for trace in self.data:
            fig.append_trace(trace, row=1, col=1)
        if isinstance(self.tiles, TilesVoro):
            heatmap = np.reshape(heatmap, self.tiles.shape)
        erp_heatmap = px.imshow(heatmap, text_auto=True,
                                x=[str(x) for x in range(1, heatmap.shape[1] + 1)],
                                y=[str(y) for y in range(1, heatmap.shape[0] + 1)])
        for trace in erp_heatmap["data"]:
            fig.append_trace(trace, row=1, col=2)

        title = f"{self.title} {self.tiles.title_with_sum_heatmaps([heatmap])}"
        if isinstance(self.tiles, Tiles):
            # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
            fig.update_yaxes(autorange="reversed")
        fig.update_layout(width=800, showlegend=False, title_text=title)
        if to_html:
            plotly.offline.plot(fig, filename=f'{self.output_folder}/{title}.html', auto_open=False)
        else:
            fig.show()


class ProjectTraces():

    def __init__(self, traces: NDArray, tiles=Tiles.default(), title_sufix=""):
        assert traces.shape[1] == 3  # check if cartesian
        self.traces = traces
        print("ProjectTraces.shape is " + str(traces.shape))
        self.title = f"{str(len(traces))}_traces_{title_sufix}"
        self.output_folder = pathlib.Path(__file__).parent.parent / 'output'
        self.tiles = tiles
        self.data = []
        if isinstance(self.tiles, Tiles):
            self.data = _sphere_data_tiles(self.tiles.t_ver, self.tiles.t_hor)
        elif isinstance(self.tiles, TilesVoro):
            self.data = _sphere_data_voro(self.tiles.voro)

    def _sphere_data_add_user(self, vpinfo):
        trajc = go.Scatter3d(x=self.traces[:, 0],
                             y=self.traces[:, 1],
                             z=self.traces[:, 2],
                             hovertemplate="<b>requested tiles=%{text}</b>",
                             text=vpinfo,
                             mode='lines', line={'width': 3, 'color': 'blue'}, name='trajectory', showlegend=False)
        self.data.append(trajc)

    def show(self, to_html=False):
        # erp heatmap
        heatmaps = []
        for trace in self.traces:
            heatmap, _, _, = self.tiles.request(trace)
            heatmaps.append(heatmap)

        # traces vp info on mouseover
        vpinfo = [np.sum(heatmap) for heatmap in heatmaps]
        self._sphere_data_add_user(vpinfo)

        # erp heatmap image
        heatmap_sum = np.sum(heatmaps, axis=0)
        if isinstance(self.tiles, TilesVoro):
            heatmap_sum = np.reshape(heatmap_sum, self.tiles.shape)
        erp_heatmap = px.imshow(heatmap_sum, text_auto=True,
                                x=[str(x) for x in range(1, heatmap_sum.shape[1] + 1)],
                                y=[str(y) for y in range(1, heatmap_sum.shape[0] + 1)])

        # subplot two figures https://stackoverflow.com/questions/67291178/how-to-create-subplots-using-plotly-express
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])
        for trace in self.data:
            fig.append_trace(trace, row=1, col=1)
        for trace in erp_heatmap["data"]:
            fig.append_trace(trace, row=1, col=2)
        title = f"{self.title} {self.tiles.title_with_sum_heatmaps(heatmaps)}"
        if isinstance(self.tiles, Tiles):
            # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
            fig.update_yaxes(autorange="reversed")
        fig.update_layout(width=800, showlegend=False, title_text=title)
        if to_html:
            plotly.offline.plot(fig, filename=f'{self.output_folder}/{title}.html', auto_open=False)
        else:
            fig.show()
