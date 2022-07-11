import pathlib
import numpy as np
import pandas as pd
from .tileset import *
from .tileset_voro import *
from head_motion_prediction.Utils import *
from scipy.spatial import SphericalVoronoi, geometric_slerp
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class Projection():

    def __init__(self, tileset=TileSet.default()):
        if isinstance(tileset, TileSetVoro):
            self.data = self._init_data_sphere_voro(tileset.voro)
        else:
            self.data = self._init_data_sphere_tiles(tileset.t_ver, tileset.t_hor)
        self.title = f"project"

    def _init_data_sphere(self) -> list:
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(theta), np.sin(phi)) * 0.98
        y = np.outer(np.sin(theta), np.sin(phi)) * 0.98
        z = np.outer(np.ones(100), np.cos(phi)) * 0.98
        # https://community.plotly.com/t/3d-surface-bug-a-custom-colorscale-defined-with-rgba-values-ignores-the-alpha-specification/30809
        colorscale = [[0, "rgba(200, 0, 0, 0.1)"], [1.0, "rgba(255, 0, 0, 0.1)"]]
        return [go.Surface(x=x, y=y, z=z, colorscale=colorscale, showlegend=False, showscale=False)]

    def _init_data_sphere_voro(self, sphere_voro: SphericalVoronoi, with_generators=False) -> list:
        data = self._init_data_sphere()
        # generator points
        if with_generators:
            gens = go.Scatter3d(x=sphere_voro.points[:, 0],
                                y=sphere_voro.points[:, 1],
                                z=sphere_voro.points[:, 2],
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
                edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2],
                                    mode='lines', line={'width': 2, 'color': 'gray'}, showlegend=False)
                data.append(edge)
        return data

    def _init_data_sphere_tiles(self, t_ver, t_hor) -> list:
        data = self._init_data_sphere()
        for row in range(t_ver):
            for col in range(t_hor):
                # -- add tiles edges
                tile_points = TileSet.tile_points(t_ver, t_hor, row, col)
                n = len(tile_points)
                t = np.linspace(0, 1, 100)
                for index in range(n):
                    start = tile_points[index]
                    end = tile_points[(index + 1) % n]
                    result = np.array(geometric_slerp(start, end, t))
                    edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                        'width': 2, 'color': 'gray'}, name='region edge', showlegend=False)
                    data.append(edge)
        return data

    def add_polygon(self, polygon: polygon.SphericalPolygon):
        self._add_points([point for point in polygon.points])  # polygon.points is a generator

    def add_polygon_as_points(self, points):
        assert points.shape[1] == 3  # check if cartesian
        self._add_points(points)

    def add_polygon_as_row_col_tile(self, t_ver, t_hor, row, col):
        points = TileSet.tile_points(t_ver, t_hor, row, col)
        self._add_points(points)

    def add_polygon_from_trace(self, trace):
        points = FOV.points(trace)
        self._add_points(points)


    def add_trace(self, trace):
        self.data.append(go.Scatter3d(x=[trace[0]], y=[trace[1]], z=[trace[2]],
                                mode='markers', marker={'size': 5, 'opacity': 1.0, 'color': 'red'}))

    def add_vp(self, trace):
        points_fov = FOV.points(trace)
        n = len(points_fov)
        gens = go.Scatter3d(x=points_fov[:, 0],
                            y=points_fov[:, 1],
                            z=points_fov[:, 2],
                            mode='markers', marker={'size': 5, 'opacity': 1.0, 'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(points_fov))]}, name='fov corner', showlegend=False)
        self.data.append(gens)
        t = np.linspace(0, 1, 100)
        for index in range(n):
            start = points_fov[index]
            end = points_fov[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0], y=result[..., 1], z=result[..., 2], mode='lines', line={
                                'width': 5, 'color': 'blue'}, name='vp edge', showlegend=False)
            self.data.append(edge)

    def _add_points(self, points):
        # points
        n = len(points)
        t = np.linspace(0, 1, 100)
        gens = go.Scatter3d(x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            mode='markers', showlegend=False, marker={'size': 5, 'opacity': 1.0, 'color': [f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(len(points))]})
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


def show_fov(trace, tileset=TileSet.default(), title_sufix="", to_html=False):
    assert len(trace) == 3  # cartesian

    # default Projection
    project = Projection(tileset)
    project.add_trace(trace)
    project.add_vp(trace)
    
    # erp heatmap
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])
    for t in project.data:
        fig.append_trace(t, row=1, col=1)
    
    heatmap = tileset.request(trace)[0]
    if isinstance(tileset, TileSetVoro):
        heatmap = np.reshape(heatmap, tileset.shape)
    erp_heatmap = px.imshow(heatmap, text_auto=True,
                            x=[str(x) for x in range(1, heatmap.shape[1] + 1)],
                            y=[str(y) for y in range(1, heatmap.shape[0] + 1)])
    for t in erp_heatmap["data"]:
        fig.append_trace(t, row=1, col=2)

    title = f"1_trace_{title_sufix}"
    title = f"{title} {tileset.title_with_sum_heatmaps([heatmap])}"
    if isinstance(tileset, TileSet):
        # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(width=800, showlegend=False, title_text=title)
    if to_html:
        output_folder = pathlib.Path(__file__).parent.parent / 'output'
        plotly.offline.plot(fig, filename=f'{output_folder}/{title}.html', auto_open=False)
    else:
        fig.show()


def show_trajects(df: pd.DataFrame, tileset=TileSet.default(), title_sufix="", to_html=False):
    
    # subplot two figures https://stackoverflow.com/questions/67291178/how-to-create-subplots-using-plotly-express
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])
    
    # default Projection
    project = Projection(tileset)
    data = project.data
    
    # add traces to row=1,col=1
    def f_traces (traces):
        scatter = go.Scatter3d(x=traces[:,0], y=traces[:,1],z=traces[:,2],
            mode='lines', line={'width': 3, 'color': 'blue'}, showlegend=False)
        data.append(scatter)
    df['traces'].apply(f_traces)
    for trace in data:
        fig.append_trace(trace, row=1, col=1)
    
    # add erp_heatmap row=1, col=2
    hmp_sums = df['hmps'].apply(lambda traces: np.sum(traces, axis=0))
    if isinstance(tileset, TileSetVoro):
        hmp_sums = np.reshape(hmp_sums, tileset.shape)
    hmp_final = np.sum(hmp_sums, axis=0)
    erp_heatmap = px.imshow(hmp_final, text_auto=True,
                        x=[str(x) for x in range(1, hmp_final.shape[1] + 1)],
                        y=[str(y) for y in range(1, hmp_final.shape[0] + 1)])
    for data in erp_heatmap["data"]:
        fig.append_trace(data, row=1, col=2)
    
    hmp_total =  np.sum(np.sum(hmp_final, axis=0))
    title = f"{str(df.shape[0])}_trajcectories_{hmp_total}_reqs"
    if isinstance(tileset, TileSet):
        # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(width=800, showlegend=False, title_text=title)
    if to_html:
        output_folder = pathlib.Path(__file__).parent.parent / 'output'
        plotly.offline.plot(fig, filename=f'{output_folder}/{title}.html', auto_open=False)
    else:
        fig.show()
