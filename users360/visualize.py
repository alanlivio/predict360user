import pathlib
from os.path import exists

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats
import swifter
from colour import Color
from numpy.random import randint
from plotly.subplots import make_subplots
from scipy.spatial import SphericalVoronoi, geometric_slerp

from .data import *
from .utils.tileset import *
from .utils.tileset_voro import *


class VizSphere():

    def __init__(self, tileset=TILESET_DEFAULT):
        if isinstance(tileset, TileSetVoro):
            self.data = self._data_sphere_voro(tileset.voro)
        else:
            self.data = self._data_sphere_tiled(tileset.t_ver, tileset.t_hor)
        self.title = "trajectory"

    def _data_sphere_surface(self) -> list:
        # https://community.plotly.com/t/3d-surface-bug-a-custom-colorscale-defined-with-rgba-values-ignores-the-alpha-specification/30809
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        colorscale = [[0, "rgba(200, 0, 0, 0.1)"], [1.0, "rgba(255, 0, 0, 0.1)"]]
        surface = go.Surface(x=np.outer(np.cos(theta), np.sin(phi)) * 0.98,
                             y=np.outer(np.sin(theta), np.sin(phi)) * 0.98,
                             z=np.outer(np.ones(100), np.cos(phi)) * 0.98,
                             colorscale=colorscale,
                             hoverinfo='skip',
                             showlegend=False, showscale=False)
        return [surface]

    def _data_sphere_voro(self, sphere_voro: SphericalVoronoi) -> list:
        data = self._data_sphere_surface()
        # tile edges
        for region in sphere_voro.regions:
            n = len(region)
            t = np.linspace(0, 1, 100)
            for index in range(n):
                start = sphere_voro.vertices[region][index]
                end = sphere_voro.vertices[region][(index + 1) % n]
                result = np.array(geometric_slerp(start, end, t))
                edge = go.Scatter3d(x=result[..., 0],
                                    y=result[..., 1],
                                    z=result[..., 2],
                                    hoverinfo='skip',
                                    mode='lines', line={'width': 2, 'color': 'gray'}, showlegend=False)
                data.append(edge)
        return data

    def _data_sphere_tiled(self, t_ver, t_hor) -> list:
        data = self._data_sphere_surface()
        # tiles edges
        for row in range(t_ver):
            for col in range(t_hor):
                tpoints = tile_points(t_ver, t_hor, row, col)
                n = len(tpoints)
                t = np.linspace(0, 1, 100)
                for index in range(n):
                    start = tpoints[index]
                    end = tpoints[(index + 1) % n]
                    result = np.array(geometric_slerp(start, end, t))
                    edge = go.Scatter3d(x=result[..., 0],
                                        y=result[..., 1],
                                        z=result[..., 2],
                                        hoverinfo='skip',
                                        mode='lines',
                                        line={'width': 2, 'color': 'gray'})
                    data.append(edge)
        return data

    def _add_polygon_lines(self, points):
        # gen points
        n = len(points)
        t = np.linspace(0, 1, 100)
        colours = [f'rgb({randint(0,256)}, {randint(0,256)}, {randint(0,256)})' for _ in range(len(points))]
        gens = go.Scatter3d(x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            hoverinfo='skip',
                            mode='markers',
                            marker={'size': 4, 'color': colours})
        self.data.append(gens)

        # edges
        for index in range(n):
            start = points[index]
            end = points[(index + 1) % n]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0],
                                y=result[..., 1],
                                z=result[..., 2],
                                hoverinfo='skip',
                                mode='lines',
                                line={'width': 4, 'color': 'blue'})
            self.data.append(edge)

    def add_polygon(self, polygon: polygon.SphericalPolygon):
        assert polygon.points.shape[1] == 3
        self._add_polygon_lines([point for point in polygon.points])

    def add_polygon_from_points(self, points):
        assert points.shape[1] == 3
        self._add_polygon_lines(points)

    def add_polygon_from_tile_row_col(self, t_ver, t_hor, row, col):
        points = tile_points(t_ver, t_hor, row, col)
        self._add_polygon_lines(points)

    def add_trace_and_fov(self, trace):
        self.data.append(go.Scatter3d(x=[trace[0]],
                                      y=[trace[1]],
                                      z=[trace[2]],
                                      mode='markers',
                                      marker={'size': 4, 'color': 'red'}))
        points = fov_points(trace)
        self._add_polygon_lines(points)

    def add_trajectory(self, trajectory):
        # start, end colors
        start_c = '#b2d8ff'
        end_c = '#00264c'

        # start, end marks
        self.data.append(go.Scatter3d(x=[trajectory[0][0]],
                                      y=[trajectory[0][1]],
                                      z=[trajectory[0][2]],
                                      mode='markers',
                                      marker={'size': 4, 'color': start_c}))
        self.data.append(go.Scatter3d(x=[trajectory[-1][0]],
                                      y=[trajectory[-1][1]],
                                      z=[trajectory[-1][2]],
                                      mode='markers',
                                      marker={'size': 4, 'color': end_c}))

        # edges
        n = len(trajectory)
        t = np.linspace(0, 1, 100)
        colors = [x.hex for x in list(Color(start_c).range_to(Color(end_c), n))]
        for index in range(n - 1):
            start = trajectory[index]
            end = trajectory[(index + 1)]
            result = np.array(geometric_slerp(start, end, t))
            edge = go.Scatter3d(x=result[..., 0],
                                y=result[..., 1],
                                z=result[..., 2],
                                hovertext=f'trajectory[{index}]',
                                hoverinfo='text',
                                mode='lines',
                                line={'width': 5, 'color': colors[index]})
            self.data.append(edge)

    def show(self):
        fig = go.Figure(data=self.data)
        fig.update_layout(width=800, showlegend=False, title_text=self.title)
        fig.show()


def _show_or_save_to_html(fig, title, to_html):
    if to_html:
        output_folder = pathlib.Path(__file__).parent.parent / 'data'
        plotly.offline.plot(fig, filename=f'{output_folder}/{title}.html', auto_open=False)
    else:
        fig.update_layout(width=800, showlegend=False, title_text=title)
        fig.show()


def show_fov(trace, tileset=TILESET_DEFAULT, to_html=False):
    assert len(trace) == 3  # cartesian

    # subplot two figures
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])

    # sphere
    sphere = VizSphere(tileset)
    sphere.add_trace_and_fov(trace)
    for t in sphere.data:
        fig.append_trace(t, row=1, col=1)

    # heatmap
    heatmap = tileset.request(trace)
    if isinstance(tileset, TileSetVoro):
        heatmap = np.reshape(heatmap, tileset.shape)
    x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
    y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
    erp_heatmap = px.imshow(heatmap, text_auto=True, x=x, y=y)
    for t in erp_heatmap["data"]:
        fig.append_trace(t, row=1, col=2)
    if isinstance(tileset, TileSet):
        fig.update_yaxes(autorange="reversed")  # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian

    # show or save html
    title = f"trace_[{trace[0]:.2},{trace[1]:.2},{trace[2]:.2}]_{tileset.prefix}"
    _show_or_save_to_html(fig, title, to_html)


def show_trajects(df: pd.DataFrame, tileset=TILESET_DEFAULT, to_html=False):
    assert (not df.empty)

    # subplot two figures
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])

    # sphere
    sphere = VizSphere(tileset)
    df['traces'].apply(lambda traces: sphere.add_trajectory(traces))
    for d in sphere.data:
        fig.append_trace(d, row=1, col=1)
    # heatmap
    # TODO: calcuate if hmps is not in df
    if 'hmps' in df:
        hmp_sums = df['hmps'].apply(lambda traces: np.sum(traces, axis=0))
        if isinstance(tileset, TileSetVoro):
            hmp_sums = np.reshape(hmp_sums, tileset.shape)
        heatmap = np.sum(hmp_sums, axis=0)
        x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
        y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
        erp_heatmap = px.imshow(heatmap, text_auto=True, x=x, y=y)
        for data in erp_heatmap["data"]:
            fig.append_trace(data, row=1, col=2)
        if isinstance(tileset, TileSet):
            # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
            fig.update_yaxes(autorange="reversed")

    # show or save html
    title = f"trajects_{str(df.shape[0])}_{tileset.title}"
    _show_or_save_to_html(fig, title, to_html)


def calc_trajects_tileset_metrics(tileset_l: list[TileSetIF], n_trajects=None):
    assert (not get_df_trajects().empty)
    if n_trajects:
        df = get_df_trajects()[:n_trajects]
    else:
        df = get_df_trajects()

    def create_tsdf(ts_idx):
        tileset = tileset_l[ts_idx]
        def f_trace(trace):
            heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
            return (int(np.sum(heatmap)), vp_quality, area_out)
        f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
        tmpdf = pd.DataFrame(df['traces'].swifter.apply(f_traject))
        tmpdf.columns = [tileset.title]
        return tmpdf
    df_tileset_metrics = pd.concat(map(create_tsdf, range(len(tileset_l))), axis=1)
    Data.singleton().df_tileset_metrics = df_tileset_metrics


def show_trajects_tileset_metrics():
    df_tileset_metrics = Data.singleton().df_tileset_metrics
    assert (not df_tileset_metrics.empty)

    # calc tileset metrics
    f_traject_reqs = lambda traces: np.sum(traces[:, 0])
    f_traject_qlt = lambda traces: np.mean(traces[:, 1])
    f_traject_lost = lambda traces: np.mean(traces[:, 2])
    data = {'tileset': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
    for name in df_tileset_metrics.columns:
        dfts = df_tileset_metrics[name]
        data['tileset'].append(name)
        data['avg_reqs'].append(dfts.apply(f_traject_reqs).mean())
        data['avg_qlt'].append(dfts.apply(f_traject_qlt).mean())
        data['avg_lost'].append(dfts.apply(f_traject_lost).mean())
        data['score'] = data['avg_qlt'][-1] / data['avg_lost'][-1]
    df = pd.DataFrame(data)

    fig = make_subplots(rows=1, cols=4, subplot_titles=("avg_reqs", "avg_lost",
                        "avg_qlt", "score=avg_qlt/avg_lost"), shared_yaxes=True)
    y = df_tileset_metrics.columns
    trace = go.Bar(y=y, x=df['avg_reqs'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=1)
    trace = go.Bar(y=y, x=df['avg_lost'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=2)
    trace = go.Bar(y=y, x=df['avg_qlt'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=3)
    trace = go.Bar(y=y, x=df['score'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=4)
    fig.update_layout(width=1500, showlegend=False, barmode="stack",)
    fig.show()
