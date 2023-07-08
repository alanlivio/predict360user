import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from colour import Color
from numpy.random import randint
from plotly.subplots import make_subplots
from scipy.spatial import SphericalVoronoi, geometric_slerp
from spherical_geometry.polygon import SphericalPolygon

from predict360user.tileset import (TILESET_DEFAULT, TileSet, TileSetVoro,
                                    tile_points)
from predict360user.utils import *


class Plot360():

  def __init__(self, tileset=TILESET_DEFAULT) -> None:
    self.tileset = tileset
    if isinstance(self.tileset, TileSetVoro):
      self.data = self._data_self_voro(self.tileset.voro)
    else:
      self.data = self._data_self_tiled(self.tileset.t_ver, self.tileset.t_hor)
    self.title = 'tracesory'

  def _data_self_surface(self) -> list:
    # https://community.plotly.com/t/3d-surface-bug-a-custom-colorscale-defined-with-rgba-values-ignores-the-alpha-specification/30809
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    colorscale = [[0, 'rgba(200, 0, 0, 0.1)'], [1.0, 'rgba(255, 0, 0, 0.1)']]
    surface = go.Surface(x=np.outer(np.cos(theta), np.sin(phi)) * 0.98,
                         y=np.outer(np.sin(theta), np.sin(phi)) * 0.98,
                         z=np.outer(np.ones(100), np.cos(phi)) * 0.98,
                         colorscale=colorscale,
                         hoverinfo='skip',
                         showlegend=False,
                         showscale=False)
    return [surface]

  def _data_self_voro(self, self_voro: SphericalVoronoi) -> list:
    data = self._data_self_surface()
    # tile edges
    for region in self_voro.regions:
      n = len(region)
      t = np.linspace(0, 1, 100)
      for index in range(n):
        start = self_voro.vertices[region][index]
        end = self_voro.vertices[region][(index + 1) % n]
        result = np.array(geometric_slerp(start, end, t))
        edge = go.Scatter3d(x=result[..., 0],
                            y=result[..., 1],
                            z=result[..., 2],
                            hoverinfo='skip',
                            mode='lines',
                            line={
                                'width': 2,
                                'color': 'gray'
                            },
                            showlegend=False)
        data.append(edge)
    return data

  def _data_self_tiled(self, t_ver, t_hor) -> list:
    data = self._data_self_surface()
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
                              line={
                                  'width': 2,
                                  'color': 'gray'
                              })
          data.append(edge)
    return data

  def _add_polygon_lines(self, points) -> None:
    # gen points
    n = len(points)
    t = np.linspace(0, 1, 100)
    colours = [
        f'rgb({randint(0,256)}, {randint(0,256)}, {randint(0,256)})'
        for _ in range(len(points))
    ]
    gens = go.Scatter3d(x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        hoverinfo='skip',
                        mode='markers',
                        marker={
                            'size': 7,
                            'color': colours
                        })
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
                          line={
                              'width': 7,
                              'color': 'blue'
                          })
      self.data.append(edge)

  def add_polygon(self, polygon: SphericalPolygon) -> None:
    assert polygon.points.shape[1] == 3
    self._add_polygon_lines([point for point in polygon.points])

  def add_polygon_from_points(self, points: np.array) -> None:
    assert points.shape[1] == 3
    self._add_polygon_lines(points)

  def add_polygon_from_tile_row_col(self, t_ver, t_hor, row, col) -> None:
    points = tile_points(t_ver, t_hor, row, col)
    self._add_polygon_lines(points)

  def add_trace_and_fov(self, trace: np.array) -> None:
    self.data.append(
        go.Scatter3d(x=[trace[0]],
                     y=[trace[1]],
                     z=[trace[2]],
                     mode='markers',
                     marker={
                         'size': 7,
                         'color': 'red'
                     }))
    points = fov_points(*trace)
    self._add_polygon_lines(points)

  dft_start_c = Color('DarkBlue').hex
  dft_end_c = Color('SkyBlue').hex

  def add_traces(self, traces: np.array, start_c = dft_start_c, end_c = dft_end_c, visible=True) -> None:
    # start, end marks
    self.data.append(
        go.Scatter3d(x=[traces[0][0]],
                     y=[traces[0][1]],
                     z=[traces[0][2]],
                     mode='markers',
                     visible=visible,
                     marker={
                         'size': 7,
                         'color': start_c
                     }))
    self.data.append(
        go.Scatter3d(x=[traces[-1][0]],
                     y=[traces[-1][1]],
                     z=[traces[-1][2]],
                     mode='markers',
                     visible=visible,
                     marker={
                         'size': 7,
                         'color': end_c
                     }))

    # edges
    n = len(traces)
    t = np.linspace(0, 1, 100)
    colors = [x.hex for x in list(Color(start_c).range_to(Color(end_c), n))]
    for index in range(n - 1):
      start = traces[index]
      end = traces[(index + 1)]
      result = np.array(geometric_slerp(start, end, t))
      edge = go.Scatter3d(x=result[..., 0],
                          y=result[..., 1],
                          z=result[..., 2],
                          visible=visible,
                          hovertext=f'tracesory[{index}]',
                          hoverinfo='text',
                          mode='lines',
                          line={
                              'width': 7,
                              'color': colors[index]
                          })
      self.data.append(edge)

  prediction_start_c = Color('DarkGreen').hex
  prediction_end_c = Color('LightGreen').hex

  def add_predictions(self, predictions: dict) -> None:
    steps = []
    n_one_pred = 24 + 2
    n_pre = len(self.data)
    n_end = len(self.data)+len(predictions)*n_one_pred
    # print(n_pre, n_end)
    # argsclean=[{"visible": [True] * n_pre + [False] * (n_end-n_pre)}]

    for i, (_, traces) in enumerate(predictions.items()):
      self.add_traces(traces, self.prediction_start_c, self.prediction_end_c, visible=False)
      step = dict(method="update",
                  args=[{"visible": [True] * n_pre + [False] * (n_end-n_pre)}])
      beg = n_pre + i * n_one_pred
      end = n_pre + (i+1) * n_one_pred
      # print (i, beg,end)
      step["args"][0]["visible"][beg:end] = [True] * n_one_pred
      # print(step["args"][0]["visible"][beg:])
      steps.append(step)

    self.sliders = [dict(
        active=0,
        currentvalue={"prefix": "Predition at: "},
        steps=steps
    )]


  def show(self) -> None:
    fig = go.Figure(data=self.data)
    fig.update_layout(width=800, showlegend=False, title_text=self.title)
    if hasattr(self, 'sliders'):
      fig.update_layout(sliders=self.sliders)
    fig.show()


  def show_fov(self, trace) -> None:
    assert len(trace) == 3  # cartesian

    # subplot two figures
    fig = make_subplots(rows=1,
                        cols=2,
                        specs=[[{
                            'type': 'surface'
                        }, {
                            'type': 'image'
                        }]])

    self.add_trace_and_fov(trace)
    for t in self.data:
      fig.append_trace(t, row=1, col=1)

    # heatmap
    heatmap: np.array = self.tileset.request(trace)
    if isinstance(self.tileset, TileSetVoro):
      heatmap = np.reshape(heatmap, self.tileset.shape)
    x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
    y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
    erp_heatmap = px.imshow(heatmap, text_auto=True, x=x, y=y)
    for t in erp_heatmap['data']:
      fig.append_trace(t, row=1, col=2)
    if isinstance(self.tileset, TileSet):
      # fix given phi 0 being the north pole at cartesian_to_eulerian
      fig.update_yaxes(autorange='reversed')

    title = f'trace_[{trace[0]:.2},{trace[1]:.2},{trace[2]:.2}]_{self.tileset.prefix}'
    fig.update_layout(width=800, showlegend=False, title_text=title)
    fig.show()

  def _get_imshow_from_trajects_hmps(self, df: pd.DataFrame) -> px.imshow:
    hmp_sums = df['traces_hmp'].apply(lambda traces: np.sum(traces, axis=0))
    if isinstance(self.tileset, TileSetVoro):
      hmp_sums = np.reshape(hmp_sums, self.tileset.shape)
    heatmap = np.sum(hmp_sums, axis=0)
    x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
    y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
    return px.imshow(heatmap, text_auto=True, x=x, y=y)


  def show_traject(self, row: pd.Series) -> None:
    assert row.shape[0] == 1
    assert 'traces' in row.columns
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'image'}]])
    self.add_traces(row['traces'].iloc[0])
    for d in self.data:  # load all data from the self
      fig.append_trace(d, row=1, col=1)

    # heatmap
    if 'traces_hmp' in row:
      erp_heatmap = self._get_imshow_from_trajects_hmps(row)
      erp_heatmap.update_layout(width=100, height=100)
      fig.append_trace(erp_heatmap.data[0], row=1, col=2)
      if isinstance(self.tileset, TileSet):
        # fix given phi 0 being the north pole at cartesian_to_eulerian
        fig.update_yaxes(autorange='reversed')

    title = f'{str(row.shape[0])}_trajects_{self.tileset.prefix}'
    fig.update_layout(width=800, showlegend=False, title_text=title)
    fig.show()


  def show_sum_trajects(self, df: pd.DataFrame) -> None:
    assert len(df) <= 4, 'df >=4 does not get a good visualization'
    assert not df.empty

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'image'}]])
    for _, row in df.iterrows():
      self.add_traces(row['traces'])
    for d in self.data:  # load all data from the self
      fig.append_trace(d, row=1, col=1)

    # heatmap
    if 'traces_hmp' in df:
      erp_heatmap = self._get_imshow_from_trajects_hmps(df)
      fig.append_trace(erp_heatmap.data[0], row=1, col=2)
      if isinstance(self.tileset, TileSet):
        # fix given phi 0 being the north pole at cartesian_to_eulerian
        fig.update_yaxes(autorange='reversed')

    title = f'{str(df.shape[0])}_trajects_{self.tileset.prefix}'
    fig.update_layout(width=800, showlegend=False, title_text=title)
    fig.show()
