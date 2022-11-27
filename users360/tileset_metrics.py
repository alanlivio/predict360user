import logging
import pickle
from os.path import exists

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from . import config
from .data import get_df_trajects
from .utils.tileset import TileSet


def get_df_tileset_metrics() -> pd.DataFrame:
  if config.df_tileset_metrics is None:
    if exists(config.df_tileset_metrics_f):
      with open(config.df_tileset_metrics_f, 'rb') as f:
        logging.info(f'loading df_tileset_metrics from {config.df_tileset_metrics_f}')
        config.df_tileset_metrics = pickle.load(f)
    else:
      logging.info(f'no {config.df_tileset_metrics_f}')
      config.df_tileset_metrics = pd.DataFrame()
  return config.df_tileset_metrics

def calc_trajects_poles_prc(testing=None) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  def f_traject(traces):
    return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)
  df_trajects['poles_prc'] = pd.Series(df_trajects['traject'].apply(f_traject))
  assert not df_trajects['poles_prc'].isna().any()
  idxs_sort = df_trajects['poles_prc'].argsort()
  trajects_len = len(df_trajects['poles_prc'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df_trajects['poles_prc'][idx_threshold_medium]
  threshold_hight = df_trajects['poles_prc'][idx_threshold_hight]
  def f_threshold(x):
    return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
  df_trajects['poles_class'] = df_trajects['poles_prc'].apply(f_threshold)
  assert not df_trajects['poles_class'].isna().any()


def show_trajects_poles_prc() -> None:
  if not {'poles_prc', 'poles_class'}.issubset(get_df_trajects().columns):
    calc_trajects_poles_prc()
  df_trajects = get_df_trajects()
  fig = px.scatter(df_trajects,
                   y='ds_user',
                   x='poles_prc',
                   color='poles_class',
                   hover_data=[df_trajects.index],
                   title='trajects poles_perc',
                   width=700)
  fig.update_yaxes(showticklabels=False)
  fig.update_traces(marker_size=2)
  fig.show()


def calc_tileset_reqs_metrics(tileset_l: list[TileSet], testing=None) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:1] if testing else df_trajects
  def create_tsdf(ts_idx) -> pd.DataFrame:
    tileset = tileset_l[ts_idx]
    def f_trace(trace) -> tuple[int, float, float]:
      heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
      return (int(np.sum(heatmap)), vp_quality, area_out)
    def f_traject (traces):
      return np.apply_along_axis(f_trace, 1, traces)
    tmpdf = pd.DataFrame(df_trajects['traject'].swifter.apply(f_traject))
    tmpdf.columns = [tileset.title]
    return tmpdf
  config.df_tileset_metrics = pd.concat(map(create_tsdf, range(len(tileset_l))), axis=1)


def show_tileset_reqs_metrics() -> None:
  df_tileset_metrics = config.df_tileset_metrics
  def f_traject_reqs(traces):
    return np.sum(traces[:, 0])
  def f_traject_qlt(traces):
    return np.mean(traces[:, 1])
  def f_traject_lost(traces):
    return np.mean(traces[:, 2])
  data = {'tileset': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
  for name in df_tileset_metrics.columns:
    dfts = df_tileset_metrics[name]
    data['tileset'].append(name)
    data['avg_reqs'].append(dfts.apply(f_traject_reqs).mean())
    data['avg_qlt'].append(dfts.apply(f_traject_qlt).mean())
    data['avg_lost'].append(dfts.apply(f_traject_lost).mean())
    data['score'] = data['avg_qlt'][-1] / data['avg_lost'][-1]
  df = pd.DataFrame(data)
  fig = make_subplots(rows=4,
                      cols=1,
                      subplot_titles=('avg_reqs', 'avg_lost', 'avg_qlt', 'score=avg_qlt/avg_lost'),
                      shared_yaxes=True)
  y = df_tileset_metrics.columns
  trace = go.Bar(y=y, x=df['avg_reqs'], orientation='h', width=0.3)
  fig.add_trace(trace, row=1, col=1)
  trace = go.Bar(y=y, x=df['avg_lost'], orientation='h', width=0.3)
  fig.add_trace(trace, row=2, col=1)
  trace = go.Bar(y=y, x=df['avg_qlt'], orientation='h', width=0.3)
  fig.add_trace(trace, row=3, col=1)
  trace = go.Bar(y=y, x=df['score'], orientation='h', width=0.3)
  fig.add_trace(trace, row=4, col=1)
  fig.update_layout(
      width=700,
      showlegend=False,
      barmode='stack',
  )
  fig.show()
