import logging
import os
import pickle
from os.path import exists

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from . import config
from .utils.tileset import TileSet

DF_TILESET_METRICS_F = os.path.join(config.DATADIR, 'df_tileset_metrics.pickle')

def get_tileset_metrics(tileset_metrics_f=DF_TILESET_METRICS_F) -> pd.DataFrame:
  if exists(tileset_metrics_f):
    with open(tileset_metrics_f, 'rb') as f:
      logging.info(f'loading tileset_metrics from {tileset_metrics_f}')
      tileset_metrics = pickle.load(f)
  else:
    logging.info(f'no {tileset_metrics_f}')
    tileset_metrics = pd.DataFrame()
  return tileset_metrics


def calc_tileset_reqs_metrics(df_trajects: pd.DataFrame,
  tileset_l: list[TileSet]) -> pd.DataFrame:
  df_l = []
  def f_trace(trace, tileset) -> tuple[int, float, float]:
    heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
    return (int(np.sum(heatmap)), vp_quality, area_out)
  def f_traject (traces, tileset):
    return np.apply_along_axis(f_trace, 1, traces, tileset=tileset)
  for tileset in tileset_l:
    tmpdf = pd.DataFrame(df_trajects['traject'].swifter.apply(f_traject, tileset=tileset))
    tmpdf.columns = [tileset.title]
    df_l.append(tmpdf)
  tileset_reqs_metrics = pd.concat(df_l)
  assert not tileset_reqs_metrics.empty
  return tileset_reqs_metrics


def show_tileset_reqs_metrics(tileset_reqs_metrics: pd.DataFrame) -> None:
  def f_traject_reqs(traces):
    return np.sum(traces[:, 0])
  def f_traject_qlt(traces):
    return np.mean(traces[:, 1])
  def f_traject_lost(traces):
    return np.mean(traces[:, 2])
  data = {'tileset': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
  for name in tileset_reqs_metrics.columns:
    dfts = tileset_reqs_metrics[name]
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
  y = tileset_reqs_metrics.columns
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
