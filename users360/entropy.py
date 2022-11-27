"""
Provides some entropy functions
"""
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats
import swifter  # pylint: disable=unused-import
from plotly.subplots import make_subplots

from . import config
from .data import get_df_trajects
from .utils.tileset import TILESET_DEFAULT, TileSetIF

ENTROPY_CLASS_COLORS = {'low': 'blue', 'medium': 'green', 'hight': 'red'}


def calc_trajects_hmps(tileset=TILESET_DEFAULT, testing=None) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  def f_trace(trace):
    return tileset.request(trace)
  def f_traject(traces):
    return np.apply_along_axis(f_trace, 1, traces)
  logging.info('calculating heatmaps ...')
  df_trajects['traject_hmps'] = pd.Series(df_trajects['traject'].swifter.apply(f_traject))
  assert not df_trajects['traject_hmps'].isnull().values.any()


def calc_trajects_entropy(testing=None) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(testing)
  # calc df_trajects.entropy
  def f_entropy(x):
    return scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
  logging.info('calculating trajects entropy ...')
  df_trajects['traject_entropy'] = df_trajects['traject_hmps'].swifter.apply(f_entropy)
  assert not df_trajects['traject_entropy'].isnull().values.any()
  # calc df_trajects.entropy_class
  idxs_sort = df_trajects['traject_entropy'].argsort()
  trajects_len = len(df_trajects['traject_entropy'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df_trajects['traject_entropy'][idx_threshold_medium]
  threshold_hight = df_trajects['traject_entropy'][idx_threshold_hight]
  def f_threshold(x):
    return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
  df_trajects['traject_entropy_class'] = df_trajects['traject_entropy'].apply(f_threshold)
  assert not df_trajects['traject_entropy_class'].isnull().values.any()
  config.df_trajects = df_trajects


def calc_trajects_entropy_users(testing=None) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(testing)
  # calc user_entropy
  logging.info('calculating users entropy ...')
  def f_entropy_user(same_user_rows) -> np.ndarray:
    same_user_rows_np = same_user_rows['traject_hmps'].to_numpy()
    hmps_sum = sum(np.sum(x, axis=0) for x in same_user_rows_np)
    entropy = scipy.stats.entropy(hmps_sum.reshape((-1)))
    return entropy
  tmpdf = df_trajects.groupby(['ds_user']).apply(f_entropy_user).reset_index()
  tmpdf.columns = ['ds_user', 'user_entropy']
  assert not tmpdf['user_entropy'].isnull().values.any()
  # calc user_entropy_class
  idxs_sort = tmpdf['user_entropy'].argsort()
  trajects_len = len(tmpdf['user_entropy'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = tmpdf['user_entropy'][idx_threshold_medium]
  threshold_hight = tmpdf['user_entropy'][idx_threshold_hight]
  def f_threshold(x):
    return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
  tmpdf['user_entropy_class'] = tmpdf['user_entropy'].apply(f_threshold)
  assert not tmpdf['user_entropy_class'].isna().any()
  config.df_trajects = pd.merge(df_trajects, tmpdf, on='ds_user')


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
                   color_discrete_map=ENTROPY_CLASS_COLORS,
                   hover_data=[df_trajects.index],
                   title='trajects poles_perc',
                   width=700)
  fig.update_yaxes(showticklabels=False)
  fig.update_traces(marker_size=2)
  fig.show()


def show_trajects_entropy(facet=None) -> None:
  if not {'traject_entropy', 'traject_entropy_class'}.issubset(get_df_trajects().columns):
    calc_trajects_entropy()
  df_trajects = get_df_trajects()
  px.box(df_trajects,
         x='ds',
         y='traject_entropy',
         color='traject_entropy_class',
         facet_row=facet,
         color_discrete_map=ENTROPY_CLASS_COLORS,
         title='traject_entropy by ds',
         width=700).show()
  px.histogram(df_trajects,
               x='ds',
               y='traject_entropy_class',
               histfunc='count',
               barmode='group',
               facet_row=facet,
               color='traject_entropy_class',
               color_discrete_map=ENTROPY_CLASS_COLORS,
               title='traject_entropy_class by ds',
               width=700).show()


def show_trajects_entropy_users(facet=None) -> None:
  if not {'traject_entropy', 'traject_entropy_class'
    }.issubset(get_df_trajects().df_trajects.columns):
    calc_trajects_entropy_users()
  df_trajects = get_df_trajects()
  px.box(df_trajects,
         x='ds',
         y='user_entropy',
         color='user_entropy_class',
         facet_row=facet,
         color_discrete_map=ENTROPY_CLASS_COLORS,
         title='user_entropy by ds',
         width=700).show()
  px.histogram(df_trajects,
               x='ds',
               y='user_entropy_class',
               histfunc='count',
               barmode='group',
               facet_row=facet,
               color='user_entropy_class',
               color_discrete_map=ENTROPY_CLASS_COLORS,
               title='user_entropy_class by ds',
               width=700).show()


def calc_tileset_reqs_metrics(tileset_l: list[TileSetIF], testing=None) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
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
    np.sum(traces[:, 0])
  def f_traject_qlt(traces):
    np.mean(traces[:, 1])
  def f_traject_lost(traces):
    np.mean(traces[:, 2])
  data = {'tileset': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
  for name in df_tileset_metrics.columns:
    dfts = df_tileset_metrics[name]
    data['tileset'].append(name)
    data['avg_reqs'].append(dfts.apply(f_traject_reqs).mean())
    data['avg_qlt'].append(dfts.apply(f_traject_qlt).mean())
    data['avg_lost'].append(dfts.apply(f_traject_lost).mean())
    data['score'] = data['avg_qlt'][-1] / data['avg_lost'][-1]
  df = pd.DataFrame(data)
  fig = make_subplots(rows=1,
                      cols=4,
                      subplot_titles=('avg_reqs', 'avg_lost', 'avg_qlt', 'score=avg_qlt/avg_lost'),
                      shared_yaxes=True)
  y = df_tileset_metrics.columns
  trace = go.Bar(y=y, x=df['avg_reqs'], orientation='h', width=0.3)
  fig.add_trace(trace, row=1, col=1)
  trace = go.Bar(y=y, x=df['avg_lost'], orientation='h', width=0.3)
  fig.add_trace(trace, row=1, col=2)
  trace = go.Bar(y=y, x=df['avg_qlt'], orientation='h', width=0.3)
  fig.add_trace(trace, row=1, col=3)
  trace = go.Bar(y=y, x=df['score'], orientation='h', width=0.3)
  fig.add_trace(trace, row=1, col=4)
  fig.update_layout(
      width=1500,
      showlegend=False,
      barmode='stack',
  )
  fig.show()
