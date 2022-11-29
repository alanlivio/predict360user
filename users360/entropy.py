"""
Provides some entropy functions
"""
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats
import swifter  # pylint: disable=unused-import

from . import config
from .trajects import get_df_trajects
from .utils.tileset import TILESET_DEFAULT

ENTROPY_CLASS_COLORS = {'low': 'blue', 'medium': 'green', 'hight': 'red'}


def calc_trajects_hmps(tileset=TILESET_DEFAULT, testing=False) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  def f_trace(trace):
    return tileset.request(trace)
  def f_traject(traces):
    return np.apply_along_axis(f_trace, 1, traces)
  logging.info('calculating heatmaps ...')
  df_trajects['traject_hmps'] = pd.Series(df_trajects['traject'].swifter.apply(f_traject))
  assert not df_trajects['traject_hmps'].isnull().any()


def calc_trajects_entropy(testing=False) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(testing=testing)
  # calc df_trajects.entropy
  def f_entropy(x):
    return scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
  logging.info('calculating trajects entropy ...')
  df_trajects['traject_entropy'] = df_trajects['traject_hmps'].swifter.apply(f_entropy)
  assert not df_trajects['traject_entropy'].isnull().any()
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
  assert not df_trajects['traject_entropy_class'].isnull().any()
  config.df_trajects = df_trajects


def calc_trajects_entropy_users(testing=False) -> None:
  df_trajects = get_df_trajects()
  df_trajects = df_trajects[:2] if testing else df_trajects
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(testing=testing)
  # calc user_entropy
  logging.info('calculating users entropy ...')
  def f_entropy_user(same_user_rows) -> np.ndarray:
    same_user_rows_np = same_user_rows['traject_hmps'].to_numpy()
    hmps_sum: np.ndarray = sum(np.sum(x, axis=0) for x in same_user_rows_np)
    entropy = scipy.stats.entropy(hmps_sum.reshape((-1)))
    return entropy
  tmpdf = df_trajects.groupby(['ds_user']).apply(f_entropy_user).reset_index()
  tmpdf.columns = ['ds_user', 'user_entropy']
  assert not tmpdf['user_entropy'].isnull().any()
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

def show_trajects_entropy(facet=None) -> None:
  if not {'traject_entropy', 'traject_entropy_class'}.issubset(get_df_trajects().columns):
    calc_trajects_entropy()
  df_trajects = get_df_trajects()
  fig = px.box(df_trajects,
         x='ds',
         y='traject_entropy',
         color='traject_entropy_class',
         points='all',
         facet_row=facet,
         color_discrete_map=ENTROPY_CLASS_COLORS,
         title='traject_entropy by ds',
         width=900)
  fig.update_traces(marker=dict(size=1))
  fig.show()
  px.histogram(df_trajects,
               x='ds',
               y='traject_entropy_class',
               histfunc='count',
               barmode='group',
               facet_row=facet,
               color='traject_entropy_class',
               color_discrete_map=ENTROPY_CLASS_COLORS,
               title='traject_entropy_class by ds',
               width=900).show()


def show_trajects_entropy_users(facet=None) -> None:
  if not {'traject_entropy', 'traject_entropy_class'
    }.issubset(get_df_trajects().columns):
    calc_trajects_entropy_users()
  df_trajects = get_df_trajects()
  fig = px.box(df_trajects,
         x='ds',
         y='user_entropy',
         points='all',
         color='user_entropy_class',
         facet_row=facet,
         color_discrete_map=ENTROPY_CLASS_COLORS,
         title='user_entropy by ds',
         width=900)
  fig.update_traces(marker=dict(size=1))
  fig.show()
  px.histogram(df_trajects,
               x='ds',
               y='user_entropy_class',
               histfunc='count',
               barmode='group',
               facet_row=facet,
               color='user_entropy_class',
               color_discrete_map=ENTROPY_CLASS_COLORS,
               title='user_entropy_class by ds',
               width=900).show()
