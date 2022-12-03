"""
Provides some entropy functions
"""
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats
from tqdm.auto import tqdm

from .utils.tileset import TILESET_DEFAULT

ENTROPY_CLASS_COLORS = {'low': 'blue', 'medium': 'green', 'hight': 'red'}

tqdm.pandas()


def _calc_traject_hmp(traces, tileset) -> np.array:
  return np.apply_along_axis(tileset.request, 1, traces)


def calc_trajects_hmps(df_trajects: pd.DataFrame, tileset=TILESET_DEFAULT) -> None:
  logging.info('calculating heatmaps ...')
  np_hmps = df_trajects['traject'].progress_apply(_calc_traject_hmp, args=(tileset))
  df_trajects['traject_hmps'] = pd.Series(np_hmps)
  assert not df_trajects['traject_hmps'].isnull().any()


def _class_by_threshold(x, threshold_medium, threshold_hight):
  return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')


def _entropy_traject(traject):
  return scipy.stats.entropy(np.sum(traject, axis=0).reshape((-1)))


def calc_trajects_entropy(df_trajects: pd.DataFrame) -> None:
  # calc hmps
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(df_trajects)
  # clean
  df_trajects.drop(['traject_entropy', 'traject_entropy_class'], axis=1, errors='ignore')
  # calc traject_entropy
  logging.info('calculating trajects entropy ...')
  df_trajects['traject_entropy'] = df_trajects['traject_hmps'].progress_apply(_entropy_traject)
  assert not df_trajects['traject_entropy'].isnull().any()
  # calc trajects_entropy_class
  idxs_sort = df_trajects['traject_entropy'].argsort()
  trajects_len = len(df_trajects['traject_entropy'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df_trajects['traject_entropy'][idx_threshold_medium]
  threshold_hight = df_trajects['traject_entropy'][idx_threshold_hight]
  df_trajects['traject_entropy_class'] = df_trajects['traject_entropy'].progress_apply(
      _class_by_threshold, args=(threshold_medium, threshold_hight))
  assert not df_trajects['traject_entropy_class'].isnull().any()


def _entropy_user(same_user_rows) -> np.ndarray:
  same_user_rows_np = same_user_rows['traject_hmps'].to_numpy()
  hmps_sum: np.ndarray = sum(np.sum(x, axis=0) for x in same_user_rows_np)
  entropy = scipy.stats.entropy(hmps_sum.reshape((-1)))
  return entropy


def calc_users_entropy(df_trajects: pd.DataFrame) -> pd.DataFrame:
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(df_trajects)
  # clean
  df_trajects.drop(['user_entropy', 'user_entropy_class'], axis=1, errors='ignore')
  # calc user_entropy
  logging.info('calculating users entropy ...')
  df_users = df_trajects.groupby(['ds_user']).progress_apply(_entropy_user).reset_index()
  df_users.columns = ['ds_user', 'user_entropy']
  assert not df_users['user_entropy'].isnull().any()
  # calc user_entropy_class
  idxs_sort = df_users['user_entropy'].argsort()
  trajects_len = len(df_users['user_entropy'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df_users['user_entropy'][idx_threshold_medium]
  threshold_hight = df_users['user_entropy'][idx_threshold_hight]
  df_users['user_entropy_class'] = df_users['user_entropy'].progress_apply(
      _class_by_threshold, args=(threshold_medium, threshold_hight))
  assert not df_users['user_entropy_class'].isna().any()

  # merge right inplace
  # https://stackoverflow.com/questions/50849102/pandas-left-join-in-place
  df_tmp = pd.merge(df_trajects[['ds_user']], df_users, on='ds_user')
  df_trajects.assign(user_entropy=df_tmp['user_entropy'].values,
                     user_entropy_class=df_tmp['user_entropy_class'].values)


def show_trajects_entropy(df_trajects: pd.DataFrame, facet=None) -> None:
  assert {'traject_entropy', 'traject_entropy_class'}.issubset(df_trajects.columns)
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


def show_trajects_entropy_users(df_trajects: pd.DataFrame, facet=None) -> None:
  assert {'traject_entropy', 'traject_entropy_class'}.issubset(df_trajects.columns)
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


def _poles_prc(traces):
  return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)


def calc_trajects_poles_prc(df_trajects: pd.DataFrame) -> None:
  # clean
  df_trajects.drop(['poles_prc', 'poles_class'], axis=1, errors='ignore')
  # calc poles_prc
  df_trajects['poles_prc'] = pd.Series(df_trajects['traject'].progress_apply(_poles_prc))
  assert not df_trajects['poles_prc'].isna().any()
  idxs_sort = df_trajects['poles_prc'].argsort()
  trajects_len = len(df_trajects['poles_prc'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df_trajects['poles_prc'][idx_threshold_medium]
  threshold_hight = df_trajects['poles_prc'][idx_threshold_hight]
  # calc poles_class
  df_trajects['poles_class'] = df_trajects['poles_prc'].progress_apply(
    _class_by_threshold, args=(threshold_medium, threshold_hight))
  assert not df_trajects['poles_class'].isna().any()


def show_trajects_poles_prc(df_trajects: pd.DataFrame) -> None:
  assert {'poles_prc', 'poles_class'}.issubset(df_trajects.columns)
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