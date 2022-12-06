"""
Provides some entropy functions
"""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats
from tqdm.auto import tqdm

from . import config
from .head_motion_prediction.Utils import (cartesian_to_eulerian,
                                           eulerian_to_cartesian)
from .utils.tileset import TILESET_DEFAULT

ENTROPY_CLASS_COLORS = {'low': 'blue', 'medium': 'green', 'hight': 'red'}
RES_WIDTH = 3840
RES_HIGHT = 2160
tqdm.pandas()


def calc_fixmps_ids(traces: np.array) -> np.array:
  # calc fixation_ids
  scale = 0.025
  n_height = int(scale * RES_HIGHT)
  n_width = int(scale * RES_WIDTH)
  im_theta = np.linspace(0,
                         2 * np.pi - 2 * np.pi / n_width,
                         n_width,
                         endpoint=True)
  im_phi = np.linspace(0 + np.pi / (2 * n_height),
                       np.pi - np.pi / (2 * n_height),
                       n_height,
                       endpoint=True)

  def calc_one_fixmap_id(trace) -> np.int64:
    fixmp = np.zeros((n_height, n_width))
    target_theta, target_thi = cartesian_to_eulerian(*trace)
    mindiff_theta = np.min(abs(im_theta - target_theta))
    im_col = np.where(np.abs(im_theta - target_theta) == mindiff_theta)[0][0]
    mindiff_phi = min(abs(im_phi - target_thi))
    im_row = np.where(np.abs(im_phi - target_thi) == mindiff_phi)[0][0]
    fixmp[im_row, im_col] = 1
    fixmp_id = np.nonzero(fixmp.reshape(-1))[0][0]
    assert isinstance(fixmp_id, np.int64)
    return fixmp_id

  fixmps_ids = np.apply_along_axis(calc_one_fixmap_id, 1, traces)
  assert fixmps_ids.shape == (len(traces), )
  return fixmps_ids


def calc_actual_entropy_from_ids(x_ids_t: np.ndarray, return_sub_len_t=False) -> float:
  assert isinstance(x_ids_t, np.ndarray)
  n = len(x_ids_t)
  sub_len_l = np.zeros(n)
  sub_len_l[0] = 1
  for i in range(1, n):
    # sub_1st as current i
    sub_1st = x_ids_t[i]
    # case sub_1st not seen, so set 1
    sub_len_l[i] = 1
    sub_1st_seen_idxs = np.nonzero(x_ids_t[0:i] == sub_1st)[0]
    if sub_1st_seen_idxs.size == 0:
      continue
    # case sub_1st seen, search longest valid k-lengh sub
    for idx in sub_1st_seen_idxs:
      k=1
      while (i + k < n  # skip the last
              and idx + k <= i # until previous i
            ):
        # given valid set current k if longer
        sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
        # try match with k-lengh from idx
        next_sub = x_ids_t[i: i + k]
        k_sub = x_ids_t[idx: idx + k]
        # if not match with k-lengh from idx
        if not np.array_equal(next_sub, k_sub):
          break
        # if match increase k and set if longer
        k+=1
        sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
  actual_entropy = (1 / ((1 / n) * np.sum(sub_len_l))) * np.log2(n)
  actual_entropy = np.round(actual_entropy,3)
  if return_sub_len_t:
    return actual_entropy, sub_len_l
  else:
    return actual_entropy


def calc_actual_entropy(traces: np.array) -> float:
  fixmps_ids = calc_fixmps_ids(traces)
  return calc_actual_entropy_from_ids(fixmps_ids)


def _calc_traject_hmp(traces, tileset) -> np.array:
  return np.apply_along_axis(tileset.request, 1, traces)


def calc_trajects_hmps(df_trajects: pd.DataFrame,
                       tileset=TILESET_DEFAULT) -> None:
  config.loginf('calculating heatmaps ...')
  np_hmps = df_trajects['traject'].progress_apply(_calc_traject_hmp,
                                                  args=(tileset))
  df_trajects['traject_hmps'] = pd.Series(np_hmps)
  assert not df_trajects['traject_hmps'].isnull().any()


def _class_by_threshold(x, threshold_medium,
                        threshold_hight) -> Literal['low', 'medium', 'hight']:
  return 'low' if x < threshold_medium else (
      'medium' if x < threshold_hight else 'hight')


def _entropy_traject(traject) -> float:
  return scipy.stats.entropy(np.sum(traject, axis=0).reshape((-1)))


def calc_trajects_entropy(df_trajects: pd.DataFrame) -> None:
  # calc hmps
  if 'traject_hmps' not in df_trajects.columns:
    calc_trajects_hmps(df_trajects)
  # clean
  df_trajects.drop(['traject_entropy', 'traject_entropy_class'],
                   axis=1,
                   errors='ignore')
  # calc traject_entropy
  config.loginf('calculating trajects entropy ...')
  df_trajects['traject_entropy'] = df_trajects['traject_hmps'].progress_apply(
      _entropy_traject)
  assert not df_trajects['traject_entropy'].isnull().any()
  # calc trajects_entropy_class
  idxs_sort = df_trajects['traject_entropy'].argsort()
  trajects_len = len(df_trajects['traject_entropy'])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df_trajects['traject_entropy'][idx_threshold_medium]
  threshold_hight = df_trajects['traject_entropy'][idx_threshold_hight]
  df_trajects['traject_entropy_class'] = df_trajects[
      'traject_entropy'].progress_apply(_class_by_threshold,
                                        args=(threshold_medium,
                                              threshold_hight))
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
  df_trajects.drop(['user_entropy', 'user_entropy_class'],
                   axis=1,
                   errors='ignore')
  # calc user_entropy
  config.loginf('calculating users entropy ...')
  df_users = df_trajects.groupby(
      ['ds_user']).progress_apply(_entropy_user).reset_index()
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
  assert {'traject_entropy',
          'traject_entropy_class'}.issubset(df_trajects.columns)
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
  assert {'traject_entropy',
          'traject_entropy_class'}.issubset(df_trajects.columns)
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


def _poles_prc(traces) -> float:
  return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)


def calc_trajects_poles_prc(df_trajects: pd.DataFrame) -> None:
  # clean
  df_trajects.drop(['poles_prc', 'poles_class'], axis=1, errors='ignore')
  # calc poles_prc
  df_trajects['poles_prc'] = pd.Series(
      df_trajects['traject'].progress_apply(_poles_prc))
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