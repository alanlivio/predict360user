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


def calc_column_thresholds(df: pd.DataFrame, column) -> tuple[float, float]:
  idxs_sort = df[column].argsort()
  trajects_len = len(df[column])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df[column][idx_threshold_medium]
  threshold_hight = df[column][idx_threshold_hight]
  return threshold_medium, threshold_hight


def get_class_by_threshold(x, threshold_medium,
                           threshold_hight) -> Literal['low', 'medium', 'hight']:
  return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')


def calc_fixmps_ids(traces: np.array) -> np.array:
  # calc fixation_ids
  scale = 0.025
  n_height = int(scale * RES_HIGHT)
  n_width = int(scale * RES_WIDTH)
  im_theta = np.linspace(0, 2 * np.pi - 2 * np.pi / n_width, n_width, endpoint=True)
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
      k = 1
      while (i + k < n  # skip the last
             and idx + k <= i  # until previous i
             ):
        # given valid set current k if longer
        sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
        # try match with k-lengh from idx
        next_sub = x_ids_t[i:i + k]
        k_sub = x_ids_t[idx:idx + k]
        # if not match with k-lengh from idx
        if not np.array_equal(next_sub, k_sub):
          break
        # if match increase k and set if longer
        k += 1
        sub_len_l[i] = k if k > sub_len_l[i] else sub_len_l[i]
  actual_entropy = (1 / ((1 / n) * np.sum(sub_len_l))) * np.log2(n)
  actual_entropy = np.round(actual_entropy, 3)
  if return_sub_len_t:
    return actual_entropy, sub_len_l
  else:
    return actual_entropy


def calc_actual_entropy(traces: np.array) -> float:
  fixmps_ids = calc_fixmps_ids(traces)
  return calc_actual_entropy_from_ids(fixmps_ids)


def calc_trajects_entropy(df: pd.DataFrame) -> None:
  # clean
  df.drop(['traject_entropy', 'traject_entropy_class'], axis=1, errors='ignore')
  # calc traject_entropy
  config.info('calculating trajects entropy ...')
  df['traject_entropy'] = df['traject'].progress_apply(calc_actual_entropy)
  assert not df['traject_entropy'].isnull().any()
  # calc trajects_entropy_class
  threshold_medium, threshold_hight = calc_column_thresholds(df, 'traject_entropy')
  df['traject_entropy_class'] = df['traject_entropy'].progress_apply(get_class_by_threshold,
                                                                     args=(threshold_medium,
                                                                           threshold_hight))
  assert not df['traject_entropy_class'].isnull().any()


def show_trajects_entropy(df: pd.DataFrame, facet=None) -> None:
  assert {'traject_entropy', 'traject_entropy_class'}.issubset(df.columns)
  px.histogram(df,
                x='traject_entropy',
                color='traject_entropy_class',
                facet_col=facet,
                color_discrete_map=ENTROPY_CLASS_COLORS,
                width=900).show()
  px.histogram(df,
                x='hmp_entropy',
                facet_col=facet,
                color='hmp_entropy_class',
                color_discrete_map=ENTROPY_CLASS_COLORS,
                width=900).show()


def _poles_prc(traces) -> float:
  return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)


def calc_trajects_poles_prc(df: pd.DataFrame) -> None:
  # clean
  df.drop(['poles_prc', 'poles_class'], axis=1, errors='ignore')
  # calc poles_prc
  df['poles_prc'] = pd.Series(df['traject'].progress_apply(_poles_prc))
  # calc poles_class
  threshold_medium, threshold_hight = calc_column_thresholds(df, 'poles_prc')
  df['poles_class'] = df['poles_prc'].progress_apply(get_class_by_threshold,
                                                     args=(threshold_medium, threshold_hight))
  assert not df['poles_class'].isna().any()


def show_trajects_poles_prc(df: pd.DataFrame) -> None:
  assert {'poles_prc', 'poles_class'}.issubset(df.columns)
  fig = px.scatter(df,
                   x='ds_user',
                   y='poles_prc',
                   color='poles_class',
                   hover_data=[df.index],
                   title='trajects poles_perc',
                   width=700)
  fig.update_yaxes(showticklabels=False)
  fig.update_traces(marker_size=2)
  fig.show()


def _calc_traject_hmp(traces) -> np.array:
  return np.apply_along_axis(TILESET_DEFAULT.request, 1, traces)


def _hmp_entropy(traject) -> float:
  return scipy.stats.entropy(np.sum(traject, axis=0).reshape((-1)))


def calc_trajects_hmp_entropy(df: pd.DataFrame) -> None:
  if not 'traject_hmps' in df.columns:
    config.info('calculating heatmaps ...')
    np_hmps = df['traject'].progress_apply(_calc_traject_hmp)
    df['traject_hmps'] = pd.Series(np_hmps)
    assert not df['traject_hmps'].isnull().any()
  # calc hmp_entropy
  config.info('calculating heatmaps entropy ...')
  df['hmp_entropy'] = df['traject_hmps'].progress_apply(_hmp_entropy)
  assert not df['hmp_entropy'].isnull().any()
  # calc trajects_entropy_class
  # clean
  df.drop(['hmp_entropy', 'hmp_entropy_class'], axis=1, errors='ignore')
  threshold_medium, threshold_hight = calc_column_thresholds(df, 'hmp_entropy')
  df['hmp_entropy_class'] = df['hmp_entropy'].progress_apply(get_class_by_threshold,
                                                             args=(threshold_medium,
                                                                   threshold_hight))
  assert not df['traject_entropy_class'].isnull().any()
