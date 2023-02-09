import io
import os
import pickle
from os.path import exists
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats

from . import config
from .utils.fov import calc_actual_entropy
from .utils.tileset import TILESET_DEFAULT


def get_class_thresholds(df, col: str) -> tuple[float, float]:
  idxs_sort = df[col].argsort()
  trajects_len = len(df[col])
  idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
  idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
  threshold_medium = df[col][idx_threshold_medium]
  threshold_hight = df[col][idx_threshold_hight]
  return threshold_medium, threshold_hight


def get_class_name(x: float, threshold_medium: float, threshold_hight: float) -> Literal['low', 'medium', 'hight']:
  return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')


class Dataset:

  def __init__(self, pickle_f=config.PICKLE_FILE) -> None:
    self.pickle_f = pickle_f
    if exists(self.pickle_f):
      with open(self.pickle_f, 'rb') as f:
        config.info(f'loading df from {self.pickle_f}')
        self.df = pickle.load(f)
    else:
      config.info(f'no {self.pickle_f}, loading df from {config.HMDDIR}')
      self.df = self._load_df_trajects_from_hmp()

  def _load_df_trajects_from_hmp(self) -> pd.DataFrame:
    # save cwd and move to head_motion_prediction for invoking funcs
    cwd = os.getcwd()
    os.chdir(config.HMDDIR)
    from .head_motion_prediction.David_MMSys_18 import Read_Dataset as david
    from .head_motion_prediction.Fan_NOSSDAV_17 import Read_Dataset as fan
    from .head_motion_prediction.Nguyen_MM_18 import Read_Dataset as nguyen
    from .head_motion_prediction.Xu_CVPR_18 import Read_Dataset as xucvpr
    from .head_motion_prediction.Xu_PAMI_18 import Read_Dataset as xupami
    ds_pkgs = [david, fan, nguyen, xucvpr, xupami]  # [:1]
    ds_idxs = range(len(ds_pkgs))

    def _load_dataset_xyz(idx, n_traces=100) -> pd.DataFrame:
      # create_and_store_sampled_dataset()
      # stores csv at head_motion_prediction/<dataset>/sampled_dataset
      if len(os.listdir(ds_pkgs[idx].OUTPUT_FOLDER)) < 2:
        ds_pkgs[idx].create_and_store_sampled_dataset()
      # load_sample_dateset() process head_motion_prediction/<dataset>/sampled_dataset
      # and return a dict with:
      # {<video1>:{
      #   <user1>:[time-stamp, x, y, z],
      #    ...
      #  },
      #  ...
      # }"
      dataset = ds_pkgs[idx].load_sampled_dataset()
      # convert dict to DataFrame
      data = [
          (
              config.DS_NAMES[idx],
              config.DS_NAMES[idx] + '_' + user,
              config.DS_NAMES[idx] + '_' + video,
              # np.around(dataset[user][video][:n_traces, 0], decimals=2),
              dataset[user][video][:n_traces, 1:]) for user in dataset.keys()
          for video in dataset[user].keys()
      ]
      tmpdf = pd.DataFrame(
          data,
          columns=[
              'ds',  # e.g., david
              'user',  # e.g., david_0
              'video',  # e.g., david_10_Cows
              # 'times',
              'traject'  # [[x,y,z], ...]
          ])
      # assert and check
      assert tmpdf['ds'].value_counts()[config.DS_NAMES[idx]] == config.DS_SIZES[idx]
      return tmpdf

    # create df for each dataset
    df = pd.concat(map(_load_dataset_xyz, ds_idxs), ignore_index=True).convert_dtypes()
    assert not df.empty
    # back to cwd
    os.chdir(cwd)
    return df

  def dump(self) -> None:
    config.info(f'saving df to {self.pickle_f}')
    with open(self.pickle_f, 'wb') as f:
      pickle.dump(self.df, f)
    with open(self.pickle_f + '.info.txt', 'w', encoding='utf-8') as f:
      buffer = io.StringIO()
      self.df.info(buf=buffer)
      f.write(buffer.getvalue())

  def random_traject(self) -> pd.Series:
    return self.df.sample(1)

  def random_trace(self, ) -> np.array:
    traject_ar = self.random_traject()['traject'].iloc[0]
    trace = traject_ar[np.random.randint(len(traject_ar - 1))]
    return trace

  def get_rows(self, video: str, user: str, ds: str) -> np.array:
    if ds == 'all':
      rows = self.df.query(f"user=='{user}' and video=='{video}'")
    else:
      rows = self.df.query(f"ds=='{ds}' and user=='{user}' and video=='{video}'")
    assert not rows.empty
    return rows

  def get_traces(self, video: str, user: str, ds: str) -> np.array:
    if ds == 'all':
      row = self.df.query(f"user=='{user}' and video=='{video}'")
    else:
      row = self.df.query(f"ds=='{ds}' and user=='{user}' and video=='{video}'")
    assert not row.empty
    return row['traject'].iloc[0]

  def get_video_ids(self) -> np.array:
    return self.df['video'].unique()

  def get_user_ids(self) -> np.array:
    return self.df['user'].unique()

  def get_ds_ids(self) -> np.array:
    return self.df['ds'].unique()

  def calc_trajects_entropy(self) -> None:
    # clean
    self.df.drop(['actS', 'actS_c'], axis=1, errors='ignore')
    # calc actS
    config.info('calculating trajects entropy ...')
    self.df['actS'] = self.df['traject'].progress_apply(calc_actual_entropy)
    assert not self.df['actS'].isnull().any()
    # calc trajects_entropy_class
    threshold_medium, threshold_hight = get_class_thresholds(self.df, 'actS')
    self.df['actS_c'] = self.df['actS'].progress_apply(get_class_name,
                                                       args=(threshold_medium, threshold_hight))
    assert not self.df['actS_c'].isnull().any()

  def show_trajects_entropy(self, facet=None) -> None:
    assert {'actS', 'actS_c'}.issubset(self.df.columns)
    px.histogram(self.df,
                 x='actS',
                 color='actS_c',
                 facet_col=facet,
                 color_discrete_map=config.ENTROPY_CLASS_COLORS,
                 width=900).show()
    px.histogram(self.df,
                 x='hmpS',
                 facet_col=facet,
                 color='hmpS_c',
                 color_discrete_map=config.ENTROPY_CLASS_COLORS,
                 width=900).show()

  def calc_trajects_poles_prc(self) -> None:
    # clean
    self.df.drop(['poles_prc', 'poles_prc_c'], axis=1, errors='ignore')

    # calc poles_prc
    def _calc_poles_prc(traces) -> float:
      return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)
    self.df['poles_prc'] = pd.Series(self.df['traject'].progress_apply(_calc_poles_prc))
    # calc poles_prc_c
    threshold_medium, threshold_hight = get_class_thresholds(self.df, 'poles_prc')
    self.df['poles_prc_c'] = self.df['poles_prc'].progress_apply(get_class_name,
                                                                 args=(threshold_medium,
                                                                       threshold_hight))
    assert not self.df['poles_prc_c'].isna().any()

  def show_trajects_poles_prc(self) -> None:
    assert {'poles_prc', 'poles_prc_c'}.issubset(self.df.columns)
    px.histogram(self.df,
                 x='poles_prc',
                 color='poles_prc_c',
                 color_discrete_map=config.ENTROPY_CLASS_COLORS,
                 width=900).show()

  def calc_trajects_hmp_entropy(self) -> None:
    if not 'traject_hmp' in self.df.columns:
      config.info('calculating heatmaps ...')

      def _calc_traject_hmp(traces) -> np.array:
        return np.apply_along_axis(TILESET_DEFAULT.request, 1, traces)

      np_hmps = self.df['traject'].progress_apply(_calc_traject_hmp)
      self.df['traject_hmp'] = pd.Series(np_hmps)
      assert not self.df['traject_hmp'].isnull().any()
    # calc hmpS
    config.info('calculating heatmaps entropy ...')

    def _hmp_entropy(traject) -> float:
      return scipy.stats.entropy(np.sum(traject, axis=0).reshape((-1)))

    self.df['hmpS'] = self.df['traject_hmp'].progress_apply(_hmp_entropy)
    assert not self.df['hmpS'].isnull().any()
    # calc trajects_entropy_class
    # clean
    self.df.drop(['hmpS', 'hmpS_c'], axis=1, errors='ignore')
    threshold_medium, threshold_hight = get_class_thresholds(self.df, 'hmpS')
    self.df['hmpS_c'] = self.df['hmpS'].progress_apply(get_class_name,
                                                       args=(threshold_medium, threshold_hight))
    assert not self.df['actS_c'].isnull().any()
