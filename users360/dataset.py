import io
import os
import pickle
from os.path import exists
from typing import Literal

import jenkspy
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats

from . import config
from .utils.fov import calc_actual_entropy
from .utils.tileset import TILESET_DEFAULT, TileSet


def get_class_thresholds(df, col: str) -> tuple[float, float]:
  _, threshold_medium, threshold_hight, _ = jenkspy.jenks_breaks(df[col], n_classes=3)
  return threshold_medium, threshold_hight


def get_class_name(x: float, threshold_medium: float,
                   threshold_hight: float) -> Literal['low', 'medium', 'hight']:
  return 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')


class Dataset:
  """:class:`Dataset` stores the original dataset in memory.
    It provides functions for data preprocessing, such user clustering by entropy, and analyses, such as tileset usage.
    Features are stored as :class:`pandas.DataFrame`.
    Attributes:
        df (str): pandas.DataFrame.
    """

  def __init__(self) -> None:
    if exists(config.PICKLE_FILE):
      with open(config.PICKLE_FILE, 'rb') as f:
        config.info(f'loading df from {config.PICKLE_FILE}')
        self.df = pickle.load(f)
    else:
      config.info(f'no {config.PICKLE_FILE}, loading df from {config.HMDDIR}')
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
    config.info(f'saving df to {config.PICKLE_FILE}')
    with open(config.PICKLE_FILE, 'wb') as f:
      pickle.dump(self.df, f)
    with open(config.PICKLE_FILE + '.info.txt', 'w', encoding='utf-8') as f:
      buffer = io.StringIO()
      self.df.info(buf=buffer)
      f.write(buffer.getvalue())
      f.write(f"{self.df['actS'].max()=}")
      f.write(f"{self.df['actS'].min()=}")
      f.write(f"{self.df['hmpS'].max()=}")
      f.write(f"{self.df['hmpS'].min()=}")

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
    self.df.drop(['actS', 'actS_c'], axis=1, errors='ignore', inplace=True)
    # calc actS
    self.df['actS'] = self.df['traject'].progress_apply(calc_actual_entropy)
    assert not self.df['actS'].isnull().any()
    # calc trajects_entropy_class
    threshold_medium, threshold_hight = get_class_thresholds(self.df, 'actS')
    self.df['actS_c'] = self.df['actS'].progress_apply(get_class_name,
                                                       args=(threshold_medium, threshold_hight))
    assert not self.df['actS_c'].isnull().any()

  def calc_trajects_entropy_hmp(self) -> None:
    self.df.drop(['hmpS', 'hmpS_c'], axis=1, errors='ignore', inplace=True)

    # calc hmpS
    if not 'traject_hmp' in self.df.columns:
      def _calc_traject_hmp(traces) -> np.array:
        return np.apply_along_axis(TILESET_DEFAULT.request, 1, traces)
      np_hmps = self.df['traject'].progress_apply(_calc_traject_hmp)
      self.df['traject_hmp'] = pd.Series(np_hmps)
      assert not self.df['traject_hmp'].isnull().any()
    def _hmp_entropy(traject) -> float:
      return scipy.stats.entropy(np.sum(traject, axis=0).reshape((-1)))
    self.df['hmpS'] = self.df['traject_hmp'].progress_apply(_hmp_entropy)
    assert not self.df['hmpS'].isnull().any()

    # calc hmpS_c
    threshold_medium, threshold_hight = get_class_thresholds(self.df, 'hmpS')
    self.df['hmpS_c'] = self.df['hmpS'].progress_apply(get_class_name,
                                                       args=(threshold_medium, threshold_hight))
    assert not self.df['hmpS_c'].isnull().any()

  def calc_trajects_poles_prc(self) -> None:
    self.df.drop(['poles_prc', 'poles_prc_c'], axis=1, errors='ignore', inplace=True)

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

  def show_histogram(self, cols=['actS', 'hmpS'], facet=None) -> None:
    cols = [col for col in cols if {col}.issubset(self.df.columns)]
    for col in cols:
      px.histogram(self.df,
                  x=col,
                  color=col+'_c',
                  facet_col=facet,
                  color_discrete_map=config.ENTROPY_CLASS_COLORS,
                  width=900).show()

  def drop_predict_cols(self) -> None:
    col_rm = [col for col in self.df.columns for model in config.ARGS_MODEL_NAMES if col.startswith(model)]
    self.df.drop(col_rm, axis=1, errors='ignore', inplace=True)


  def calc_tileset_reqs_metrics(self, tileset_l: list[TileSet]) -> None:
    if len(self.df) >= 4:
      config.log("df.size >= 4, it will take for some time")
    def _trace_mestrics_np(trace, tileset) -> np.array:
      heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
      return np.array([np.sum(heatmap), vp_quality, area_out])
    def _traject_metrics_np(traces, tileset) -> np.array:
      return np.apply_along_axis(_trace_mestrics_np, 1, traces, tileset=tileset)
    for tileset in tileset_l:
      column_name = f'metrics_{tileset.title}'
      metrics_np = self.df['traject'].progress_apply(_traject_metrics_np,
                                                        tileset=tileset)
      self.df[column_name] = metrics_np
      assert not self.df[column_name].empty


  def show_tileset_reqs_metrics(self) -> None:
    # check
    columns = [column for column in df.columns if column.startswith('metrics_')]
    assert len(columns), 'run calc_tileset_reqs_metrics'
    # create dftmp
    data = []
    for name in [
        column for column in df.columns if column.startswith('metrics_')
    ]:
      avg_reqs = float(self.df[name].apply(lambda traces: np.sum(traces[:, 0])).mean())
      avg_qlt = self.df[name].apply(lambda traces: np.sum(traces[:, 1])).mean()
      avg_lost = self.df[name].apply(lambda traces: np.sum(traces[:, 2])).mean()
      score = avg_qlt / avg_lost
      data.append(
          (name.removeprefix('metrics_'), avg_reqs, avg_qlt, avg_lost, score))
    assert len(data) > 0
    columns = ['tileset', 'avg_reqs', 'avg_qlt', 'avg_lost', 'score']
    dftmp = pd.DataFrame(data, columns=columns)
    # show dftmp
    fig = make_subplots(rows=4,
                        cols=1,
                        subplot_titles=columns[1:],
                        shared_yaxes=True)
    trace = go.Bar(y=dftmp['tileset'], x=dftmp['avg_reqs'], orientation='h')
    fig.add_trace(trace, row=1, col=1)
    trace = go.Bar(y=dftmp['tileset'], x=dftmp['avg_lost'], orientation='h')
    fig.add_trace(trace, row=2, col=1)
    trace = go.Bar(y=dftmp['tileset'], x=dftmp['avg_qlt'], orientation='h')
    fig.add_trace(trace, row=3, col=1)
    trace = go.Bar(y=dftmp['tileset'], x=dftmp['score'], orientation='h')
    fig.add_trace(trace, row=4, col=1)
    fig.update_layout(
        width=500,
        showlegend=False,
        barmode='stack',
    )
    fig.show()
