import io
import os
import pickle
from os.path import exists

import numpy as np
import pandas as pd

from . import config


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
