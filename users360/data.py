"""
Provides some data functions
"""
from __future__ import annotations

import io
import logging
import os
import pickle
from os.path import exists

import numpy as np
import pandas as pd

from . import config


def _load_df_trajects_from_hmp() -> pd.DataFrame:
  logging.info('loading trajects from head_motion_prediction project')
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
            'ds', # e.g., david
            'ds_user', # e.g., david_0
            'ds_video', # e.g., david_10_Cows
            # 'times',
            'traject' # [[x,y,z], ...]
        ])
    # assert and check
    assert tmpdf['ds'].value_counts()[config.DS_NAMES[idx]] == config.DS_SIZES[idx]
    return tmpdf

  # create df_trajects for each dataset
  df_trajects = pd.concat(map(_load_dataset_xyz, ds_idxs), ignore_index=True).convert_dtypes()
  assert not df_trajects.empty
  # back to cwd
  os.chdir(cwd)
  return df_trajects


def get_df_trajects() -> pd.DataFrame:
  if config.df_trajects is None:
    if exists(config.df_trajects_f):
      with open(config.df_trajects_f, 'rb') as f:
        logging.info(f'loading df_trajects from {config.df_trajects_f}')
        config.df_trajects = pickle.load(f)
    else:
      logging.info(f'no {config.df_trajects_f}')
      config.df_trajects = _load_df_trajects_from_hmp()
  return config.df_trajects


def dump_df_trajects() -> None:
  logging.info(f'saving df_trajects to {config.df_trajects_f}')
  with open(config.df_trajects_f, 'wb') as f:
    pickle.dump(config.df_trajects, f)
  with open(config.df_trajects_f+'.info.txt', 'w', encoding='utf-8') as f:
    buffer = io.StringIO()
    config.df_trajects.info(buf=buffer)
    f.write(buffer.getvalue())


def get_one_trace() -> np.array:
  return config.df_trajects.iloc[0]['traject'][0]


def get_traces(video, user, ds='David_MMSys_18') -> np.array:
  # TODO: df indexed by (ds, ds_user, ds_video)
  if ds == 'all':
    row = config.df_trajects.query(f"ds_user=='{user}' and ds_video=='{video}'")
  else:
    row = config.df_trajects.query(
        f"ds=='{ds}' and ds_user=='{user}' and ds_video=='{video}'")
  assert not row.empty
  return row['traject'].iloc[0]


def get_video_ids(ds='David_MMSys_18') -> np.array:
  df = config.df_trajects
  return df.loc[df['ds'] == ds]['ds_video'].unique()


def get_user_ids(ds='David_MMSys_18') -> np.array:
  df = config.df_trajects
  return df.loc[df['ds'] == ds]['ds_user'].unique()


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
