"""
Provides some data functions
"""
import io
import logging
import os
import pickle
from os.path import exists

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from . import config
from .utils.tileset import TILESET_DEFAULT, TileSet
from .utils.tileset_voro import TileSetVoro
from .utils.viz_sphere import VizSphere


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


def show_trajects(df: pd.DataFrame, tileset=TILESET_DEFAULT) -> None:
  assert not df.empty

  # subplot two figures
  fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'image'}]])

  # sphere
  sphere = VizSphere(tileset)
  for _, row in df.iterrows():
    sphere.add_trajectory(row['traject'])
  for d in sphere.data:  # load all data from the sphere
    fig.append_trace(d, row=1, col=1)
  # heatmap
  # TODO: calcuate if hmps is not in df
  if 'traject_hmps' in df:
    hmp_sums = df['traject_hmps'].apply(lambda traces: np.sum(traces, axis=0))
    if isinstance(tileset, TileSetVoro):
      hmp_sums = np.reshape(hmp_sums, tileset.shape)
    heatmap = np.sum(hmp_sums, axis=0)
    x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
    y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
    erp_heatmap = px.imshow(heatmap, text_auto=True, x=x, y=y)
    erp_heatmap.update_layout(width=100, height=100)
    fig.append_trace(erp_heatmap.data[0], row=1, col=2)
    if isinstance(tileset, TileSet):
      # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
      fig.update_yaxes(autorange='reversed')

  title = f'{str(df.shape[0])}_trajects_{tileset.prefix}'
  fig.update_layout(width=800, showlegend=False, title_text=title)
  fig.show()
