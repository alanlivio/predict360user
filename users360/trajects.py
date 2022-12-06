"""
Provides some data functions
"""
import io
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

DF_TRAJECTS_F = os.path.join(config.DATADIR, 'df_trajects.pickle')


def _load_df_trajects_from_hmp() -> pd.DataFrame:
  config.loginf('loading trajects from head_motion_prediction project')
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
            'ds_user',  # e.g., david_0
            'ds_video',  # e.g., david_10_Cows
            # 'times',
            'traject'  # [[x,y,z], ...]
        ])
    # assert and check
    assert tmpdf['ds'].value_counts()[
        config.DS_NAMES[idx]] == config.DS_SIZES[idx]
    return tmpdf

  # create df_trajects for each dataset
  df_trajects = pd.concat(map(_load_dataset_xyz, ds_idxs),
                          ignore_index=True).convert_dtypes()
  assert not df_trajects.empty
  # back to cwd
  os.chdir(cwd)
  return df_trajects


def get_df_trajects(df_trajects_f=DF_TRAJECTS_F) -> pd.DataFrame:
  if exists(df_trajects_f):
    with open(df_trajects_f, 'rb') as f:
      config.loginf(f'loading df_trajects from {df_trajects_f}')
      df_trajects = pickle.load(f)
  else:
    config.loginf(f'no {df_trajects_f}')
    df_trajects = _load_df_trajects_from_hmp()
  return df_trajects


def dump_df_trajects(df_trajects: pd.DataFrame,
                     df_trajects_f=DF_TRAJECTS_F) -> None:
  config.loginf(f'saving df_trajects to {df_trajects_f}')
  with open(df_trajects_f, 'wb') as f:
    pickle.dump(df_trajects, f)
  with open(df_trajects_f + '.info.txt', 'w', encoding='utf-8') as f:
    buffer = io.StringIO()
    df_trajects.info(buf=buffer)
    f.write(buffer.getvalue())


def sample_traject_row(df_trajects: pd.DataFrame) -> pd.Series:
  return df_trajects.sample(1)


def sample_one_trace_from_traject_row(one_row: pd.Series) -> np.array:
  traject_ar = one_row['traject'].iloc[0]
  trace = traject_ar[np.random.randint(len(traject_ar - 1))]
  return trace

def get_traces(df_trajects: pd.DataFrame, video: str, user: str,
               ds: str) -> np.array:
  # TODO: df indexed by (ds, ds_user, ds_video)
  if ds == 'all':
    row = df_trajects.query(f"ds_user=='{user}' and ds_video=='{video}'")
  else:
    row = df_trajects.query(
        f"ds=='{ds}' and ds_user=='{user}' and ds_video=='{video}'")
  assert not row.empty
  return row['traject'].iloc[0]


def get_video_ids(df_trajects: pd.DataFrame) -> np.array:
  return df_trajects['ds_video'].unique()


def get_user_ids(df_trajects: pd.DataFrame) -> np.array:
  return df_trajects['ds_user'].unique()


def get_ds_ids(df_trajects: pd.DataFrame) -> np.array:
  return df_trajects['ds'].unique()


def get_imshow_from_trajects_hmps(df_trajects: pd.DataFrame,
                                  tileset=TILESET_DEFAULT) -> px.imshow:
  hmp_sums = df_trajects['traject_hmps'].apply(
      lambda traces: np.sum(traces, axis=0))
  if isinstance(tileset, TileSetVoro):
    hmp_sums = np.reshape(hmp_sums, tileset.shape)
  heatmap = np.sum(hmp_sums, axis=0)
  x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
  y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
  return px.imshow(heatmap, text_auto=True, x=x, y=y)


def show_one_traject(row: pd.Series, tileset=TILESET_DEFAULT) -> None:
  assert row.shape[0] == 1
  assert 'traject' in row.columns
  # subplot two figures
  fig = make_subplots(rows=1,
                      cols=2,
                      specs=[[{
                          'type': 'surface'
                      }, {
                          'type': 'image'
                      }]])
  # sphere
  sphere = VizSphere(tileset)
  sphere.add_trajectory(row['traject'].iloc[0])
  for d in sphere.data:  # load all data from the sphere
    fig.append_trace(d, row=1, col=1)

  # heatmap
  if 'traject_hmps' in row:
    erp_heatmap = get_imshow_from_trajects_hmps(row, tileset)
    erp_heatmap.update_layout(width=100, height=100)
    fig.append_trace(erp_heatmap.data[0], row=1, col=2)
    if isinstance(tileset, TileSet):
      # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
      fig.update_yaxes(autorange='reversed')

  title = f'{str(row.shape[0])}_trajects_{tileset.prefix}'
  fig.update_layout(width=800, showlegend=False, title_text=title)
  fig.show()


def show_sum_trajects(df_trajects: pd.DataFrame,
                      tileset=TILESET_DEFAULT) -> None:
  assert len(
      df_trajects) <= 4, 'df_trajects >=4 does not get a good visualization'
  assert not df_trajects.empty

  # subplot two figures
  fig = make_subplots(rows=1,
                      cols=2,
                      specs=[[{
                          'type': 'surface'
                      }, {
                          'type': 'image'
                      }]])

  # sphere
  sphere = VizSphere(tileset)
  for _, row in df_trajects.iterrows():
    sphere.add_trajectory(row['traject'])
  for d in sphere.data:  # load all data from the sphere
    fig.append_trace(d, row=1, col=1)

  # heatmap
  if 'traject_hmps' in df_trajects:
    erp_heatmap = get_imshow_from_trajects_hmps(df_trajects, tileset)
    fig.append_trace(erp_heatmap.data[0], row=1, col=2)
    if isinstance(tileset, TileSet):
      # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
      fig.update_yaxes(autorange='reversed')

  title = f'{str(df_trajects.shape[0])}_trajects_{tileset.prefix}'
  fig.update_layout(width=800, showlegend=False, title_text=title)
  fig.show()
