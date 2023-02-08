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

DF_FILE = os.path.join(config.DATADIR, 'df_trajects.pickle')


def _load_df_trajects_from_hmp() -> pd.DataFrame:
  config.info('loading trajects from head_motion_prediction project')
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


def get_df_trajects(df_file=DF_FILE) -> pd.DataFrame:
  if exists(df_file):
    with open(df_file, 'rb') as f:
      config.info(f'loading df from {df_file}')
      df = pickle.load(f)
  else:
    config.info(f'no {df_file}')
    df = _load_df_trajects_from_hmp()
  return df


def dump_df_trajects(df: pd.DataFrame, df_file=DF_FILE) -> None:
  config.info(f'saving df to {df_file}')
  with open(df_file, 'wb') as f:
    pickle.dump(df, f)
  with open(df_file + '.info.txt', 'w', encoding='utf-8') as f:
    buffer = io.StringIO()
    df.info(buf=buffer)
    f.write(buffer.getvalue())


def sample_traject_row(df: pd.DataFrame) -> pd.Series:
  return df.sample(1)


def sample_one_trace_from_traject_row(one_row: pd.Series) -> np.array:
  traject_ar = one_row['traject'].iloc[0]
  trace = traject_ar[np.random.randint(len(traject_ar - 1))]
  return trace


def get_rows(df: pd.DataFrame, video: str, user: str, ds: str) -> np.array:
  if ds == 'all':
    rows = df.query(f"user=='{user}' and video=='{video}'")
  else:
    rows = df.query(f"ds=='{ds}' and user=='{user}' and video=='{video}'")
  assert not rows.empty
  return rows


def get_traces(df: pd.DataFrame, video: str, user: str, ds: str) -> np.array:
  if ds == 'all':
    row = df.query(f"user=='{user}' and video=='{video}'")
  else:
    row = df.query(f"ds=='{ds}' and user=='{user}' and video=='{video}'")
  assert not row.empty
  return row['traject'].iloc[0]


def get_video_ids(df: pd.DataFrame) -> np.array:
  return df['video'].unique()


def get_user_ids(df: pd.DataFrame) -> np.array:
  return df['user'].unique()


def get_ds_ids(df: pd.DataFrame) -> np.array:
  return df['ds'].unique()


def _get_imshow_from_trajects_hmps(df: pd.DataFrame, tileset=TILESET_DEFAULT) -> px.imshow:
  hmp_sums = df['traject_hmp'].apply(lambda traces: np.sum(traces, axis=0))
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
  fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'image'}]])
  # sphere
  sphere = VizSphere(tileset)
  sphere.add_trajectory(row['traject'].iloc[0])
  for d in sphere.data:  # load all data from the sphere
    fig.append_trace(d, row=1, col=1)

  # heatmap
  if 'traject_hmp' in row:
    erp_heatmap = _get_imshow_from_trajects_hmps(row, tileset)
    erp_heatmap.update_layout(width=100, height=100)
    fig.append_trace(erp_heatmap.data[0], row=1, col=2)
    if isinstance(tileset, TileSet):
      # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
      fig.update_yaxes(autorange='reversed')

  title = f'{str(row.shape[0])}_trajects_{tileset.prefix}'
  fig.update_layout(width=800, showlegend=False, title_text=title)
  fig.show()


def show_sum_trajects(df: pd.DataFrame, tileset=TILESET_DEFAULT) -> None:
  assert len(df) <= 4, 'df >=4 does not get a good visualization'
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
  if 'traject_hmp' in df:
    erp_heatmap = _get_imshow_from_trajects_hmps(df, tileset)
    fig.append_trace(erp_heatmap.data[0], row=1, col=2)
    if isinstance(tileset, TileSet):
      # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
      fig.update_yaxes(autorange='reversed')

  title = f'{str(df.shape[0])}_trajects_{tileset.prefix}'
  fig.update_layout(width=800, showlegend=False, title_text=title)
  fig.show()
