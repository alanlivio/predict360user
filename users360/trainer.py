import os
import pathlib
import sys
from contextlib import redirect_stderr
from dataclasses import dataclass
from itertools import product
from os.path import exists, join
from typing import Any, Generator

import IPython
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from users360.head_motion_prediction.Utils import (all_metrics,
                                                   cartesian_to_eulerian,
                                                   eulerian_to_cartesian)

from . import config
from .dataset import *
from .utils.fov import *

METRIC = all_metrics['orthodromic']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tqdm.pandas()


def train_test_split_entropy(df: pd.DataFrame, entropy_type: str, train_entropy: str, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  if train_entropy == 'all':
    df = df
  elif train_entropy == 'nohight':
    df = df[df[entropy_type+'_c'] != 'hight']
  else:
    df = df[df[entropy_type+'_c'] == train_entropy]
  return train_test_split(df, random_state=1, test_size=test_size)

def count_entropy(df: pd.DataFrame, entropy_type: str) -> tuple[int, int, int, int]:
  a_len = len(df)
  l_len = len(df[df[entropy_type+'_c'] == 'low'])
  m_len = len(df[df[entropy_type+'_c'] == 'medium'])
  h_len = len(df[df[entropy_type+'_c'] == 'hight'])
  return a_len, l_len, m_len, h_len

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch) -> np.array:
  positions_in_batch = np.array(positions_in_batch)
  eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch]
                      for batch in positions_in_batch]
  eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
  return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions) -> np.array:
  positions = positions * np.array([2 * np.pi, np.pi])
  eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
  return np.array(eulerian_samples)


@dataclass
class Trainer():

  # dataclass attrs
  model_name: str = 'pos_only'
  dataset_name: str = 'all'
  h_window: int = 25
  init_window: int = 30
  m_window: int = 5
  perc_test: float = 0.2
  train_entropy: str = 'all'
  # dataclass attrs: train()
  epochs: int = config.DEFAULT_EPOCHS
  # dataclass attrs: evaluate()
  test_user: str = ''
  test_video: str = ''
  dry_run: bool = False

  def __post_init__(self) -> None:
    assert self.model_name in config.ARGS_MODEL_NAMES
    assert self.dataset_name in config.ARGS_DS_NAMES
    assert self.train_entropy in config.ARGS_ENTROPY_NAMES + config.ARGS_ENTROPY_AUTO_NAMES
    self.using_auto = self.train_entropy.startswith('auto')
    self.entropy_type = 'hmpS' if self.train_entropy.endswith('hmp') else 'actS'
    if self.dataset_name == 'all' and self.train_entropy == 'all':
      self.model_fullname = self.model_name
    elif self.train_entropy == 'all':
      self.model_fullname = f'{self.model_name},{self.dataset_name},,'
    else:
      self.train_entropy = self.train_entropy.removesuffix('_hmp')
      self.model_fullname = f'{self.model_name},{self.dataset_name},{self.entropy_type},{self.train_entropy}'
    self.model_dir = join(config.DATADIR, self.model_fullname)
    self.model_weights = join(self.model_dir, 'weights.hdf5')
    self.end_window = self.h_window
    config.info(self)

  def create_model(self) -> Any:
    if self.model_name == 'pos_only':
      from users360.head_motion_prediction.position_only_baseline import \
          create_pos_only_model
      return create_pos_only_model(self.m_window, self.h_window)
    else:
      raise NotImplementedError

  def generate_batchs(self, wins: list) -> Generator:
    while True:
      encoder_pos_inputs_for_batch = []
      # encoder_sal_inputs_for_batch = []
      decoder_pos_inputs_for_batch = []
      # decoder_sal_inputs_for_batch = []
      decoder_outputs_for_batch = []
      count = 0
      np.random.shuffle(wins)
      for ids in wins:
        user = ids['user']
        video = ids['video']
        x_i = ids['trace_id']
        # load the data
        if self.model_name == 'pos_only':
          encoder_pos_inputs_for_batch.append(
              self.ds.get_traces(video, user, 'all')[x_i - self.m_window:x_i])
          decoder_pos_inputs_for_batch.append(self.ds.get_traces(video, user, 'all')[x_i:x_i + 1])
          decoder_outputs_for_batch.append(
              self.ds.get_traces(video, user, 'all')[x_i + 1:x_i + self.h_window + 1])
        else:
          raise NotImplementedError
        count += 1
        if count == config.BATCH_SIZE:
          count = 0
          if self.model_name == 'pos_only':
            yield ([
                transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
                transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)
            ], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
          else:
            raise NotImplementedError
          encoder_pos_inputs_for_batch = []
          # encoder_sal_inputs_for_batch = []
          decoder_pos_inputs_for_batch = []
          # decoder_sal_inputs_for_batch = []
          decoder_outputs_for_batch = []
      if count != 0:
        if self.model_name == 'pos_only':
          yield ([
              transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
              transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)
          ], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
        else:
          raise NotImplementedError

  def _get_ds(self) -> None:
    if not hasattr(self, 'ds'):
      self.ds = Dataset()

  def partition(self) -> None:
    config.info('partitioning...')
    self._get_ds()

    if self.dataset_name != 'all':
      df_to_split = self.ds.df[self.ds.df['ds'] == self.dataset_name]
    else:
      df_to_split = self.ds.df

    # split x_train, x_test
    if self.test_user and self.test_video:
      _, self.x_test = self.ds.get_rows(self.test_video, self.test_user, self.dataset_name)
      return
    elif self.train_entropy != 'all' and not self.using_auto:
      self.x_train, self.x_test = train_test_split_entropy(df_to_split, self.entropy_type, self.train_entropy, self.perc_test)
    else:
      self.x_train, self.x_test = train_test_split(df_to_split, random_state=1, test_size=self.perc_test)
    # split x_train, x_val
    self.x_train, self.x_val = train_test_split(self.x_train, random_state=1, test_size=0.125) # 0.125 * 0.8 = 0.1



  def train(self) -> None:
    assert not self.using_auto, "train(): train_entropy should not be auto"
    config.info('train()')
    config.info('model_dir=' + self.model_dir)
    if exists(self.model_weights):
      config.info('train() previous done')
      sys.exit()

    # create x_train_wins, x_val_wins
    self.partition()
    self.x_train_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id
    } for row in self.x_train.iterrows()\
      for trace_id in range(self.init_window, row[1]['traject'].shape[0] -self.end_window)]
    self.x_val_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id,
    } for row in self.x_val.iterrows()\
      for trace_id in range(self.init_window, row[1]['traject'].shape[0] -self.end_window)]

    # fit
    config.info('creating model ...')
    if self.dry_run:
      return
    if not exists(self.model_dir):
      os.makedirs(self.model_dir)
    model = self.create_model()
    assert model
    steps_per_ep_train = np.ceil(len(self.x_train_wins) / config.BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(self.x_val_wins) / config.BATCH_SIZE)
    csv_logger_f = join(self.model_dir, 'train_results.csv')
    csv_logger = CSVLogger(csv_logger_f)
    tb_callback = TensorBoard(log_dir=f'{self.model_dir}/logs')
    model_checkpoint = ModelCheckpoint(self.model_weights,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1)
    if self.model_name == 'pos_only':
      generator = self.generate_batchs(self.x_train_wins)
      validation_data = self.generate_batchs(self.x_val_wins)
      model.fit_generator(generator=generator,
                          verbose=1,
                          steps_per_epoch=steps_per_ep_train,
                          epochs=self.epochs,
                          callbacks=[csv_logger, model_checkpoint, tb_callback],
                          validation_data=validation_data,
                          validation_steps=steps_per_ep_validate)
    else:
      raise NotImplementedError

  def evaluate(self) -> None:
    config.info('evaluate()')
    config.info('model_dir=' + self.model_dir)
    self.partition()
    self.x_test_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id,
        'actS_c': row[1]['actS_c']
    } for row in self.x_test.iterrows()\
      for trace_id in range(self.init_window, row[1]['traject'].shape[0] -self.end_window)]
    fmt = '''x_train has {} trajectories: {} low, {} medium, {} hight
             x_val has {} trajectories: {} low, {} medium, {} hight
             x_test has {} trajectories: {} low, {} medium, {} hight'''
    config.info(fmt.format(*count_entropy(self.x_train, self.entropy_type),
                           *count_entropy(self.x_val,   self.entropy_type),
                           *count_entropy(self.x_test,  self.entropy_type)))

    if not self.model_fullname in self.ds.df.columns:
      empty = pd.Series([{} for _ in range(len(self.ds.df))]).astype(object)
      self.ds.df[self.model_fullname] = empty

    # creating model
    config.info('creating model ...')
    if self.using_auto:
      prefix = join(config.DATADIR, f'{self.model_name},{self.dataset_name},actS,')
      model_weights_low = join(prefix + 'low', 'weights.hdf5')
      model_weights_medium = join(prefix + 'medium', 'weights.hdf5')
      model_weights_hight = join(prefix + 'hight', 'weights.hdf5')
      config.info('model_weights_low=' + model_weights_low)
      config.info('model_weights_medium=' + model_weights_medium)
      config.info('model_weights_hight=' + model_weights_hight)
    else:
      model_weights = join(self.model_dir, 'weights.hdf5')
      config.info(f'model_weights={model_weights}')

    if self.dry_run:
      return

    if not exists(self.model_dir):
      os.makedirs(self.model_dir)
    if self.using_auto:
      assert exists(model_weights_low)
      assert exists(model_weights_medium)
      assert exists(model_weights_hight)
      model_low = self.create_model()
      model_low.load_weights(model_weights_low)
      model_medium = self.create_model()
      model_medium.load_weights(model_weights_medium)
      model_hight = self.create_model()
      model_hight.load_weights(model_weights_hight)
      self.threshold_medium, self.threshold_hight = get_class_thresholds(self.ds.df, 'actS')
    else:
      model = self.create_model()
      assert exists(model_weights)
      model.load_weights(model_weights)

    # predict by each pred_windows
    tb_callback = TensorBoard(log_dir=f'{self.model_dir}/logs')
    for ids in tqdm(self.x_test_wins, desc='position predictions'):
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']

      if self.model_name == 'pos_only':
        encoder_pos_inputs_for_sample = np.array(
            [self.ds.get_traces(video, user, self.dataset_name)[x_i - self.m_window:x_i]])
        decoder_pos_inputs_for_sample = np.array(
            [self.ds.get_traces(video, user, self.dataset_name)[x_i:x_i + 1]])
      else:
        raise NotImplementedError

      if self.using_auto:
        # actS_c
        if self.train_entropy == 'auto':
          actS_c = ids['actS_c']
        elif self.train_entropy == 'auto_m_window':
          window = self.ds.get_traces(video, user, self.dataset_name)[x_i - self.m_window:x_i]
          a_ent = calc_actual_entropy(window)
          actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_hight)
        elif self.train_entropy == 'auto_since_start':
          window = self.ds.get_traces(video, user, self.dataset_name)[0:x_i]
          a_ent = calc_actual_entropy(window)
          actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_hight)
        else:
          raise RuntimeError()
        if actS_c == 'low':
          model = model_low
        elif actS_c == 'medium':
          model = model_medium
        elif actS_c == 'hight':
          model = model_hight
        else:
          raise NotImplementedError

      # predict
      if self.model_name == 'pos_only':
        model_pred = model.predict([
            transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
            transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)
        ], callbacks=[tb_callback])[0]
        model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
      else:
        raise NotImplementedError

      # save prediction
      traject_row = self.ds.df.loc[(self.ds.df['video'] == video) & (self.ds.df['user'] == user)]
      assert not traject_row.empty
      index = traject_row.index[0]
      traject_row.loc[index, self.model_fullname][x_i] = model_prediction

    # save on df
    self.ds.dump()

  def compare_results(self) -> None:
    self._get_ds()

    # range_win = range(self.h_window)
    range_win = range(self.h_window)[::4]
    columns = ['model_name', 'S_type', 'S_class']
    s_types = ['actS_c', 'hmpS_c']
    s_classes = ['low', 'medium', 'nohight', 'hight']

    # create df_res
    self.df_res = pd.DataFrame(columns=columns + list(range_win), dtype=np.float32)

    # create df_res for debug with random data
    # models = ['pos_only,all,actS_c,low', 'pos_only,all,actS_c,medium', 'pos_only,all,actS_c,hight', 'pos_only,david,actS_c,medium', 'pos_only,david,actS_c,hight', 'pos_only,david,actS_c,nohight', 'pos_only,david,hmpS_c,nohight', 'pos_only,all,hmpS_c,medium', 'pos_only,david,,']
    # iters = [models,  s_types, s_classes]
    # index = pd.MultiIndex.from_product(iters, names=columns)
    # data_cols = np.random.randn(np.prod(list(map(len,iters))), self.h_window)
    # self.df_res = pd.DataFrame(data_cols, index=index)
    # self.df_res.reset_index(inplace=True)
    # IPython.display.display(self.df_res)

    # create targets in format (model, s_type, s_class, mask)
    models_cols = sorted([col for col in self.ds.df.columns if col.startswith(self.model_name)])
    config.info(f"processing results from models: [{', '.join(models_cols)}]")
    if self.dataset_name == 'all':
      for ds_name in config.ARGS_DS_NAMES[1:]:
        models_cols = [col for col in models_cols if ds_name not in col]
    else:
        models_cols = [col for col in models_cols if col.startswith(f'{self.model_name},{self.dataset_name}')]
    config.info(f"processing results from models: [{', '.join(models_cols)}]")
    targets = [(model, 'all', 'all', pd.Series(True, index=self.ds.df.index)) for model in models_cols]
    for model, s_type, s_class in product(models_cols, s_types, s_classes):
      model_split = model.split(',')
      if len(model_split) > 1 and model_split[2] and model_split[3]:
        if (model_split[2] == 'actS_c' and s_type == 'hmpS_c') or (model_split[2] == 'hmpS_c' and  s_type == 'actS_c') :
          continue # skip 'pos_only,david,actS,ANY' for 'hmpS_c,ANY'
        if ('auto' in model and s_type == 'hmpS_c'):
          continue # skip 'pos_only,ANY,actS,auto' for 'hmpS_c,ANY'
      if s_class == 'nohight':
        # TODO: filter mask empty
        targets.append((model, s_type, s_class, self.ds.df[s_type] != 'hight'))
      else:
        targets.append((model, s_type, s_class, self.ds.df[s_type] == s_class))

    # fill df_res from moldel results column at df
    def _calc_wins_error(df_wins_cols, errors_per_timestamp) -> None:
      traject_index = df_wins_cols.name
      traject = self.ds.df.loc[traject_index, 'traject']
      win_pos_l = df_wins_cols.index
      for win_pos in win_pos_l:
        pred_win = df_wins_cols[win_pos]
        if isinstance(df_wins_cols[win_pos], float):
          break # TODO: review why some pred ends at 51
        true_win = traject[win_pos + 1:win_pos + self.h_window + 1]
        for t in range_win:
          if t not in errors_per_timestamp:
            errors_per_timestamp[t] = []
          try:
            errors_per_timestamp[t].append(METRIC(true_win[t], pred_win[t]))
          except:
            print('error')
    for model, s_type, s_class, mask in targets:
      # create df_win by expading all model predictions
      not_empty = self.ds.df[model].apply(lambda x: len(x) != 0)
      model_srs = self.ds.df.loc[not_empty & mask, model]
      if len(model_srs) == 0:
        config.error(f"skipping {model=} {s_type=} {s_class=}")
        continue
      model_df_wins = pd.DataFrame.from_dict(model_srs.values.tolist())
      model_df_wins.index = model_srs.index

      # df_wins.apply by column and add errors_per_timestamp
      errors_per_timestamp = {idx: [] for idx in range_win}
      model_df_wins.apply(_calc_wins_error, axis=1, args=(errors_per_timestamp, ))
      # model_df_wins[:2].apply(_calc_wins_error, axis=1, args=(errors_per_timestamp, ))
      newid = len(self.df_res)
      # save df_res for s_type, s_class
      # avg_error_per_timestamp = [np.mean(errors_per_timestamp[t]) for t in range_win ]
      avg_error_per_timestamp = [
          np.mean(errors_per_timestamp[t]) if len(errors_per_timestamp[t]) else np.nan
          for t in range_win
      ]
      self.df_res.loc[newid, ['model_name', 'S_type', 'S_class']] = [model, s_type, s_class]
      self.df_res.loc[newid, range_win] = avg_error_per_timestamp
    self._show_compare_results()

  def _show_compare_results(self, df_res=None) -> None:
    df_res = self.df_res if df_res is None else df_res
    range_win = range(self.h_window)[::4]
    # create vis table
    assert len(df_res), 'run -evaluate first'
    props = 'text-decoration: underline'
    output = df_res.dropna()\
      .style\
      .background_gradient(axis=0, cmap='coolwarm')\
      .highlight_min(subset=list(range_win), props=props)\
      .highlight_max(subset=list(range_win), props=props)
    if 'ipykernel' in sys.modules:
      IPython.display.display(output)
    else:
      html_file = join(config.DATADIR, 'compare_results.html')
      output.to_html(html_file)
      print(pathlib.Path(html_file).as_uri())

  def show_train_test_split(self) -> None:
    self.partition()
    self.x_train['partition'] = 'train'
    self.x_test['partition'] = 'test'
    self.ds.df = pd.concat([self.x_train, self.x_test])
    self.ds.show_histogram(facet='partition')