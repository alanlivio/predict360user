import os
import sys
from contextlib import redirect_stderr
from dataclasses import dataclass
from os.path import exists, join
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from sklearn.model_selection import train_test_split

from users360.head_motion_prediction.Utils import (all_metrics,
                                                   cartesian_to_eulerian,
                                                   eulerian_to_cartesian)

from .entropy import *
from .trajects import *

METRIC = all_metrics['orthodromic']
RATE = 0.2
BATCH_SIZE = 128.0

def get_train_test_split(df: pd.DataFrame, entropy: str,
                         perc_test: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  args = {'test_size': perc_test, 'random_state': 1}
  if entropy.startswith('auto'):
    raise RuntimeError()
  if entropy != 'all':
    if entropy.endswith('_hmp'):
      entropy = entropy.removesuffix('_hmp')
      if entropy == 'nohight':
        df = df[df['hmpS_c'] != 'hight']
      else:
        df = df[df['hmpS_c'] == entropy]
    else:
      if entropy == 'nohight':
        df = df[df['actS_c'] != 'hight']
      else:
        df = df[df['actS_c'] == entropy]
  return train_test_split(df, **args)


def count_traject_entropy_classes(df) -> tuple[int, int, int, int]:
  a_len = len(df)
  l_len = len(df[df['actS_c'] == 'low'])
  m_len = len(df[df['actS_c'] == 'medium'])
  h_len = len(df[df['actS_c'] == 'hight'])
  return a_len, l_len, m_len, h_len


def show_train_test_split(df: pd.DataFrame, entropy: str, perc_test: float) -> None:
  x_train, x_test = get_train_test_split(df, entropy, perc_test)
  x_train['partition'] = 'train'
  x_test['partition'] = 'test'
  df = pd.concat([x_train, x_test])
  show_trajects_entropy(df, facet='partition')


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
    if self.train_entropy != 'all':
      self.entropy_type = 'hmpS' if self.train_entropy.endswith('hmp') else 'actS'
      self.train_entropy = self.train_entropy.removesuffix('_hmp')
    # if any filter, use model_fullname with ','
    if self.dataset_name != 'all' or self.train_entropy != 'all':
      self.model_fullname = f'{self.model_name},{self.dataset_name},{self.entropy_type},{self.train_entropy}'
    else:
      self.model_fullname = self.model_name
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
              get_traces(self.df, video, user, 'all')[x_i - self.m_window:x_i])
          decoder_pos_inputs_for_batch.append(get_traces(self.df, video, user, 'all')[x_i:x_i + 1])
          decoder_outputs_for_batch.append(
              get_traces(self.df, video, user, 'all')[x_i + 1:x_i + self.h_window + 1])
        else:
          raise NotImplementedError
        count += 1
        if count == BATCH_SIZE:
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

  def partition(self) -> None:
    config.info('partioning...')
    self.df = get_df_trajects()
    df_to_split = self.df[self.df['ds'] == self.dataset_name] \
      if self.dataset_name != 'all' else self.df
    entropy = 'all' if self.using_auto else self.train_entropy
    self.x_train, self.x_test = get_train_test_split(df_to_split, entropy, self.perc_test)
    if self.test_user and self.test_video:
      self.x_test = get_rows(self.df, self.test_video, self.test_user, self.dataset_name)
    self.x_train_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id
    } for row in self.x_train.iterrows()\
      for trace_id in range( self.init_window, row[1]['traject'].shape[0] -self.end_window)]
    self.x_test_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id,
        'actS_c': row[1]['actS_c']
    } for row in self.x_test.iterrows()\
      for trace_id in range(self.init_window, row[1]['traject'].shape[0] -self.end_window)]
    self.x_val = self.x_test
    self.x_val_wins = self.x_test_wins

  def train(self) -> None:
    assert not self.using_auto, "train(): train_entropy should not be auto"
    config.info('train()')
    config.info('model_dir=' + self.model_dir)
    if exists (self.model_weights):
      config.info('train() previous done')
      sys.exit()
    self.partition()
    fmt = 'x_train has {} trajectories: {} low, {} medium, {} hight'
    config.info(fmt.format(*count_traject_entropy_classes(self.x_train)))
    fmt = 'x_val has {} trajectories: {} low, {} medium, {} hight'
    config.info(fmt.format(*count_traject_entropy_classes(self.x_val)))

    with redirect_stderr(open(os.devnull, 'w')):
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
      os.environ['CUDA_VISIBLE_DEVICES'] = 0
      import tensorflow.keras as keras
      from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

    config.info('creating model ...')
    if self.dry_run:
      return
    if not exists(self.model_dir):
      os.makedirs(self.model_dir)
    model = self.create_model()
    assert model
    steps_per_ep_train = np.ceil(len(self.x_train_wins) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(self.x_val_wins) / BATCH_SIZE)
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
    fmt = 'x_test has {} trajectories: {} low, {} medium, {} hight'
    config.info(fmt.format(*count_traject_entropy_classes(self.x_test)))

    if not self.model_fullname in self.df.columns:
      empty = pd.Series([{} for _ in range(len(self.df))]).astype(object)
      self.df[self.model_fullname] = empty

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
    else:
      model = self.create_model()
      assert exists(model_weights)
      model.load_weights(model_weights)

    # predict by each pred_windows
    threshold_medium, threshold_hight = calc_column_thresholds(self.df, 'actS')
    for ids in tqdm(self.x_test_wins, desc='position predictions'):
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']

      if self.model_name == 'pos_only':
        encoder_pos_inputs_for_sample = np.array(
            [get_traces(self.df, video, user, self.dataset_name)[x_i - self.m_window:x_i]])
        decoder_pos_inputs_for_sample = np.array(
            [get_traces(self.df, video, user, self.dataset_name)[x_i:x_i + 1]])
      else:
        raise NotImplementedError

      if self.using_auto:
        # actS_c
        if self.train_entropy == 'auto':
          actS_c = ids['actS_c']
        elif self.train_entropy == 'auto_m_window':
          window = get_traces(self.df, video, user, self.dataset_name)[x_i - self.m_window:x_i]
          a_ent = calc_actual_entropy(window)
          actS_c = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        elif self.train_entropy == 'auto_since_start':
          window = get_traces(self.df, video, user, self.dataset_name)[0:x_i]
          a_ent = calc_actual_entropy(window)
          actS_c = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
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
        ])[0]
        model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
      else:
        raise NotImplementedError

      # save prediction
      traject_row = self.df.loc[(self.df['video'] == video) & (self.df['user'] == user)]
      assert not traject_row.empty
      index = traject_row.index[0]
      traject_row.loc[index,self.model_fullname][x_i] = model_prediction

    # save on df
    dump_df_trajects(self.df)


  def compare_results(self) -> None:
    return