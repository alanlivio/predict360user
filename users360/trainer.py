import os
import sys
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from os.path import exists, join
from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from users360.head_motion_prediction.Utils import (all_metrics,
                                                   cartesian_to_eulerian,
                                                   eulerian_to_cartesian)

from .entropy import *
from .trajects import *

METRIC = all_metrics['orthodromic']
RATE = 0.2
BATCH_SIZE = 128.0

def get_train_test_split(df_trajects: pd.DataFrame, entropy: str,
                         perc_test: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  args = {'test_size': perc_test, 'random_state': 1}
  # _users
  if entropy.endswith('_users') and entropy != 'all':
    x_train, x_test = train_test_split(
        df_trajects[df_trajects['user_entropy_class'] == entropy.removesuffix('_users')], **args)
  elif entropy != 'all':
    x_train, x_test = train_test_split(df_trajects[df_trajects['traject_entropy_class'] == entropy],
                                       **args)
  else:
    x_train, x_test = train_test_split(df_trajects, **args)

  return x_train, x_test


def show_train_test_split(df_trajects: pd.DataFrame, entropy: str, perc_test: float) -> None:
  x_train, x_test = get_train_test_split(df_trajects, entropy, perc_test)
  x_train['partition'] = 'train'
  x_test['partition'] = 'test'
  df_trajects = pd.concat([x_train, x_test])
  show_trajects_entropy(df_trajects, facet='partition')
  show_trajects_entropy_users(df_trajects, facet='partition')


def create_model(model_name, m_window, h_window) -> Any:
  if model_name == 'pos_only':
    from users360.head_motion_prediction.position_only_baseline import \
        create_pos_only_model
    return create_pos_only_model(m_window, h_window)
  else:
    raise NotImplementedError

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

def create_pred_windows(x_train: pd.DataFrame, x_test: pd.DataFrame, init_window: int, end_window:int, skip_train = False) -> dict:
  pred_windows = {}
  if not skip_train:
    fmt = 'x_train has {} trajectories: {} low, {} medium, {} hight'
    t_len = len(x_train)
    l_len = len(x_train[x_train['traject_entropy_class'] == 'low'])
    m_len = len(x_train[x_train['traject_entropy_class'] == 'medium'])
    h_len = len(x_train[x_train['traject_entropy_class'] == 'hight'])
    config.info(fmt.format(t_len, l_len, m_len, h_len))
    pred_windows['train'] = [{
        'video': row[1]['ds_video'],
        'user': row[1]['ds_user'],
        'trace_id': trace_id
    } for row in x_train.iterrows() \
      for trace_id in range(
        init_window, row[1]['traject'].shape[0] -end_window)]
    p_len = len(pred_windows['train'])
    config.info("pred_windows['train'] has {} positions".format(p_len))
  fmt = 'x_test has {} trajectories: {} low, {} medium, {} hight'
  t_len = len(x_test)
  l_len = len(x_test[x_test['traject_entropy_class'] == 'low'])
  m_len = len(x_test[x_test['traject_entropy_class'] == 'medium'])
  h_len = len(x_test[x_test['traject_entropy_class'] == 'hight'])
  config.info(fmt.format(t_len, l_len, m_len, h_len))
  pred_windows['test'] = [{
      'video': row[1]['ds_video'],
      'user': row[1]['ds_user'],
      'trace_id': trace_id,
      'traject_entropy_class': row[1]['traject_entropy_class']
  } for row in x_test.iterrows() \
    for trace_id in range(
      init_window, row[1]['traject'].shape[0] -end_window)]
  p_len = len(pred_windows['test'])
  config.info("pred_windows['test'] has {} positions".format(p_len))
  return pred_windows


def generate_batchs(model_name: str, df_trajects: pd.DataFrame, pred_windows: dict, m_window: int, h_window: int) -> Generator:
  while True:
    encoder_pos_inputs_for_batch = []
    # encoder_sal_inputs_for_batch = []
    decoder_pos_inputs_for_batch = []
    # decoder_sal_inputs_for_batch = []
    decoder_outputs_for_batch = []
    count = 0
    np.random.shuffle(pred_windows)
    for ids in pred_windows:
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']
      # Load the data
      if model_name == 'pos_only':
        encoder_pos_inputs_for_batch.append(
            get_traces(df_trajects, video, user, 'all')[x_i - m_window:x_i])
        decoder_pos_inputs_for_batch.append(
            get_traces(df_trajects, video, user, 'all')[x_i:x_i + 1])
        decoder_outputs_for_batch.append(
            get_traces(df_trajects, video, user, 'all')[x_i + 1:x_i + h_window + 1])
      else:
        raise NotImplementedError
      count += 1
      if count == BATCH_SIZE:
        count = 0
        if model_name == 'pos_only':
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
      if model_name == 'pos_only':
        yield ([
            transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
            transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)
        ], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
      else:
        raise NotImplementedError


@dataclass
class Trainer():

  dataset_name: str = 'all'
  model_name: str = 'pos_only'
  train_entropy: str = 'all'
  epochs: int = 100
  h_window: int = 25
  init_window: int = 30
  m_window: int = 5
  perc_test: float = 0.2
  dry_run: bool = False
  model_dir: str = '' # path
  model_weights: str = '' # path

  def __post_init__(self) -> None:
    dataset_suffix = '' if self.dataset_name == 'all' else f'_{self.dataset_name}'
    basedir = join(config.DATADIR, self.model_name + dataset_suffix)
    self.model_dir = basedir + ('' if self.train_entropy == 'all' else
                                f'_{self.train_entropy}_entropy')
    self.model_weights = join(self.model_dir, 'weights.hdf5')
    self.end_window = self.h_window
    if not exists(self.model_dir):
      os.makedirs(self.model_dir)

  def train(self) -> None:
    config.info('train: ' + self.repr())


    # pred_windows
    config.info('partioning...')

    self.df_trajects = get_df_trajects()
    if self.dataset_name != 'all':
      df_trajects = df_trajects[df_trajects['ds'] == self.dataset_name]
    config.info(f'x_train, x_test entropy is {self.train_entropy}')
    x_train, x_test = get_train_test_split(df_trajects, self.train_entropy, self.perc_test)
    pred_windows = create_pred_windows(x_train, x_test)

    with redirect_stderr(open(os.devnull, 'w')):  # pylint: disable=unspecified-encoding
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
      import tensorflow.keras as keras

    steps_per_ep_train = np.ceil(len(pred_windows['train']) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(pred_windows['test']) / BATCH_SIZE)

    # creating model
    config.info('creating model ...')
    # model_weights
    model_weights = join(model_dir, 'weights.hdf5')
    config.info(f'model_weights={model_weights}')
    if self.dry_run:
      return
    model = create_model()
    assert model

    # train
    csv_logger_f = join(model_dir, 'train_results.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
    tb_callback = keras.callbacks.TensorBoard(log_dir=f'{model_dir}/logs')
    model_checkpoint = keras.callbacks.ModelCheckpoint(model_weights,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='auto',
                                                      period=1)
    if self.model_name == 'pos_only':
      model.fit_generator(generator=generate_batchs(df_trajects, pred_windows['train'], h_window=self.h_window),
                          verbose=1,
                          steps_per_epoch=steps_per_ep_train,
                          epochs=self.epochs,
                          callbacks=[csv_logger, model_checkpoint, tb_callback],
                          validation_data=generate_batchs(df_trajects, pred_windows['test'],
                                                          h_window=self.h_window),
                          validation_steps=steps_per_ep_validate)
    else:
      raise NotImplementedError