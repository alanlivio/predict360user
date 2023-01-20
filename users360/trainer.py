import os
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from os.path import exists, join
from typing import Any, Generator, Iterable

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


def count_traject_entropy_classes(df) -> tuple[int, int, int, int]:
  a_len = len(df)
  l_len = len(df[df['traject_entropy_class'] == 'low'])
  m_len = len(df[df['traject_entropy_class'] == 'medium'])
  h_len = len(df[df['traject_entropy_class'] == 'hight'])
  return a_len, l_len, m_len, h_len


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


def generate_batchs(model_name, df: pd.DataFrame, wins: Iterable, m_window, h_window) -> Generator:
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
      if model_name == 'pos_only':
        encoder_pos_inputs_for_batch.append(get_traces(df, video, user, 'all')[x_i - m_window:x_i])
        decoder_pos_inputs_for_batch.append(get_traces(df, video, user, 'all')[x_i:x_i + 1])
        decoder_outputs_for_batch.append(
            get_traces(df, video, user, 'all')[x_i + 1:x_i + h_window + 1])
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

  def _partition(self) -> None:
    config.info('partioning...')
    self.df_trajects = get_df_trajects()
    df_to_split = self.df_trajects[self.df_trajects['ds'] == self.dataset_name] \
      if self.dataset_name != 'all' else self.df_trajects
    self.x_train, self.x_test = get_train_test_split(df_to_split, self.train_entropy, self.perc_test)
    self.x_train_wins = [{
        'video': row[1]['ds_video'],
        'user': row[1]['ds_user'],
        'trace_id': trace_id
    } for row in self.x_train.iterrows()\
      for trace_id in range( self.init_window, row[1]['traject'].shape[0] -self.end_window)]
    self.x_test_wins = [{
        'video': row[1]['ds_video'],
        'user': row[1]['ds_user'],
        'trace_id': trace_id,
        'traject_entropy_class': row[1]['traject_entropy_class']
    } for row in self.x_test.iterrows()\
      for trace_id in range( self.init_window, row[1]['traject'].shape[0] -self.end_window)]

  def train(self) -> None:
    config.info('train()')
    config.info(self)
    self._partition()

    with redirect_stderr(open(os.devnull, 'w')):
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
      import tensorflow.keras as keras
      from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
    model = create_model(self.model_name, self.m_window, self.h_window)
    assert model
    steps_per_ep_train = np.ceil(len(self.x_train_wins) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(self.x_test_wins) / BATCH_SIZE)
    csv_logger_f = join(self.model_dir, 'train_results.csv')
    csv_logger = CSVLogger(csv_logger_f)
    tb_callback = TensorBoard(log_dir=f'{self.model_dir}/logs')
    model_checkpoint = ModelCheckpoint(self.model_weights,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1)
    if self.model_name == 'pos_only':
      generator = generate_batchs(self.model_name, self.df_trajects, self.x_train_wins,
                                  self.m_window, self.h_window)
      validation_data = generate_batchs(self.model_name, self.df_trajects, self.x_test_wins,
                                        self.m_window, self.h_window)
      model.fit_generator(generator=generator,
                          verbose=1,
                          steps_per_epoch=steps_per_ep_train,
                          epochs=self.epochs,
                          callbacks=[csv_logger, model_checkpoint, tb_callback],
                          validation_data=validation_data,
                          validation_steps=steps_per_ep_validate)
    else:
      raise NotImplementedError