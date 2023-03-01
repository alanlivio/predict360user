import os
from contextlib import redirect_stderr
from os.path import exists, join
from typing import Any, Generator

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm.auto import tqdm

with redirect_stderr(open(os.devnull, 'w')):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  import tensorflow.keras as keras
  from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from . import config
from .dataset import *
from .head_motion_prediction.position_only_baseline import (
    create_pos_only_model, metric_orth_dist)
from .head_motion_prediction.Utils import (all_metrics, cartesian_to_eulerian,
                                           eulerian_to_cartesian)
from .utils.fov import *

METRIC = all_metrics['orthodromic']
tqdm.pandas()


def filter_df_by_entropy(df: pd.DataFrame, entropy_type: str, train_entropy: str) -> pd.DataFrame:
  if train_entropy == 'all':
    df = df
  elif train_entropy == 'nohight':
    df = df[df[entropy_type + '_c'] != 'hight']
  elif train_entropy == 'nolow':
    df = df[df[entropy_type + '_c'] != 'low']
  else:
    df = df[df[entropy_type + '_c'] == train_entropy]
  return df


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


class Trainer():
  def __init__(self,
               model_name='pos_only',
               dataset_name='all',
               h_window=25,
               init_window=30,
               m_window=5,
               test_size=0.2,
               train_entropy='all',
               epochs=config.DEFAULT_EPOCHS,
               dry_run=False) -> None:
    self.model_name = model_name
    self.dataset_name = dataset_name
    self.h_window = h_window
    self.init_window = init_window
    self.m_window = m_window
    self.test_size = test_size
    self.train_entropy = train_entropy
    self.epochs = epochs
    self.dry_run = dry_run
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
    self.train_csv_log_f = join(self.model_dir, 'train_results.csv')
    self.model_file = join(self.model_dir, 'model.h5')
    self.end_window = self.h_window
    config.info(self.__str__())

  def __str__(self) -> str:
    return "Trainer(" + ", ".join(f'{elem}={getattr(self, elem)}' for elem in [
        'model_name', 'dataset_name', 'h_window', 'init_window', 'm_window', 'test_size',
        'train_entropy', 'epochs', 'dry_run'
    ]) + ")"

  def create_model(self) -> Any:
    if self.model_name == 'pos_only':
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
      shuffle(wins, random_state=1)
      for ids in wins:
        user = ids['user']
        video = ids['video']
        x_i = ids['trace_id']
        # load the data
        if self.model_name == 'pos_only':
          encoder_pos_inputs_for_batch.append(
              self.ds.get_traces(video, user)[x_i - self.m_window:x_i])
          decoder_pos_inputs_for_batch.append(self.ds.get_traces(video, user)[x_i:x_i + 1])
          decoder_outputs_for_batch.append(
              self.ds.get_traces(video, user)[x_i + 1:x_i + self.h_window + 1])
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
    df = self.ds.df if self.dataset_name == 'all' else self.ds.df[self.ds.df['ds'] ==
                                                                  self.dataset_name]
    # split x_train, x_test
    self.x_train, self.x_test = train_test_split(df,
                                                 random_state=1,
                                                 test_size=self.test_size,
                                                 stratify=df[self.entropy_type + '_c'])
    # split x_train, x_val
    self.x_train, self.x_val = train_test_split(self.x_train,
                                                random_state=1,
                                                test_size=0.125,
                                                stratify=self.x_train[self.entropy_type +
                                                                      '_c'])  # 0.125 * 0.8 = 0.1

  def train(self) -> None:
    config.info('train()')
    assert not self.using_auto, "train(): train_entropy should not be auto"
    config.info('model_dir=' + self.model_dir)

    # partition
    self.partition()
    if self.train_entropy != 'all':
      pre_filter_x_train_len = len(self.x_train)
      pre_filter_epochs = self.epochs
      config.info('train_entropy != all, so filtering x_train, x_val')
      self.x_train = filter_df_by_entropy(self.x_train, self.entropy_type, self.train_entropy)
      self.x_val = filter_df_by_entropy(self.x_val, self.entropy_type, self.train_entropy)
      config.info('x_train filtred has {} trajectories: {} low, {} medium, {} hight'.format(
          *count_entropy(self.x_train, self.entropy_type)))
      config.info('x_val filtred has {} trajectories: {} low, {} medium, {} hight'.format(
          *count_entropy(self.x_val, self.entropy_type)))
      pos_filter_x_train_len = len(self.x_train)
      # given pre_filter_x_train_len < pos_filter_x_train_len, increase epochs
      self.epochs = self.epochs + round(
          0.1 * self.epochs * pre_filter_x_train_len / pos_filter_x_train_len)
      config.info('given x_train filtred, compensate by changing epochs from {} to {} '.format(
          pre_filter_epochs, self.epochs))
    else:
      config.info('x_train has {} trajectories: {} low, {} medium, {} hight'.format(
          *count_entropy(self.x_train, self.entropy_type)))
      config.info('x_val has {} trajectories: {} low, {} medium, {} hight'.format(
          *count_entropy(self.x_val, self.entropy_type)))

    if self.dry_run:
      return

    # check model
    config.info('creating model ...')
    model: keras.models.Model
    co_metric = {'metric_orth_dist': metric_orth_dist}
    if exists(self.model_file) and exists(self.train_csv_log_f):
      done_epochs = int(pd.read_csv(self.train_csv_log_f).iloc[-1]['epoch'])
      if done_epochs >= self.epochs:
        config.info(f'{self.train_csv_log_f} has {self.epochs}>=epochs. stopping.')
        return
      else:
        config.info(f'{self.train_csv_log_f} has {self.epochs}<epochs. continuing.')
        model = keras.models.load_model(self.model_file, custom_objects=co_metric)
        initial_epoch = done_epochs
    else:
      model = self.create_model()
      initial_epoch = 0
      if not exists(self.model_dir):
        os.makedirs(self.model_dir)
    assert model

    # create x_train_wins, x_val_wins
    self.x_train_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id
    } for row in self.x_train.iterrows()\
      for trace_id in range(self.init_window, row[1]['traces'].shape[0] -self.end_window)]
    self.x_val_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id,
    } for row in self.x_val.iterrows()\
      for trace_id in range(self.init_window, row[1]['traces'].shape[0] -self.end_window)]

    # fit
    steps_per_ep_train = np.ceil(len(self.x_train_wins) / config.BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(self.x_val_wins) / config.BATCH_SIZE)
    csv_logger = CSVLogger(self.train_csv_log_f, append=True)
    # tb_callback = TensorBoard(log_dir=f'{self.model_dir}/logs')
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10,
        monitor='loss',
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(self.model_file,
                                       mode='auto',
                                       save_freq=1,
                                       multiprocessing=True)
    callbacks = [csv_logger, model_checkpoint, early_stopping_cb]
    if self.model_name == 'pos_only':
      generator = self.generate_batchs(self.x_train_wins)
      validation_data = self.generate_batchs(self.x_val_wins)
      model.fit(x=generator,
                verbose=1,
                steps_per_epoch=steps_per_ep_train,
                validation_data=validation_data,
                validation_steps=steps_per_ep_validate,
                validation_freq=config.BATCH_SIZE,
                epochs=self.epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                use_multiprocessing=True)
    else:
      raise NotImplementedError

  def evaluate(self) -> None:
    config.info('evaluate()')
    config.info('model_dir=' + self.model_dir)

    # partition
    self.partition()
    config.info('x_test has {} trajectories: {} low, {} medium, {} hight'.format(
        *count_entropy(self.x_test, self.entropy_type)))

    if self.dry_run:
      return

    # create model
    config.info('creating model ...')
    model: keras.models.Model
    if self.using_auto:
      prefix = join(config.DATADIR, f'{self.model_name},{self.dataset_name},actS,')
      model_file_low = join(prefix + 'low', 'model.h5')
      model_file_medium = join(prefix + 'medium', 'model.h5')
      model_file_hight = join(prefix + 'hight', 'model.h5')
      config.info('model_file_low=' + model_file_low)
      config.info('model_file_medium=' + model_file_medium)
      config.info('model_file_hight=' + model_file_hight)
    else:
      model_file = join(self.model_dir, 'model.h5')
      config.info(f'model_file={model_file}')

    if self.using_auto:
      assert exists(model_file_low)
      assert exists(model_file_medium)
      assert exists(model_file_hight)
      model_low = keras.models.load_model(model_file_low)
      model_medium = keras.models.load_model(model_file_medium)
      model_hight = keras.models.load_model(model_file_hight)
      self.threshold_medium, self.threshold_hight = get_class_thresholds(self.ds.df, 'actS')
    else:
      assert exists(model_file)
      model = keras.models.load_model(model_file)

    # predict by each x_test_wins
    self.x_test_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id,
        'actS_c': row[1]['actS_c']
    } for row in self.x_test.iterrows()\
      for trace_id in range(self.init_window, row[1]['traces'].shape[0] -self.end_window)]

    if not self.model_fullname in self.ds.df.columns:
      empty = pd.Series([{} for _ in range(len(self.ds.df))]).astype(object)
      self.ds.df[self.model_fullname] = empty

    for ids in tqdm(self.x_test_wins, desc='position predictions'):
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']

      if self.model_name == 'pos_only':
        encoder_pos_inputs_for_sample = np.array(
            [self.ds.get_traces(video, user)[x_i - self.m_window:x_i]])
        decoder_pos_inputs_for_sample = np.array(
          [self.ds.get_traces(video, user)[x_i:x_i + 1]])
      else:
        raise NotImplementedError

      if self.using_auto:
        # actS_c
        if self.train_entropy == 'auto':
          actS_c = ids['actS_c']
        elif self.train_entropy == 'auto_m_window':
          window = self.ds.get_traces(video, user)[x_i - self.m_window:x_i]
          a_ent = calc_actual_entropy(window)
          actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_hight)
        elif self.train_entropy == 'auto_since_start':
          window = self.ds.get_traces(video, user)[0:x_i]
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
        ])[0]
        model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
      else:
        raise NotImplementedError

      # save prediction
      traject_row = self.ds.df.loc[(self.ds.df['video'] == video) & (self.ds.df['user'] == user)]
      assert not traject_row.empty
      index = traject_row.index[0]
      traject_row.loc[index, self.model_fullname][x_i] = model_prediction

    # save on df
    self.ds.dump_column(self.model_fullname)

  def compare_train(self) -> None:
    result_csv = 'train_results.csv'
    # find result_csv files
    csv_df_l = [(dir_name, pd.read_csv(join(config.DATADIR, dir_name, file_name)))
                for dir_name in os.listdir(config.DATADIR)
                if os.path.isdir(join(config.DATADIR, dir_name))
                for file_name in os.listdir(join(config.DATADIR, dir_name))
                if file_name == result_csv]
    csv_df_l = [df.assign(model=dir_name) for (dir_name, df) in csv_df_l]
    assert csv_df_l, f'no data/<model>/{result_csv} files, run -train'

    # plot
    df_compare = pd.concat(csv_df_l)
    fig = px.line(df_compare,
                  x='epoch',
                  y='loss',
                  color='model',
                  title='compare_train_loss',
                  width=800)
    config.show_or_save(fig)
    fig = px.line(df_compare,
                  x='epoch',
                  y='val_loss',
                  color='model',
                  title='compare_train_val_loss',
                  width=800)
    config.show_or_save(fig)

  def compare_evaluate(self) -> None:
    self._get_ds()
    # horizon timestamps to be calculated
    range_win = range(self.h_window)[::4]

    # create df_res
    columns = ['model_name', 'S_type', 'S_class']
    self.df_res = pd.DataFrame(columns=columns + list(range_win), dtype=np.float32)

    # create targets in format (model, s_type, s_class, mask)
    models_cols = sorted([
      col for col in self.ds.df.columns \
      if (col.startswith(self.model_name)) \
      and not any(ds_name in col for ds_name in config.ARGS_DS_NAMES[1:])
    ])
    config.info(f"processing results from models: {', '.join(models_cols)}")
    targets = []
    for model in models_cols:
      targets.append((model, 'all', 'all', pd.Series(True, index=self.ds.df.index)))
      targets.append((model, self.entropy_type, 'low', self.ds.df[self.entropy_type] != 'low'))
      targets.append(
          (model, self.entropy_type, 'medium', self.ds.df[self.entropy_type] != 'medium'))
      targets.append(
          (model, self.entropy_type, 'nohight', self.ds.df[self.entropy_type] != 'hight'))
      targets.append((model, self.entropy_type, 'hight', self.entropy_type == 'hight'))

    # fill df_res from moldel results column at df
    def _calc_wins_error(df_wins_cols, errors_per_timestamp) -> None:
      traject_index = df_wins_cols.name
      traject = self.ds.df.loc[traject_index, 'traces']
      win_pos_l = df_wins_cols.index
      for win_pos in win_pos_l:
        pred_win = df_wins_cols[win_pos]
        if isinstance(df_wins_cols[win_pos], float):
          break  # TODO: review why some pred ends at 51
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
    self._show_compare_evaluate()

  def _show_compare_evaluate(self, df_res=None) -> None:
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
    config.show_or_save(output, 'compare_evaluate.html')

  def show_train_test_split(self) -> None:
    self.partition()
    self.x_train['partition'] = 'train'
    self.x_test['partition'] = 'test'
    self.ds.df = pd.concat([self.x_train, self.x_test])
    self.ds.show_histogram(facet='partition')