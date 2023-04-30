import os
import pickle
from contextlib import redirect_stderr
from os.path import exists, isdir, join
from typing import Generator

import absl.logging
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Model
from tqdm.auto import tqdm

from . import config
from .dataset import Dataset, count_entropy, get_class_thresholds
from .fov import compute_orthodromic_distance
from .models import *

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def filter_df_by_entropy(df: pd.DataFrame, entropy_type: str, train_entropy: str) -> pd.DataFrame:
  if train_entropy == 'all':
    filter_df = df
  elif train_entropy == 'nohight':
    filter_df = df[df[entropy_type + '_c'] != 'hight']
  elif train_entropy == 'nolow':
    filter_df = df[df[entropy_type + '_c'] != 'low']
  else:
    filter_df = df[df[entropy_type + '_c'] == train_entropy]
  return filter_df


class Trainer():

  def __init__(self,
               model_name='pos_only',
               dataset_name='all',
               h_window=25,
               init_window=30,
               m_window=5,
               test_size=0.2,
               gpu_id=0,
               train_entropy='all',
               epochs=config.DEFAULT_EPOCHS,
               dry_run=False) -> None:
    self.model_name = model_name
    self.dataset_name = dataset_name
    self.train_entropy = train_entropy
    assert self.model_name in config.ARGS_MODEL_NAMES
    assert self.dataset_name in config.ARGS_DS_NAMES
    assert self.train_entropy in config.ARGS_ENTROPY_NAMES + config.ARGS_ENTROPY_AUTO_NAMES
    self.using_auto = self.train_entropy.startswith('auto')
    self.h_window = h_window
    self.init_window = init_window
    self.m_window = m_window
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if model_name == 'pos_only':
      self.model = PosOnly(m_window, h_window)
    elif model_name == 'pos_only_3d':
      self.model = PosOnly3D(m_window, h_window)
    elif model_name == 'no_motion':
      self.model = NoMotion(m_window, h_window)
    elif self.using_auto:
      self.model = PosOnly_Auto(m_window, h_window)
    else:
      raise RuntimeError
    self.test_size = test_size
    self.epochs = epochs
    self.dry_run = dry_run
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
    self.ckpt_path = join(self.model_dir, 'cp-{epoch:04d}-{loss:.2f}.ckpt')
    self.end_window = self.h_window
    config.info(self.__str__())

  def __str__(self) -> str:
    return "Trainer(" + ", ".join(f'{elem}={getattr(self, elem)}' for elem in [
        'model_name', 'dataset_name', 'h_window', 'init_window', 'm_window', 'test_size',
        'train_entropy', 'epochs', 'dry_run'
    ]) + ")"

  def generate_batchs(self, wins: list) -> Generator:
    while True:
      shuffle(wins, random_state=1)
      for count, _ in enumerate(wins[::config.BATCH_SIZE]):
        end = count+config.BATCH_SIZE if count+config.BATCH_SIZE <= len(wins) else len(wins)
        traces_l = [self.ds.get_traces(win['video'],win['user']) for win in wins[count:end]]
        x_i_l = [win['trace_id'] for win in wins[count:end]]
        yield self.model.generate_batch(traces_l,x_i_l)

  def _get_ds(self) -> None:
    if not hasattr(self, 'ds'):
      self.ds = Dataset()

  def partition(self) -> None:
    config.info('partitioning...')
    self._get_ds()
    df = self.ds.df if self.dataset_name == 'all' \
       else self.ds.df[self.ds.df['ds'] == self.dataset_name]
    # split x_train, x_test (0.2)
    self.x_train, self.x_test = \
      train_test_split(df, random_state=1, test_size=self.test_size, stratify=df[self.entropy_type + '_c'])
    # split x_train, x_val (0.125 * 0.8 = 0.1)
    self.x_train, self.x_val = \
      train_test_split(self.x_train,random_state=1, test_size=0.125, stratify=self.x_train[self.entropy_type + '_c'])

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
    if exists(self.train_csv_log_f):
      done_epochs = int(pd.read_csv(self.train_csv_log_f).iloc[-1]['epoch'])
      if done_epochs > self.epochs:
        config.info(f'{self.train_csv_log_f} has {self.epochs}>=epochs. stopping.')
        return
      else:
        config.info(f'train_csv_log_f has {self.epochs}<epochs. continuing from {done_epochs+1}.')
        model = self.model.load(self.ckpt_path)
        initial_epoch = done_epochs
    else:
      model = self.model.build()
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
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model_checkpoint = ModelCheckpoint(self.ckpt_path,
                                       save_weights_only=True,
                                       verbose=1)
    callbacks = [csv_logger, model_checkpoint]
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
              # use_multiprocessing=True
              )

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
    if self.using_auto:
      prefix = join(config.DATADIR, f'{self.model_name},{self.dataset_name},actS,')
      threshold_medium, threshold_hight = get_class_thresholds(self.ds.df, 'actS')
      self.model.load_models( join(prefix + 'low', 'saved_model'), join(prefix + 'medium', 'saved_model'),
                      join(prefix + 'hight', 'saved_model'), threshold_medium, threshold_hight)
    elif self.model_name != 'no_motion':
      ckpt_path = join(self.model_dir, 'cp-{epoch:04d}.ckpt')
      self.model.load(ckpt_path)

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
      # predict
      model_prediction = self.model.predict(self.ds.get_traces(video, user), x_i)
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
                if isdir(join(config.DATADIR, dir_name))
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

  def compare_evaluate(self, load_saved=False) -> None:
    # horizon timestamps to be calculated
    range_win = range(self.h_window)[::4]

    # create df_res
    columns = ['model_name', 'S_type', 'S_class']
    if not hasattr(self, 'df_res'):
      if load_saved:
        self._load_compare_evaluate()
      else:
        self.df_res = pd.DataFrame(columns=columns + list(range_win), dtype=np.float32)

    # create targets in format (model, s_type, s_class, mask)
    self.partition()
    models_cols = sorted([
      col for col in self.x_test.columns \
      if any(m_name in col for m_name in config.ARGS_MODEL_NAMES)\
      and not any(ds_name in col for ds_name in config.ARGS_DS_NAMES[1:])\
      and not any(self.df_res['model_name'] == col) # already calculated
      ])
    # models_cols = ['no_motion']
    targets = []
    for model in models_cols:
      assert len(self.x_test) == len(self.x_test[model].apply(lambda x: len(x) != 0))
      targets.append((model, 'all', 'all', pd.Series(True, index=self.x_test.index)))
      targets.append(
          (model, self.entropy_type, 'low', self.x_test[self.entropy_type + '_c'] == 'low'))
      targets.append(
          (model, self.entropy_type, 'nolow', self.x_test[self.entropy_type + '_c'] != 'low'))
      targets.append(
          (model, self.entropy_type, 'medium', self.x_test[self.entropy_type + '_c'] == 'medium'))
      targets.append(
          (model, self.entropy_type, 'nohight', self.x_test[self.entropy_type + '_c'] != 'hight'))
      targets.append(
          (model, self.entropy_type, 'hight', self.x_test[self.entropy_type + '_c'] == 'hight'))

    # fill df_res from moldel results column at df
    def _calc_wins_error(df_wins_cols, errors_per_timestamp) -> None:
      traject_index = df_wins_cols.name
      traject = self.x_test.loc[traject_index, 'traces']
      win_pos_l = df_wins_cols.index
      for win_pos in win_pos_l:
        pred_win = df_wins_cols[win_pos]
        if isinstance(df_wins_cols[win_pos], float):
          break  # TODO: review why some pred ends at 51
        true_win = traject[win_pos + 1:win_pos + self.h_window + 1]
        for t in range_win:
          if t not in errors_per_timestamp:
            errors_per_timestamp[t] = []
          errors_per_timestamp[t].append(compute_orthodromic_distance(true_win[t], pred_win[t]))

    config.info(f"compare results for models: {', '.join(models_cols)}")
    config.info(f"for each model, compare users: {config.ARGS_ENTROPY_NAMES[:6]}")
    for model, s_type, s_class, mask in tqdm(targets):
      # print(model, s_type, s_class, mask.values.sum(), type(mask))
      # create df_win with columns as timestamps
      model_srs = self.x_test.loc[mask, model]
      if len(model_srs) == 0:
        raise RuntimeError(f"empty {model=}, {s_type}={s_class}")
      model_df_wins = pd.DataFrame.from_dict(model_srs.values.tolist())
      model_df_wins.index = model_srs.index
      # calc errors_per_timestamp from df_wins
      errors_per_timestamp = {idx: [] for idx in range_win}
      model_df_wins.apply(_calc_wins_error, axis=1, args=(errors_per_timestamp, ))
      newid = len(self.df_res)
      avg_error_per_timestamp = [
          np.mean(errors_per_timestamp[t]) if len(errors_per_timestamp[t]) else np.nan
          for t in range_win
      ]
      self.df_res.loc[newid, ['model_name', 'S_type', 'S_class']] = [model, s_type, s_class]
      self.df_res.loc[newid, range_win] = avg_error_per_timestamp
    self._show_compare_evaluate()

  RES_EVALUATE = os.path.join(config.DATADIR, 'df_res_evaluate.pickle')

  def _save_compare_evaluate(self) -> None:
    with open(self.RES_EVALUATE, 'wb') as f:
      config.info(f'saving df_res to {self.RES_EVALUATE}')
      pickle.dump(self.df_res, f)

  def _load_compare_evaluate(self) -> None:
    with open(self.RES_EVALUATE, 'rb') as f:
      config.info(f'loading df_res from {self.RES_EVALUATE}')
      self.df_res = pickle.load(f)

  def _show_compare_evaluate(self, df_res=None) -> None:
    df_res = self.df_res if df_res is None else df_res
    range_win = range(self.h_window)[::4]
    # create vis table
    assert len(df_res), 'run -evaluate first'
    props = 'text-decoration: underline'
    output = df_res.dropna()\
      .sort_values(by=list(range_win))\
      .style\
      .background_gradient(axis=0, cmap='coolwarm')\
      .highlight_min(subset=list(range_win), props=props)\
      .highlight_max(subset=list(range_win), props=props)
    config.show_or_save(output, 'compare_evaluate')

  def show_train_test_split(self) -> None:
    self.partition()
    self.x_train['partition'] = 'train'
    self.x_test['partition'] = 'test'
    self.ds.df = pd.concat([self.x_train, self.x_test])
    self.ds.show_histogram(facet='partition')