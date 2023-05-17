import os
import pickle
from os.path import exists, isdir, join
from typing import Generator

import absl.logging
import numpy as np
import pandas as pd
import plotly.express as px
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm.auto import tqdm

from predict360user import config
from predict360user.dataset import (Dataset, calc_actual_entropy,
                                    count_entropy, get_class_name,
                                    get_class_thresholds)
from predict360user.models import (BaseModel, Interpolation, NoMotion, PosOnly,
                                   PosOnly3D)
from predict360user.utils import orth_dist_cartesian

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def filter_df_by_entropy(df: pd.DataFrame, entropy_type: str, train_entropy: str) -> pd.DataFrame:
  if train_entropy == 'all':
    filter_df = df
  elif train_entropy == 'nohigh':
    filter_df = df[df[entropy_type + '_c'] != 'high']
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
               savedir=config.DEFAULT_SAVEDIR) -> None:
    # properties from constructor
    assert model_name in config.ARGS_MODEL_NAMES
    assert dataset_name in config.ARGS_DS_NAMES
    assert train_entropy in config.ARGS_ENTROPY_NAMES + config.ARGS_ENTROPY_AUTO_NAMES
    self.model_name = model_name
    self.dataset_name = dataset_name
    self.train_entropy = train_entropy
    self.savedir = savedir
    self.h_window = h_window
    self.init_window = init_window
    self.m_window = m_window
    self.test_size = test_size
    self.epochs = epochs
    self.end_window = self.h_window
    if gpu_id:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
      config.info(f"set visible cpu to {gpu_id}")
    # properties others
    self.compare_eval_pickle = join(self.savedir, 'df_compare_evaluate.pickle')
    self.using_auto = self.train_entropy.startswith('auto')
    self.entropy_type = 'hmpS' if self.train_entropy.endswith('hmp') else 'actS'
    if self.dataset_name == 'all' and self.train_entropy == 'all':
      self.model_fullname = self.model_name
    elif self.train_entropy == 'all':
      self.model_fullname = f'{self.model_name},{self.dataset_name},,'
    else:
      self.train_entropy = self.train_entropy.removesuffix('_hmp')
      self.model_fullname = f'{self.model_name},{self.dataset_name},{self.entropy_type},{self.train_entropy}'
    self.model_dir = join(self.savedir, self.model_fullname)
    self.train_csv_log_f = join(self.model_dir, 'train_results.csv')
    self.model_path = join(self.model_dir, 'weights.hdf5')
    config.info(self.__str__())

  def __str__(self) -> str:
    return "Trainer(" + ", ".join(f'{elem}={getattr(self, elem)}' for elem in [
        'model_name', 'dataset_name', 'h_window', 'init_window', 'm_window', 'test_size',
        'train_entropy', 'epochs', 'savedir'
    ]) + ")"

  def create_model(self, model_path='') -> BaseModel:
    if self.model_name == 'pos_only':
      model = PosOnly(self.m_window, self.h_window)
    elif self.model_name == 'pos_only_3d':
      model = PosOnly3D(self.m_window, self.h_window)
    elif self.model_name == 'interpolation':
      return Interpolation(self.h_window) # does not need training
    elif self.model_name == 'no_motion':
      return NoMotion(self.h_window) # does not need training
    else:
      raise RuntimeError
    if model_path:
      model.load_weights(model_path)
    return model

  def generate_batchs(self, model: BaseModel, wins: list) -> Generator:
    while True:
      shuffle(wins, random_state=1)
      for count, _ in enumerate(wins[::config.BATCH_SIZE]):
        end = count+config.BATCH_SIZE if count+config.BATCH_SIZE <= len(wins) else len(wins)
        traces_l = [self.ds.get_traces(win['video'],win['user']) for win in wins[count:end]]
        x_i_l = [win['trace_id'] for win in wins[count:end]]
        yield model.generate_batch(traces_l,x_i_l)

  @property
  def ds(self) -> Dataset:
    if not hasattr(self, '_ds'):
      self._ds = Dataset(savedir=self.savedir)
    return self._ds

  def _train_partition(self) -> None:
    config.info('partitioning...')
    df = self.ds.df if self.dataset_name == 'all' \
       else self.ds.df[self.ds.df['ds'] == self.dataset_name]
    # split x_train, x_test (0.2)
    self.x_train, self.x_test = \
      train_test_split(df, random_state=1, test_size=self.test_size, stratify=df[self.entropy_type + '_c'])
    # split x_train, x_val (0.125 * 0.8 = 0.1)
    self.x_train, self.x_val = \
      train_test_split(self.x_train,random_state=1, test_size=0.125, stratify=self.x_train[self.entropy_type + '_c'])

    if self.train_entropy != 'all':
      pre_filter_x_train_len = len(self.x_train)
      pre_filter_epochs = self.epochs
      config.info('train_entropy != all, so filtering x_train, x_val')
      self.x_train = filter_df_by_entropy(self.x_train, self.entropy_type, self.train_entropy)
      self.x_val = filter_df_by_entropy(self.x_val, self.entropy_type, self.train_entropy)
      config.info('x_train filtred has {} trajectories: {} low, {} medium, {} high'.format(
          *count_entropy(self.x_train, self.entropy_type)))
      config.info('x_val filtred has {} trajectories: {} low, {} medium, {} high'.format(
          *count_entropy(self.x_val, self.entropy_type)))
      pos_filter_x_train_len = len(self.x_train)
      # given pre_filter_x_train_len < pos_filter_x_train_len, increase epochs
      self.epochs = self.epochs + round(
          0.1 * self.epochs * pre_filter_x_train_len / pos_filter_x_train_len)
      config.info('given x_train filtred, compensate by changing epochs from {} to {} '.format(
          pre_filter_epochs, self.epochs))
    else:
      config.info('x_train has {} trajectories: {} low, {} medium, {} high'.format(
          *count_entropy(self.x_train, self.entropy_type)))
      config.info('x_val has {} trajectories: {} low, {} medium, {} high'.format(
          *count_entropy(self.x_val, self.entropy_type)))

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

  def _evaluate_partition(self) -> None:
    config.info('partitioning...')
    self.ds
    df = self.ds.df if self.dataset_name == 'all' \
       else self.ds.df[self.ds.df['ds'] == self.dataset_name]
    # split x_train, x_test (0.2)
    self.x_train, self.x_test = \
      train_test_split(df, random_state=1, test_size=self.test_size, stratify=df[self.entropy_type + '_c'])

    config.info('x_test has {} trajectories: {} low, {} medium, {} high'.format(
        *count_entropy(self.x_test, self.entropy_type)))

    # predict by each x_test_wins
    self.x_test_wins = [{
        'video': row[1]['video'],
        'user': row[1]['user'],
        'trace_id': trace_id,
        'actS_c': row[1]['actS_c']
    } for row in self.x_test.iterrows()\
      for trace_id in range(self.init_window, row[1]['traces'].shape[0] -self.end_window)]

  def train(self) -> None:
    config.info('train()')
    assert not self.using_auto, "train_entropy should not be auto"
    assert self.model_name not in config.MODELS_NAMES_NO_TRAIN, f"{self.model_name} does not need training"
    config.info('model_dir=' + self.model_dir)

    # check model
    config.info('creating model ...')
    if exists(self.train_csv_log_f):
      lines = pd.read_csv(self.train_csv_log_f)
      lines.dropna(how="all", inplace=True)
      done_epochs = int(lines.iloc[-1]['epoch']) + 1
      if done_epochs >= self.epochs:
        config.info(f'train_csv_log_f has {done_epochs}>=epochs. stopping.')
        return
      else:
        config.info(f'train_csv_log_f has {self.epochs}<epochs. continuing from {done_epochs}.')
        model = self.create_model(self.model_path)
        initial_epoch = done_epochs
    else:
      model = self.create_model()
      initial_epoch = 0
      if not exists(self.model_dir):
        os.makedirs(self.model_dir)
    assert model

    # partition
    self._train_partition()

    # fit
    steps_per_ep_train = np.ceil(len(self.x_train_wins) / config.BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(self.x_val_wins) / config.BATCH_SIZE)
    csv_logger = CSVLogger(self.train_csv_log_f, append=True)
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model_checkpoint = ModelCheckpoint(self.model_path,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       verbose=1)
    callbacks = [csv_logger, model_checkpoint]
    generator = self.generate_batchs(model, self.x_train_wins)
    validation_data = self.generate_batchs(model, self.x_val_wins)
    model.fit(x=generator,
              verbose="1",
              steps_per_epoch=steps_per_ep_train,
              validation_data=validation_data,
              validation_steps=steps_per_ep_validate,
              validation_freq=config.BATCH_SIZE,
              epochs=self.epochs,
              initial_epoch=initial_epoch,
              callbacks=callbacks)

  def _auto_select_model(self, traces: np.array, x_i) -> BaseModel:
    if self.train_entropy == 'auto':
      window = traces
    elif self.train_entropy == 'auto_m_window':
      window = traces[x_i - self.m_window:x_i]
    elif self.train_entropy == 'auto_since_start':
      window = traces[0:x_i]
    else:
      raise RuntimeError()
    a_ent = calc_actual_entropy(window)
    actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_high)
    if actS_c == 'low':
      return self.model_low
    elif actS_c == 'medium':
      return self.model_medium
    elif actS_c == 'high':
      return self.model_high

  def evaluate(self) -> None:
    config.info('evaluate()')
    config.info('model_dir=' + self.model_dir)

    # partition
    self._evaluate_partition()

    # create model
    config.info('creating model ...')
    if self.using_auto:
      prefix = join(self.savedir, f'{self.model_name},{self.dataset_name},actS,')
      self.threshold_medium, self.threshold_high = get_class_thresholds(self.ds.df, 'actS')
      self.model_low = self.create_model(join(prefix + 'low'))
      self.model_medium = self.create_model(join(prefix + 'medium'))
      self.model_high = self.create_model(join(prefix + 'high'))
    else:
      model = self.create_model(self.model_path)

    if not self.model_fullname in self.ds.df.columns:
      empty = pd.Series([{} for _ in range(len(self.ds.df))]).astype(object)
      self.ds.df[self.model_fullname] = empty

    for ids in tqdm(self.x_test_wins, desc=f'evaluate model {self.model_fullname}'):
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']
      traces = self.ds.get_traces(video, user)
      # predict
      if self.using_auto:
        model = self._auto_select_model(traces,x_i)
      pred = model.predict_for_sample(traces, x_i)
      # save prediction
      traject_row = self.ds.df.loc[(self.ds.df['video'] == video) & (self.ds.df['user'] == user)]
      assert not traject_row.empty
      index = traject_row.index[0]
      traject_row.loc[index, self.model_fullname][x_i] = pred

    # save on df
    self.ds.dump_column(self.model_fullname)

  def compare_train(self) -> None:
    assert exists(self.savedir)
    result_csv = 'train_results.csv'
    # find result_csv files
    csv_df_l = [(dir_name, pd.read_csv(join(self.savedir, dir_name, file_name)))
                for dir_name in os.listdir(self.savedir)
                if isdir(join(self.savedir, dir_name))
                for file_name in os.listdir(join(self.savedir, dir_name))
                if file_name == result_csv]
    csv_df_l = [df.assign(model=dir_name) for (dir_name, df) in csv_df_l]
    assert csv_df_l, f'no <savedir>/<model>/{result_csv} files, run -train'

    # plot
    df_compare = pd.concat(csv_df_l)
    fig = px.line(df_compare,
                  x='epoch',
                  y='loss',
                  color='model',
                  title='compare_train_loss',
                  width=800)
    config.show_or_save(fig, self.savedir, 'compare_train')

  def list_done_evaluate(self) -> None:
    self.ds
    models_cols = sorted([
      col for col in self.ds.df.columns \
      if any(m_name in col for m_name in config.ARGS_MODEL_NAMES)\
      and not any(ds_name in col for ds_name in config.ARGS_DS_NAMES[1:])])
    if models_cols:
      for model in models_cols:
        preds = len(self.ds.df[model].apply(lambda x: len(x) != 0))
        config.info(f"{model} has {preds} predict wins calculated")
    else:
      config.error('no evaluate done')

  def compare_evaluate(self) -> None:
    # horizon timestamps to be calculated
    self.range_win = range(self.h_window)[::4]

    # create df_compare_evaluate
    columns = ['model_name', 'S_type', 'S_class']
    if not hasattr(self, 'df_compare_evaluate'):
      if exists(self.compare_eval_pickle):
        with open(self.compare_eval_pickle, 'rb') as f:
          config.info(f'loading df_compare_evaluate from {self.compare_eval_pickle}')
          self.df_compare_evaluate = pickle.load(f)
      else:
        self.df_compare_evaluate = pd.DataFrame(columns=columns + list(self.range_win), dtype=np.float32)

    self.partition()

    # models_cols
    models_cols = sorted([
      col for col in self.x_test.columns \
      if any(m_name in col for m_name in config.ARGS_MODEL_NAMES)\
      and not any(ds_name in col for ds_name in config.ARGS_DS_NAMES[1:])\
      and not any(self.df_compare_evaluate['model_name'] == col) # already calculated
      ])
    if models_cols:
      config.info(f"compare evaluate for models: {', '.join(models_cols)}")
      config.info(f"for each model, compare users: {config.ARGS_ENTROPY_NAMES[:6]}")
    else:
      config.info(f"evaluate models already calculated. skip to visualize")

    # create targets in format (model, s_type, s_class, mask)
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
          (model, self.entropy_type, 'nohigh', self.x_test[self.entropy_type + '_c'] != 'high'))
      targets.append(
          (model, self.entropy_type, 'high', self.x_test[self.entropy_type + '_c'] == 'high'))

    # function to fill self.df_compare_evaluate from each moldel column at self.df
    def _evaluate_model_wins(df_wins_cols, errors_per_timestamp) -> None:
      traject_index = df_wins_cols.name
      traject = self.x_test.loc[traject_index, 'traces']
      win_pos_l = df_wins_cols.index
      for win_pos in win_pos_l:
        if isinstance(df_wins_cols[win_pos], float):
          break  # TODO: review why some pred ends at 51
        true_win = traject[win_pos + 1:win_pos + self.h_window + 1]
        pred_win = df_wins_cols[win_pos]
        for t in self.range_win:
          errors_per_timestamp[t].append(orth_dist_cartesian(true_win[t], pred_win[t]))

    for model, s_type, s_class, mask in tqdm(targets, desc='compare evaluate'):
      # create df_win with columns as timestamps
      model_srs = self.x_test.loc[mask, model]
      if len(model_srs) == 0:
        raise RuntimeError(f"empty {model=}, {s_type}={s_class}")
      model_df_wins = pd.DataFrame.from_dict(model_srs.values.tolist())
      model_df_wins.index = model_srs.index
      # calc errors_per_timestamp from df_wins
      errors_per_timestamp = {idx: [] for idx in self.range_win}
      model_df_wins.apply(_evaluate_model_wins, axis=1, args=(errors_per_timestamp, ))
      newid = len(self.df_compare_evaluate)
      avg_error_per_timestamp = [
          np.mean(errors_per_timestamp[t]) if len(errors_per_timestamp[t]) else np.nan
          for t in self.range_win
      ]
      self.df_compare_evaluate.loc[newid, ['model_name', 'S_type', 'S_class']] = [model, s_type, s_class]
      self.df_compare_evaluate.loc[newid, self.range_win] = avg_error_per_timestamp

    self.compare_evaluate_show()

  def compare_evaluate_show(self, model_filter=[], entropy_filter=[]) -> None:
    # create vis table
    assert len(self.df_compare_evaluate), 'run -evaluate first'
    props = 'text-decoration: underline'
    df = self.df_compare_evaluate
    if model_filter:
      df = df.loc[df['model_name'].isin(model_filter)]
    if entropy_filter:
      df = df.loc[df['S_class'].isin(entropy_filter)]
    output = df.dropna()\
      .sort_values(by=list(self.range_win))\
      .style\
      .background_gradient(axis=0, cmap='coolwarm')\
      .highlight_min(subset=list(self.range_win), props=props)\
      .highlight_max(subset=list(self.range_win), props=props)
    config.show_or_save(output, self.savedir, 'compare_evaluate')

  def compare_evaluate_save(self) -> None:
    assert hasattr(self, 'df_compare_evaluate')
    with open(self.compare_eval_pickle, 'wb') as f:
      config.info(f'saving df_compare_evaluate to {self.compare_eval_pickle}')
      pickle.dump(self.df_compare_evaluate, f)


  def show_train_test_split(self) -> None:
    self.partition()
    self.x_train['partition'] = 'train'
    self.x_test['partition'] = 'test'
    self.ds.df = pd.concat([self.x_train, self.x_test])
    self.ds.show_histogram(facet='partition')