import logging
import os
import pickle
from dataclasses import dataclass
from os.path import basename, exists, isdir, join
from typing import Generator

import absl.logging
import hydra
import numpy as np
import pandas as pd
import plotly.express as px
from hydra.core.config_store import ConfigStore
from keras.callbacks import CSVLogger, ModelCheckpoint
from omegaconf import OmegaConf
from sklearn.utils import shuffle
from tqdm.auto import tqdm

from predict360user.dataset import (Dataset, get_class_name,
                                    get_class_thresholds)
from predict360user.models import (BaseModel, Interpolation, NoMotion, PosOnly,
                                   PosOnly3D)
from predict360user.utils import (calc_actual_entropy, orth_dist_cartesian,
                                  show_or_save)

ARGS_ENTROPY_NAMES = [ 'all', 'low', 'medium', 'high', 'nohigh', 'nolow', 'allminsize' ]
ARGS_MODEL_NAMES = ['pos_only', 'pos_only_3d', 'no_motion', 'interpolation', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
MODELS_NAMES_NO_TRAIN = ['no_motion', 'interpolation']
ARGS_DS_NAMES = ['all', 'david', 'fan', 'nguyen', 'xucvpr', 'xupami']
ARGS_ENTROPY_AUTO_NAMES = ['auto', 'auto_m_window', 'auto_since_start']
log = logging.getLogger(basename(__file__))

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@dataclass
class TrainerCfg():

  batch_size: int = 128
  dataset_name: str = 'all'
  epochs: int = 30
  gpu_id: int = 0
  h_window: int = 25
  init_window: int = 30
  lr: float = 0.0005
  m_window: int = 5
  model_name: str = 'pos_only'
  savedir: str = 'saved'
  train_size: float = 0.8
  test_size: float = 0.2
  train_entropy: str = 'all'

  def __post_init__(self) -> None:
    assert self.model_name in ARGS_MODEL_NAMES
    assert self.dataset_name in ARGS_DS_NAMES
    assert self.train_entropy in ARGS_ENTROPY_NAMES + ARGS_ENTROPY_AUTO_NAMES

  def __str__(self) -> str:
    return OmegaConf.to_yaml(self)

cs = ConfigStore.instance()
cs.store(name="trainer", group="trainer", node=TrainerCfg)

class Trainer():
  cfg: TrainerCfg

  def __init__(self, cfg: TrainerCfg) -> None:
    log.info("TrainerCfg:\n-------\n"+ OmegaConf.to_yaml(cfg) + "-------")
    self.cfg = cfg

    if self.cfg.gpu_id:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)
      log.info(f"set visible cpu to {self.cfg.gpu_id}")

    # properties others
    self.compare_eval_pickle = join(self.cfg.savedir, 'df_compare_evaluate.pickle')
    self.using_auto = self.cfg.train_entropy.startswith('auto')
    self.entropy_type = 'actS'
    if self.cfg.dataset_name == 'all' and self.cfg.train_entropy == 'all':
      self.model_fullname = self.cfg.model_name
    elif self.cfg.train_entropy == 'all':
      self.model_fullname = f'{self.cfg.model_name},{self.cfg.dataset_name},,'
    else:
      self.model_fullname = f'{self.cfg.model_name},{self.cfg.dataset_name},{self.entropy_type},{self.cfg.train_entropy}'
    self.model_dir = join(self.cfg.savedir, self.model_fullname)
    self.train_csv_log_f = join(self.model_dir, 'train_results.csv')
    self.model_path = join(self.model_dir, 'weights.hdf5')
    self.model: BaseModel
    if self.cfg.model_name == 'pos_only':
      self.model = PosOnly(self.cfg)
    elif self.cfg.model_name == 'pos_only_3d':
      self.model = PosOnly3D(self.cfg)
    elif self.cfg.model_name == 'interpolation':
      self.model = Interpolation(self.cfg)
    elif self.cfg.model_name == 'no_motion':
      self.model = NoMotion(self.cfg)
    else:
      raise RuntimeError

  def generate_batchs(self, model: BaseModel, wins: list) -> Generator:
    while True:
      shuffle(wins, random_state=1)
      for count, _ in enumerate(wins[::self.cfg.batch_size]):
        end = count + self.cfg.batch_size if count + self.cfg.batch_size <= len(wins) else len(wins)
        traces_l = [self.ds.get_traces(win['video'], win['user']) for win in wins[count:end]]
        x_i_l = [win['trace_id'] for win in wins[count:end]]
        yield model.generate_batch(traces_l, x_i_l)

  def drop_predict_cols(self) -> None:
    col_rm = [col for col in self.ds.df.columns for model in ARGS_MODEL_NAMES if col.startswith(model)]
    self.ds.df.drop(col_rm, axis=1, errors='ignore', inplace=True)

  def _auto_select_model(self, traces: np.array, x_i) -> BaseModel:
    if self.cfg.train_entropy == 'auto':
      window = traces
    elif self.cfg.train_entropy == 'auto_m_window':
      window = traces[x_i - self.cfg.m_window:x_i]
    elif self.cfg.train_entropy == 'auto_since_start':
      window = traces[0:x_i]
    else:
      raise RuntimeError()
    a_ent = calc_actual_entropy(window)
    actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_high)
    if actS_c == 'low':
      return self.model_low
    if actS_c == 'medium':
      return self.model_medium
    if actS_c == 'high':
      return self.model_high
    raise RuntimeError()

  def run(self) -> None:
    if not exists(self.model_dir):
      os.makedirs(self.model_dir)
    log.info('model_dir=' + self.model_dir)

    if not hasattr(self, 'ds'):
      log.info('loading dataset ...')
      # TODO: filter by dataset name
      self.ds = Dataset(savedir=self.cfg.savedir)
      self.ds.partition(entropy_filter=self.cfg.train_entropy, train_size=self.cfg.train_size, test_size=self.cfg.test_size)
      self.ds.create_wins(init_window=self.cfg.init_window, h_window=self.cfg.h_window)

    if not self.using_auto and self.cfg.model_name not in MODELS_NAMES_NO_TRAIN:
      log.info('train ...')
      if exists(self.model_path):
        self.model.load_weights(self.model_path)

      # setting initial_epoch
      initial_epoch = 0
      if exists(self.train_csv_log_f):
        lines = pd.read_csv(self.train_csv_log_f)
        lines.dropna(how="all", inplace=True)
        done_epochs = int(lines.iloc[-1]['epoch']) + 1
        assert done_epochs <= self.cfg.epochs
        initial_epoch = done_epochs
        log.info(f'train_csv_log_f has {initial_epoch} epochs ')

      if initial_epoch >= self.cfg.epochs:
        log.info(f'train_csv_log_f has {initial_epoch}>={self.cfg.epochs}. not training.')
      else:
        # fit
        steps_per_ep_train = np.ceil(len(self.ds.x_train_wins) / self.cfg.batch_size)
        steps_per_ep_validate = np.ceil(len(self.ds.x_val_wins) / self.cfg.batch_size)
        csv_logger = CSVLogger(self.train_csv_log_f, append=True)
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        model_checkpoint = ModelCheckpoint(self.model_path,
                                          save_weights_only=True,
                                          verbose=1)
        callbacks = [csv_logger, model_checkpoint]
        generator = self.generate_batchs(self.model, self.ds.x_train_wins)
        validation_data = self.generate_batchs(self.model, self.ds.x_val_wins)
        self.model.fit(x=generator,
                  verbose=1,
                  steps_per_epoch=steps_per_ep_train,
                  validation_data=validation_data,
                  validation_steps=steps_per_ep_validate,
                  validation_freq=self.cfg.batch_size,
                  epochs=self.cfg.epochs,
                  initial_epoch=initial_epoch,
                  callbacks=callbacks)

    log.info('evaluate ...')
    if self.using_auto:
      prefix = join(self.cfg.savedir, f'{self.cfg.model_name},{self.cfg.dataset_name},actS,')
      log.info('creating model auto ...')
      self.threshold_medium, self.threshold_high = get_class_thresholds(self.ds.df, 'actS')
      self.model_low = self.model.copy()
      self.model_low.load_weights(join(prefix + 'low', 'weights.hdf5'))
      self.model_medium = self.model.copy()
      self.model_medium.load_weights(join(prefix + 'medium', 'weights.hdf5'))
      self.model_high = self.model.copy()
      self.model_high.load_weights(join(prefix + 'high', 'weights.hdf5'))

    if not self.model_fullname in self.ds.df.columns:
      empty = pd.Series([{} for _ in range(len(self.ds.df))]).astype(object)
      self.ds.df[self.model_fullname] = empty

    for ids in tqdm(self.ds.x_test_wins, desc=f'evaluate model {self.model_fullname}'):
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']
      traces = self.ds.get_traces(video, user)
      # predict
      if self.using_auto:
        pred = self._auto_select_model(traces, x_i).predict_for_sample(traces, x_i)
      else:
        pred = self.model.predict_for_sample(traces, x_i)
      # save prediction
      traject_row = self.ds.df.loc[(self.ds.df['video'] == video) & (self.ds.df['user'] == user)]
      assert not traject_row.empty
      index = traject_row.index[0]
      traject_row.loc[index, self.model_fullname][x_i] = pred

    # save on df
    self.ds.dump_column(self.model_fullname)

  #
  # compare-related methods TODO: replace then by a log in a model registry
  #

  def compare_train(self) -> None:
    assert exists(self.cfg.savedir), f'the save folder {self.cfg.savedir} does not exist. do -train call'
    result_csv = 'train_results.csv'
    # find result_csv files
    csv_df_l = [(dir_name, pd.read_csv(join(self.cfg.savedir, dir_name, file_name)))
                for dir_name in os.listdir(self.cfg.savedir) if isdir(join(self.cfg.savedir, dir_name))
                for file_name in os.listdir(join(self.cfg.savedir, dir_name))
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
    show_or_save(fig, self.cfg.savedir, 'compare_train')

  def list_done_evaluate(self) -> None:
    models_cols = sorted([
      col for col in self.ds.df.columns \
      if any(m_name in col for m_name in ARGS_MODEL_NAMES)\
      and not any(ds_name in col for ds_name in ARGS_DS_NAMES[1:])])
    if models_cols:
      for model in models_cols:
        preds = len(self.ds.df[model].apply(lambda x: len(x) != 0))
        log.info(f"{model} has {preds} predict wins calculated")
    else:
      log.error('no evaluate done')

  def compare_evaluate(self) -> None:
    # horizon timestamps to be calculated
    self.range_win = range(self.cfg.h_window)[::4]

    # create df_compare_evaluate
    columns = ['model_name', 'S_type', 'S_class']
    if not hasattr(self, 'df_compare_evaluate'):
      if exists(self.compare_eval_pickle):
        with open(self.compare_eval_pickle, 'rb') as f:
          log.info(f'loading df_compare_evaluate from {self.compare_eval_pickle}')
          self.df_compare_evaluate = pickle.load(f)
      else:
        self.df_compare_evaluate = pd.DataFrame(columns=columns + list(self.range_win),
                                                dtype=np.float32)

    if not hasattr(self, 'ds') and not hasattr(self.ds, 'x_test'):
      # TODO: filter by dataset name
      self.ds = Dataset(savedir=self.cfg.savedir)
      self.ds.partition(entropy_filter=self.cfg.train_entropy, test_size=self.cfg.test_size)

    # models_cols
    models_cols = sorted([
      col for col in self.ds.x_test.columns \
      if any(m_name in col for m_name in ARGS_MODEL_NAMES)\
      and not any(ds_name in col for ds_name in ARGS_DS_NAMES[1:])\
      and not any(self.df_compare_evaluate['model_name'] == col) # already calculated
      ])
    if models_cols:
      log.info(f"compare evaluate for models: {', '.join(models_cols)}")
      log.info(f"for each model, compare users: {ARGS_ENTROPY_NAMES[:6]}")
    else:
      log.info(f"evaluate models already calculated. skip to visualize")

    # create targets in format (model, s_type, s_class, mask)
    targets = []
    for model in models_cols:
      assert len(self.ds.x_test) == len(self.ds.x_test[model].apply(lambda x: len(x) != 0))
      targets.append((model, 'all', 'all', pd.Series(True, index=self.ds.x_test.index)))
      targets.append(
          (model, self.entropy_type, 'low', self.ds.x_test[self.entropy_type + '_c'] == 'low'))
      targets.append((model, self.entropy_type, 'nolow', self.ds.x_test[self.entropy_type + '_c']
                      != 'low'))
      targets.append(
          (model, self.entropy_type, 'medium', self.ds.x_test[self.entropy_type + '_c'] == 'medium'))
      targets.append((model, self.entropy_type, 'nohigh', self.ds.x_test[self.entropy_type + '_c']
                      != 'high'))
      targets.append(
          (model, self.entropy_type, 'high', self.ds.x_test[self.entropy_type + '_c'] == 'high'))

    # function to fill self.df_compare_evaluate from each moldel column at self.df
    def _evaluate_model_wins(df_wins_cols, errors_per_timestamp) -> None:
      traject_index = df_wins_cols.name
      traject = self.ds.x_test.loc[traject_index, 'traces']
      win_pos_l = df_wins_cols.index
      for win_pos in win_pos_l:
        if isinstance(df_wins_cols[win_pos], float):
          break  # TODO: review why some pred ends at 51
        true_win = traject[win_pos + 1:win_pos + self.cfg.h_window + 1]
        pred_win = df_wins_cols[win_pos]
        for t in self.range_win:
          errors_per_timestamp[t].append(orth_dist_cartesian(true_win[t], pred_win[t]))

    for model, s_type, s_class, mask in tqdm(targets, desc='compare evaluate'):
      # create df_win with columns as timestamps
      model_srs = self.ds.x_test.loc[mask, model]
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
      self.df_compare_evaluate.loc[newid,
                                   ['model_name', 'S_type', 'S_class']] = [model, s_type, s_class]
      self.df_compare_evaluate.loc[newid, self.range_win] = avg_error_per_timestamp

    self.compare_evaluate_show()

  def compare_evaluate_show(self, model_filter=None, entropy_filter=None) -> None:
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
    show_or_save(output, self.cfg.savedir, 'compare_evaluate')

  def compare_evaluate_save(self) -> None:
    assert hasattr(self, 'df_compare_evaluate')
    with open(self.compare_eval_pickle, 'wb') as f:
      log.info(f'saving df_compare_evaluate to {self.compare_eval_pickle}')
      pickle.dump(self.df_compare_evaluate, f)


@hydra.main(version_base=None, config_path="conf", config_name="trainer")
def trainer_cli(cfg: TrainerCfg) -> None:
  exp = Trainer(cfg)
  exp.run()