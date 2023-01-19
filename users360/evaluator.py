import os
import sys
from dataclasses import dataclass
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm

from .trainer import *


@dataclass
class Evaluator():

  dataset_name: str = 'all'
  test_model_entropy :str = 'all'
  test_entropy :str  = 'all'
  h_window: int = 25
  init_window: int = 30
  m_window: int = 5
  model_name: str = 'pos_only'
  perc_test: float = 0.2
  dry_run: bool = False
  oneuser :str = ''
  onevideo :str  = ''
  model_column: str = ''
  evaluate_prefix: str = '' # path

  def __post_init__(self) -> None:
    dataset_suffix = '' if self.dataset_name == 'all' else f'_{self.dataset_name}'
    basedir = join(config.DATADIR, self.model_name + dataset_suffix)
    self.model_dir = basedir + ('' if self.test_model_entropy == 'all' else
                                f'_{self.test_model_entropy}_entropy')
    self.end_window = self.h_window
    self.evaluate_auto = self.test_model_entropy.startswith('auto')
    self.test_prefix_perc = f"test_{str(self.perc_test).replace('.',',')}"
    self.model_column = self.model_name + dataset_suffix + ('' if self.test_model_entropy == 'all' else f'_{self.test_model_entropy}_entropy')
    if self.oneuser and self.onevideo:
      self.evaluate_prefix = join(self.model_dir,
      f'{self.test_prefix_perc}_{self.test_entropy}_{self.oneuser}_{self.onevideo}')
    else:
      self.evaluate_prefix = join(self.model_dir, f'{self.test_prefix_perc}_{self.test_entropy}')

  def _partition(self) -> None:
    config.info('partioning...')
    df = get_df_trajects()
    if self.dataset_name != 'all':
      self.df_trajects = self.df_trajects[self.df_trajects['ds'] == self.dataset_name]
    if self.oneuser and self.onevideo:
      self.x_test = get_rows(df, self.onevideo, self.oneuser, self.dataset_name)
    else:
      tmp_df = df[df['ds'] == self.dataset_name] if self.dataset_name != 'all' else df
      _, self.x_test = get_train_test_split(tmp_df, self.test_entropy, self.perc_test)
    self.videos_test = self.x_test['ds_video'].unique()
    self.x_test_wins = [{
        'video': row[1]['ds_video'],
        'user': row[1]['ds_user'],
        'trace_id': trace_id,
        'traject_entropy_class': row[1]['traject_entropy_class']
    } for row in self.x_test.iterrows()\
      for trace_id in range( self.init_window, row[1]['traject'].shape[0] -self.end_window)]

  def evaluate(self) -> None:
    config.info('evaluate: ' + self.repr())
    self._partition()

    if not self.model_column in df.columns:
      df[self.model_column] = pd.Series([{} for _ in range(len(df))]).astype(object)

    # creating model
    config.info('creating model ...')
    # model_weights
    if not self.test_model_entropy.startswith('auto'):
      model_weights = join(self.model_dir, 'weights.hdf5')
      config.info(f'model_weights={model_weights}')
      assert exists(model_weights)
    if self.test_model_entropy.startswith('auto'):
      model_weights_low = join(self.model_ds_dir + "_low_entropy", 'weights.hdf5')
      model_weights_medium = join(self.model_ds_dir + "_medium_entropy", 'weights.hdf5')
      model_weights_hight = join(self.model_ds_dir + "_hight_entropy", 'weights.hdf5')
      config.info('model_weights_low=' + model_weights_low)
      config.info('model_weights_medium=' + model_weights_medium)
      config.info('model_weights_hight=' + model_weights_hight)
      assert exists(model_weights_low)
      assert exists(model_weights_medium)
      assert exists(model_weights_hight)

    if self.dry_run:
      return

    if self.evaluate_auto:
      model_low =create_model(self.model_name, self.m_window, self.h_window)
      model_low.load_weights(model_weights_low)
      model_medium =create_model(self.model_name, self.m_window, self.h_window)
      model_medium.load_weights(model_weights_medium)
      model_hight =create_model(self.model_name, self.m_window, self.h_window)
      model_hight.load_weights(model_weights_hight)
    else:
     model = create_model(self.model_name, self.m_window, self.h_window)
    model.load_weights(model_weights)

    # predict by each pred_windows
    errors_per_video = {}
    errors_per_timestep = {}
    threshold_medium, threshold_hight = get_trajects_entropy_threshold(df)
    for ids in tqdm(pred_windows['test'], desc='position predictions'):
      user = ids['user']
      video = ids['video']
      x_i = ids['trace_id']

      if self.model_name == 'pos_only':
        encoder_pos_inputs_for_sample = np.array(
            [get_traces(df, video, user, self.dataset_name)[x_i - self.m_window:x_i]])
        decoder_pos_inputs_for_sample = np.array(
            [get_traces(df, video, user, self.dataset_name)[x_i:x_i + 1]])
      else:
        raise NotImplementedError

      current_model = model
      if self.evaluate_auto:
        # traject_entropy_class
        if self.test_model_entropy == 'auto':
          traject_entropy_class = ids['traject_entropy_class']
        if self.test_model_entropy == 'auto_m_window':
          window = get_traces(df, video, user, self.dataset_name)[x_i - self.m_window:x_i]
          a_ent = calc_actual_entropy(window)
          traject_entropy_class = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        elif self.test_model_entropy == 'auto_since_start':
          window = get_traces(df, video, user, self.dataset_name)[0:x_i]
          a_ent = calc_actual_entropy(window)
          traject_entropy_class = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        else:
          raise RuntimeError()
        # current_model
        if traject_entropy_class == 'low':
          current_model = model_low
        elif traject_entropy_class == 'medium':
          current_model = model_medium
        elif traject_entropy_class == 'hight':
          current_model = model_hight
        else:
          raise NotImplementedError

      # predict
      if self.model_name == 'pos_only':
        model_pred = current_model.predict([
            transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
            transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)
        ])[0]
        model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
      else:
        raise NotImplementedError

      # save prediction
      traject_row = df.loc[(df['ds_video'] == video) & (df['ds_user'] == user)]
      assert not traject_row.empty
      index = traject_row.index[0]
      traject_row[model_column][index][x_i] = model_prediction

      # save error
      groundtruth = get_traces(df, video, user, self.dataset_name)[x_i + 1:x_i + self.h_window + 1]
      if not video in errors_per_video:
        errors_per_video[video] = {}
      for t in range(len(groundtruth)):
        if t not in errors_per_video[video]:
          errors_per_video[video][t] = []
        errors_per_video[video][t].append(METRIC(groundtruth[t], model_prediction[t]))
        if t not in errors_per_timestep:
          errors_per_timestep[t] = []
        errors_per_timestep[t].append(METRIC(groundtruth[t], model_prediction[t]))

    # save on df
    dump_df_trajects(df)

    # avg_error_per_timestep
    avg_error_per_timestep = []
    for t in range(self.h_window):
      avg = np.mean(errors_per_timestep[t])
      avg_error_per_timestep.append(avg)

    # avg_error_per_timestep.csv
    result_file = f'{evaluate_prefix}_avg_error_per_timestep'
    config.info(f'saving {result_file}.csv')
    np.savetxt(f'{result_file}.csv', avg_error_per_timestep)

    # avg_error_per_timestep.png
    plt.plot(np.arange(self.h_window) + 1 * RATE, avg_error_per_timestep)
    met = 'orthodromic'
    plt.title(f'Average {met} in {self.dataset_name} dataset using {self.model_name} model')
    plt.ylabel(met)
    plt.xlim(2.5)
    plt.xlabel('Prediction step s (sec.)')
    config.info(f'saving {result_file}.png')
    plt.savefig(result_file, bbox_inches='tight')

    # avg_error_per_video
    avg_error_per_video = []
    for video_name in self.videos_test:
      for t in range(self.h_window):
        if not video_name in errors_per_video:
          config.error(f'missing {video_name} in videos_test')
          continue
        avg = np.mean(errors_per_video[video_name][t])
        avg_error_per_video.append(f'video={video_name} {t} {avg}')
    result_file = f'{evaluate_prefix}_avg_error_per_video.csv'
    np.savetxt(result_file, avg_error_per_video, fmt='%s')
    config.info(f'saving {result_file}')

def compare_results(model_name, test_prefix_perc) -> None:
  suffix = '_avg_error_per_timestep.csv'

  # find files with suffix
  dirs = [d for d in os.listdir(config.DATADIR) if d.startswith(model_name)]
  csv_file_l = [(dir_name, file_name) for dir_name in dirs
                for file_name in os.listdir(join(config.DATADIR, dir_name))
                if (file_name.endswith(suffix) and file_name.startswith(test_prefix_perc))]
  csv_data_l = [
      (f'{dir_name}_{file_name.removesuffix(suffix)}', horizon, error)
      for (dir_name, file_name) in csv_file_l
      for horizon, error in enumerate(np.loadtxt(join(config.DATADIR, dir_name, file_name)))
  ]
  assert csv_data_l, f'no data/<model>/{test_prefix_perc}_*, run -evaluate'

  # plot image
  df_compare = pd.DataFrame(csv_data_l, columns=['name', 'horizon', 'vidoes_avg_error'])
  df_compare = df_compare.sort_values(ascending=False, by="vidoes_avg_error")
  fig = px.line(df_compare,
                x='horizon',
                y="vidoes_avg_error",
                color='name',
                color_discrete_sequence=px.colors.qualitative.G10)
  result_file = join(config.DATADIR, f'compare_{model_name}.png')
  config.info(f'saving {result_file}')
  fig.write_image(result_file)
