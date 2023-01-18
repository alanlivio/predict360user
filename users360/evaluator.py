import os
import sys
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm

from .trainer import *


class Evaluator():

  def __init__(self, cfg: dict) -> None:
    self.dataset_name = cfg['dataset_name']
    self.model_name = cfg['model_name']
    config.info('dataset=' +
                (self.dataset_name if self.dataset_name != 'all' else repr(config.DS_NAMES)))
    dataset_suffix = '' if cfg['dataset_name'] == 'all' else f'_{self.dataset_name}'
    self.model_ds = self.model_name + dataset_suffix
    self.model_ds_dir = join(config.DATADIR, self.model_ds)
    self.perc_test = cfg['perc_test']
    self.epochs = cfg['epochs']
    self.init_window = cfg['init_window']
    self.m_window = cfg['m_window']
    self.h_window = cfg['h_window']
    self.end_window = self.h_window
    self.test_prefix_perc = f"test_{str(self.perc_test).replace('.',',')}"
    self.test_model_entropy = cfg['test_model_entropy']
    self.evaluate_auto = cfg['test_model_entropy'].startswith('auto')
    self.train_entropy = cfg['train_entropy']
    self.test_entropy = cfg['test_entropy']
    self.oneuser = cfg['oneuser'] if 'oneuser' in cfg else ''
    self.onevideo = cfg['onevideo'] if 'onevideo' in cfg else ''
    self.dry_run = cfg['dry_run']
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if 'gpu_id' in cfg:
      os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

  def evaluate(self) -> None:
    config.info(f'-- evaluate() perc_test={self.perc_test}')

    # model_dir
    self.model_dir = self.model_ds_dir + ('' if self.test_model_entropy == 'all' else
                                          f'_{self.test_model_entropy}_entropy')
    model_column = self.model_ds + ('' if self.test_model_entropy == 'all' else
                                    f'_{self.test_model_entropy}_entropy')
    config.info(f'model_dir={self.model_dir}')
    # evaluate_prefix
    if self.oneuser and self.onevideo:
      evaluate_prefix = join(
          self.model_dir,
          f'{self.test_prefix_perc}_{self.test_entropy}_{self.oneuser}_{self.onevideo}')
    else:
      evaluate_prefix = join(self.model_dir, f'{self.test_prefix_perc}_{self.test_entropy}')
    config.info(f'evaluate_prefix={evaluate_prefix}')

    # pred_windows, videos_test
    config.info('partioning...')
    df = get_df_trajects()
    threshold_medium, threshold_hight = get_trajects_entropy_threshold(df)
    if self.oneuser and self.onevideo:
      rows = get_rows(df, self.onevideo, self.oneuser, self.dataset_name)
      videos_test = rows['ds_video'].unique()
      pred_windows = create_pred_windows(None, rows, True)
    else:
      tmp_df = df[df['ds'] == self.dataset_name] if self.dataset_name != 'all' else df
      config.info(f'x_test entropy={self.test_entropy}')
      _, x_test = get_train_test_split(tmp_df, self.test_entropy, self.perc_test)
      videos_test = x_test['ds_video'].unique()
      pred_windows = create_pred_windows(
          x_train=None,
          x_test=x_test,
          init_window=self.init_window,
          end_window=self.end_window,
          skip_train=True,
      )

    if not model_column in df.columns:
      df[model_column] = pd.Series([{} for _ in range(len(df))]).astype(object)

    # creating model
    config.info('creating model ...')
    # model_weights
    # check existing if one model
    if not self.test_model_entropy.startswith('auto'):
      model_weights = join(self.model_dir, 'weights.hdf5')
      config.info(f'model_weights={model_weights}')
      assert exists(model_weights)
    # check exists if using mutiple models
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
      sys.exit()
    if self.evaluate_auto:
      model_low = create_model()
      model_low.load_weights(model_weights_low)
      model_medium = create_model()
      model_medium.load_weights(model_weights_medium)
      model_hight = create_model()
      model_hight.load_weights(model_weights_hight)
    else:
      model = create_model()
      model.load_weights(model_weights)

    # predict by each pred_windows
    errors_per_video = {}
    errors_per_timestep = {}

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
    for video_name in videos_test:
      for t in range(self.h_window):
        if not video_name in errors_per_video:
          config.error(f'missing {video_name} in videos_test')
          continue
        avg = np.mean(errors_per_video[video_name][t])
        avg_error_per_video.append(f'video={video_name} {t} {avg}')
    result_file = f'{evaluate_prefix}_avg_error_per_video.csv'
    np.savetxt(result_file, avg_error_per_video, fmt='%s')
    config.info(f'saving {result_file}')

  def compare_results(self) -> None:
    suffix = '_avg_error_per_timestep.csv'

    # find files with suffix
    dirs = [d for d in os.listdir(config.DATADIR) if d.startswith(self.model_name)]
    csv_file_l = [(dir_name, file_name) for dir_name in dirs
                  for file_name in os.listdir(join(config.DATADIR, dir_name))
                  if (file_name.endswith(suffix) and file_name.startswith(self.test_prefix_perc))]
    csv_data_l = [
        (f'{dir_name}_{file_name.removesuffix(suffix)}', horizon, error)
        for (dir_name, file_name) in csv_file_l
        for horizon, error in enumerate(np.loadtxt(join(config.DATADIR, dir_name, file_name)))
    ]
    assert csv_data_l, f'no data/<model>/{self.test_prefix_perc}_*, run -evaluate'

    # plot image
    df_compare = pd.DataFrame(csv_data_l, columns=['name', 'horizon', 'vidoes_avg_error'])
    df_compare = df_compare.sort_values(ascending=False, by="vidoes_avg_error")
    fig = px.line(df_compare,
                  x='horizon',
                  y="vidoes_avg_error",
                  color='name',
                  color_discrete_sequence=px.colors.qualitative.G10)
    result_file = join(config.DATADIR, f'compare_{self.model_name}.png')
    config.info(f'saving {result_file}')
    fig.write_image(result_file)
