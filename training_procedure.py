#!env python

import argparse
import os
import sys
from contextlib import redirect_stderr
from os.path import exists, join
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm

from users360 import (calc_actual_entropy, calc_trajects_entropy, config,
                      dump_df_trajects, get_class_by_threshold,
                      get_df_trajects, get_traces, get_train_test_split,
                      get_trajects_entropy_threshold)
from users360.head_motion_prediction.Utils import (all_metrics,
                                                   cartesian_to_eulerian,
                                                   eulerian_to_cartesian)

METRIC = all_metrics['orthodromic']
RATE = 0.2
BATCH_SIZE = 128.0


def create_model() -> Any:
  if MODEL_NAME == 'pos_only':
    from users360.head_motion_prediction.position_only_baseline import \
        create_pos_only_model
    return create_pos_only_model(M_WINDOW, H_WINDOW)
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


def generate_arrays(df_trajects: pd.DataFrame, pred_windows: dict, future_window) -> Generator:
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
      if MODEL_NAME == 'pos_only':
        encoder_pos_inputs_for_batch.append(
            get_traces(df_trajects, video, user, DATASET_NAME)[x_i - M_WINDOW:x_i])
        decoder_pos_inputs_for_batch.append(
            get_traces(df_trajects, video, user, DATASET_NAME)[x_i:x_i + 1])
        decoder_outputs_for_batch.append(
            get_traces(df_trajects, video, user, DATASET_NAME)[x_i + 1:x_i + future_window + 1])
      else:
        raise NotImplementedError
      count += 1
      if count == BATCH_SIZE:
        count = 0
        if MODEL_NAME == 'pos_only':
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
      if MODEL_NAME == 'pos_only':
        yield ([
            transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
            transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)
        ], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
      else:
        raise NotImplementedError


def train() -> None:
  config.loginf(f'-- train() PERC_TEST={PERC_TEST}, EPOCHS={EPOCHS}')

  # model_folder
  model_folder = MODEL_DS_PREFIX + ('' if TRAIN_ENTROPY == 'all' else
                                    f'_{TRAIN_ENTROPY}_entropy')
  config.loginf(f'model_folder={model_folder}')
  if not exists(model_folder):
    os.makedirs(model_folder)

  # model_weights
  model_weights = join(model_folder, 'weights.hdf5')
  config.loginf(f'model_weights={model_weights}')

  # x_train, x_test, pred_windows
  config.loginf('partioning...')
  df_trajects = get_df_trajects()
  if DATASET_NAME != 'all':
    df_trajects = df_trajects[df_trajects['ds'] == DATASET_NAME]
  config.loginf(f'x_train, x_test entropy is {TRAIN_ENTROPY}')
  x_train, x_test = get_train_test_split(df_trajects, TRAIN_ENTROPY, PERC_TEST)
  pred_windows = create_pred_windows(x_train, x_test)

  with redirect_stderr(open(os.devnull, 'w')):  # pylint: disable=unspecified-encoding
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow.keras as keras

  steps_per_ep_train = np.ceil(len(pred_windows['train']) / BATCH_SIZE)
  steps_per_ep_validate = np.ceil(len(pred_windows['test']) / BATCH_SIZE)

  # creating model
  config.loginf('creating model ...')
  # sys.exit()
  model = create_model()
  assert model

  # train
  csv_logger_f = join(model_folder, 'train_results.csv')
  csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
  tb_callback = keras.callbacks.TensorBoard(log_dir=f'{model_folder}/logs')
  model_checkpoint = keras.callbacks.ModelCheckpoint(model_weights,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)
  if MODEL_NAME == 'pos_only':
    model.fit_generator(generator=generate_arrays(df_trajects, pred_windows['train'], future_window=H_WINDOW),
                        verbose=1,
                        steps_per_epoch=steps_per_ep_train,
                        epochs=EPOCHS,
                        callbacks=[csv_logger, model_checkpoint, tb_callback],
                        validation_data=generate_arrays(df_trajects, pred_windows['test'],
                                                        future_window=H_WINDOW),
                        validation_steps=steps_per_ep_validate)
  else:
    raise NotImplementedError


def evaluate() -> None:
  config.loginf(f'-- evaluate() PERC_TEST={PERC_TEST}')

  # model_folder
  model_folder = MODEL_DS_PREFIX + ('' if TEST_MODEL_ENTROPY == 'all' else
                                        f'_{TEST_MODEL_ENTROPY}_entropy')
  config.loginf(f'model_folder={model_folder}')

  # evaluate_prefix
  evaluate_prefix = join(model_folder, f'{TEST_PREFIX_PERC}_{args.test_entropy}')
  config.loginf(f'evaluate_prefix={evaluate_prefix}')

  # model_weights
  # check existing if one model
  if not TEST_MODEL_ENTROPY.startswith('auto'):
    model_weights = join(model_folder, 'weights.hdf5')
    config.loginf(f'model_weights={model_weights}')
    assert exists(model_weights)
  # check exists if using mutiple models
  if TEST_MODEL_ENTROPY.startswith('auto'):
    model_weights_low = join(MODEL_DS_PREFIX + "_low_entropy", 'weights.hdf5')
    model_weights_medium = join(MODEL_DS_PREFIX + "_medium_entropy", 'weights.hdf5')
    model_weights_hight = join(MODEL_DS_PREFIX + "_hight_entropy", 'weights.hdf5')
    config.loginf('model_weights_low=' + model_weights_low)
    config.loginf('model_weights_medium=' + model_weights_medium)
    config.loginf('model_weights_hight=' + model_weights_hight)
    assert exists(model_weights_low)
    assert exists(model_weights_medium)
    assert exists(model_weights_hight)

  # x_test, pred_windows, videos_test
  config.loginf('partioning...')
  df_trajects = get_df_trajects()
  if DATASET_NAME != 'all':
    df_trajects = df_trajects[df_trajects['ds'] == DATASET_NAME]
  config.loginf(f'x_test entropy={args.test_entropy}')
  _, x_test = get_train_test_split(df_trajects, TEST_ENTROPY, PERC_TEST)
  pred_windows = create_pred_windows(None, x_test, True)
  videos_test = x_test['ds_video'].unique()

  # creating model
  config.loginf('creating model ...')
  # sys.exit()
  if EVALUATE_AUTO:
    model_low = create_model()
    model_low.load_weights(model_weights_low)
    model_medium = create_model()
    model_medium.load_weights(model_weights_medium)
    model_hight = create_model()
    model_hight.load_weights(model_weights_hight)
    threshold_medium, threshold_hight = get_trajects_entropy_threshold(df_trajects)
  else:
    model = create_model()
    model.load_weights(model_weights)

  # MODEL.predict
  errors_per_video = {}
  errors_per_timestep = {}

  for ids in tqdm(pred_windows['test'], desc='position predictions'):
    user = ids['user']
    video = ids['video']
    x_i = ids['trace_id']

    if MODEL_NAME == 'pos_only':
      encoder_pos_inputs_for_sample = np.array(
          [get_traces(df_trajects, video, user, DATASET_NAME)[x_i - M_WINDOW:x_i]])
      decoder_pos_inputs_for_sample = np.array(
          [get_traces(df_trajects, video, user, DATASET_NAME)[x_i:x_i + 1]])
    else:
      raise NotImplementedError

    groundtruth = get_traces(df_trajects, video, user, DATASET_NAME)[x_i + 1:x_i + H_WINDOW + 1]

    if MODEL_NAME == 'pos_only':
      current_model = model
      if EVALUATE_AUTO:
        if TEST_MODEL_ENTROPY == 'auto':
          traject_entropy_class = ids['traject_entropy_class']
        if TEST_MODEL_ENTROPY == 'auto_m_window':
          window = get_traces(df_trajects, video, user, DATASET_NAME)[x_i - M_WINDOW:x_i]
          a_ent = calc_actual_entropy(window)
          traject_entropy_class = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        elif TEST_MODEL_ENTROPY == 'auto_since_start':
          window = get_traces(df_trajects, video, user, DATASET_NAME)[0:x_i]
          a_ent = calc_actual_entropy(window)
          traject_entropy_class = get_class_by_threshold(a_ent, threshold_medium, threshold_hight)
        else:
          raise RuntimeError()
        if traject_entropy_class == 'low':
          current_model = model_low
        elif traject_entropy_class == 'medium':
          current_model = model_medium
        elif traject_entropy_class == 'hight':
          current_model = model_hight
        else:
          raise NotImplementedError
      model_pred = current_model.predict([
          transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
          transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)
      ])[0]
      model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
    else:
      raise NotImplementedError

    if not video in errors_per_video:
      errors_per_video[video] = {}
    for t in range(len(groundtruth)):
      if t not in errors_per_video[video]:
        errors_per_video[video][t] = []
      errors_per_video[video][t].append(METRIC(groundtruth[t], model_prediction[t]))
      if t not in errors_per_timestep:
        errors_per_timestep[t] = []
      errors_per_timestep[t].append(METRIC(groundtruth[t], model_prediction[t]))

  # avg_error_per_timestep
  avg_error_per_timestep = []
  for t in range(H_WINDOW):
    avg = np.mean(errors_per_timestep[t])
    avg_error_per_timestep.append(avg)
  # avg_error_per_timestep.csv
  result_file = f'{evaluate_prefix}_avg_error_per_timestep'
  config.loginf(f'saving {result_file}.csv')
  np.savetxt(f'{result_file}.csv', avg_error_per_timestep)

  # avg_error_per_timestep.png
  plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep)
  met = 'orthodromic'
  plt.title(f'Average {met} in {DATASET_NAME} dataset using {MODEL_NAME} model')
  plt.ylabel(met)
  plt.xlim(2.5)
  plt.xlabel('Prediction step s (sec.)')
  config.loginf(f'saving {result_file}.png')
  plt.savefig(result_file, bbox_inches='tight')

  # avg_error_per_video
  avg_error_per_video = []
  for video_name in videos_test:
    for t in range(H_WINDOW):
      if not video_name in errors_per_video:
        config.logerr(f'missing {video_name} in videos_test')
        continue
      avg = np.mean(errors_per_video[video_name][t])
      avg_error_per_video.append(f'video={video_name} {t} {avg}')
  result_file = f'{evaluate_prefix}_avg_error_per_video.csv'
  np.savetxt(result_file, avg_error_per_video, fmt='%s')
  config.loginf(f'saving {result_file}')

def create_pred_windows(x_train: pd.DataFrame, x_test: pd.DataFrame, skip_train = False) -> dict:
  pred_windows = {}
  if not skip_train:
    fmt = 'x_train has {} trajectories: {} low, {} medium, {} hight'
    t_len = len(x_train)
    l_len = len(x_train[x_train['traject_entropy_class'] == 'low'])
    m_len = len(x_train[x_train['traject_entropy_class'] == 'medium'])
    h_len = len(x_train[x_train['traject_entropy_class'] == 'hight'])
    config.loginf(fmt.format(t_len, l_len, m_len, h_len))
    pred_windows['train'] = [{
        'video': row[1]['ds_video'],
        'user': row[1]['ds_user'],
        'trace_id': trace_id
    } for row in x_train.iterrows() \
      for trace_id in range(
        INIT_WINDOW, row[1]['traject'].shape[0] -END_WINDOW)]
    p_len = len(pred_windows['train'])
    config.loginf("pred_windows['train'] has {} positions".format(p_len))
  fmt = 'x_test has {} trajectories: {} low, {} medium, {} hight'
  t_len = len(x_test)
  l_len = len(x_test[x_test['traject_entropy_class'] == 'low'])
  m_len = len(x_test[x_test['traject_entropy_class'] == 'medium'])
  h_len = len(x_test[x_test['traject_entropy_class'] == 'hight'])
  config.loginf(fmt.format(t_len, l_len, m_len, h_len))
  pred_windows['test'] = [{
      'video': row[1]['ds_video'],
      'user': row[1]['ds_user'],
      'trace_id': trace_id,
      'traject_entropy_class': row[1]['traject_entropy_class']
  } for row in x_test.iterrows() \
    for trace_id in range(
      INIT_WINDOW, row[1]['traject'].shape[0] -END_WINDOW)]
  p_len = len(pred_windows['test'])
  config.loginf("pred_windows['test'] has {} positions".format(p_len))
  return pred_windows

def compare_results() -> None:
  suffix = '_avg_error_per_timestep.csv'

  # find files with suffix
  dirs = [d for d in os.listdir(config.DATADIR) if d.startswith(MODEL_NAME)]
  csv_file_l = [(dir_name, file_name) for dir_name in dirs
                for file_name in os.listdir(join(config.DATADIR, dir_name))
                if (file_name.endswith(suffix) and file_name.startswith(TEST_PREFIX_PERC))]
  csv_data_l = [
      (f'{dir_name}_{file_name.removesuffix(suffix)}', horizon, error)
      for (dir_name, file_name) in csv_file_l
      for horizon, error in enumerate(np.loadtxt(join(config.DATADIR, dir_name, file_name)))
  ]
  assert csv_data_l, f'no data/<model>/{TEST_PREFIX_PERC}_*, run -evaluate'

  # plot image
  df_compare = pd.DataFrame(csv_data_l, columns=['name', 'horizon', 'vidoes_avg_error'])
  df_compare = df_compare.sort_values(ascending=False, by="vidoes_avg_error")
  fig = px.line(df_compare,
                x='horizon',
                y="vidoes_avg_error",
                color='name',
                color_discrete_sequence=px.colors.qualitative.G10)
  result_file = join(config.DATADIR, f'compare_{MODEL_NAME}.png')
  config.loginf(f'saving {result_file}')
  fig.write_image(result_file)


if __name__ == '__main__':
  # argparse
  psr = argparse.ArgumentParser()
  psr.description = 'train or evaluate users360 models and datasets'
  model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
  entropy_l = [
      'all',
      'low',
      'medium',
      'hight',
      'low_users',
      'medium_users',
      'hight_users',
  ]
  dataset_names = ['all', *config.DS_NAMES]

  # main actions params
  grp = psr.add_mutually_exclusive_group()
  grp.add_argument('-calculate_entropy',
                   action='store_true',
                   help='load dataset, calculate entropy and save it ')
  grp.add_argument('-compare_results', action='store_true', help='compare -evaluate results ')
  grp.add_argument('-train', action='store_true', help='train model')
  grp.add_argument('-evaluate', action='store_true', help='evaluate model')

  # train only params
  psr.add_argument('-epochs',
                   nargs='?',
                   type=int,
                   default=100,
                   help='epochs numbers (default is 500)')

  psr.add_argument('-train_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=entropy_l,
                   help='entropy to filter data model train  (default all)')

  # evaluate only params
  test_model_l = entropy_l + ['auto', 'auto_m_window', 'auto_since_start']
  psr.add_argument('-test_model_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=test_model_l,
                   help='''entropy of the model to be used.
                          auto selects from traject entropy.
                          auto_window selects from last window''')
  psr.add_argument('-test_entropy',
                   nargs='?',
                   type=str,
                   default='all',
                   choices=entropy_l,
                   help='entropy class to filter -evaluate data (default all)')

  # train/evaluate params
  psr.add_argument('-gpu_id', nargs='?', type=int, default=0, help='Used cuda gpu (default: 0)')
  psr.add_argument('-model_name',
                   nargs='?',
                   choices=model_names,
                   default=model_names[0],
                   help='reference model to used (default: pos_only)')
  psr.add_argument('-dataset_name',
                   nargs='?',
                   choices=dataset_names,
                   default=dataset_names[0],
                   help='dataset used to train this network  (default: all)')
  psr.add_argument('-init_window',
                   nargs='?',
                   type=int,
                   default=30,
                   help='initial buffer to avoid stationary part (default: 30)')
  psr.add_argument('-m_window',
                   nargs='?',
                   type=int,
                   default=5,
                   help='buffer window in timesteps (default: 5)')
  psr.add_argument('-h_window',
                   nargs='?',
                   type=int,
                   default=25,
                   help='''forecast window in timesteps (5 timesteps = 1 second)
                           used to predict (default: 25)''')
  psr.add_argument('-perc_test',
                   nargs='?',
                   type=float,
                   default=0.2,
                   help='test percetage (default: 0.2)')
  args = psr.parse_args()

  # global vars
  DATASET_NAME = args.dataset_name
  MODEL_NAME = args.model_name
  config.loginf('dataset=' + (DATASET_NAME if DATASET_NAME != 'all' else repr(config.DS_NAMES)))
  dataset_suffix = '' if args.dataset_name == 'all' else f'_{DATASET_NAME}'
  MODEL_DS_PREFIX = join(config.DATADIR, f'{MODEL_NAME}{dataset_suffix}')
  PERC_TEST = args.perc_test
  EPOCHS = args.epochs
  INIT_WINDOW = args.init_window
  M_WINDOW = args.m_window
  H_WINDOW = args.h_window
  END_WINDOW = H_WINDOW
  TEST_PREFIX_PERC = f"test_{str(PERC_TEST).replace('.',',')}"
  TEST_MODEL_ENTROPY = args.test_model_entropy
  EVALUATE_AUTO = args.test_model_entropy.startswith('auto')
  TRAIN_ENTROPY = args.train_entropy
  TEST_ENTROPY = args.test_entropy
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  if args.gpu_id:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

  # -calculate_entropy
  if args.calculate_entropy:
    df_tmp = get_df_trajects()
    calc_trajects_entropy(df_tmp)
    dump_df_trajects(df_tmp)
  # -compare_results
  elif args.compare_results:
    compare_results()
  # -train
  elif args.train:
    train()
  # -evaluate
  elif args.evaluate:
    evaluate()
  sys.exit()