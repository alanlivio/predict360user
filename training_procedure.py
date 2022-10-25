#!env python
import argparse
import logging
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from users360 import *
from users360.head_motion_prediction.position_only_baseline import \
    create_pos_only_model

# consts
METRIC = all_metrics['orthodromic']
EPOCHS = 1
# EPOCHS = 500
NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
RATE = 0.2
# PERC_VIDEOS_TRAIN = 0.8
# PERC_USERS_TRAIN = 0.5
PERC_VIDEOS_TRAIN = 0.99
PERC_USERS_TRAIN = 0.99
PERC_TEST = 0.01
BATCH_SIZE = 128.0

# argparse vars
ARGS = None
DATASET_NAME: str
MAKE_DATASET = False
TRAIN_MODEL = False
EVALUATE_MODEL = False
ENTROPY_CLASS: str
MODEL_NAME: str
M_WINDOW: int
H_WINDOW: int
INIT_WINDOW: int
END_WINDOW: int

# other vars
MODEL: keras.models.Model
USERS: list
VIDEOS: list
VIDEOS_TRAIN: list
VIDEOS_TEST: list
USERS_TRAIN: list
USERS_TEST: list
PARTITION: dict
RESULTS_FOLDER: str
MODELS_FOLDER: str
DATASET_SAMPLED_FOLDER: str
EXP_NAME: str


def create_model() -> keras.models.Model:
    if MODEL_NAME == "pos_only":
        return create_pos_only_model(M_WINDOW, H_WINDOW)
    else:
        raise NotImplemented()


def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch) -> np.array:
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2])
                         for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
    return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions) -> np.array:
    positions = positions * np.array([2 * np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)


def generate_arrays(list_IDs, future_window) -> Generator:
    while True:
        encoder_pos_inputs_for_batch = []
        encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
        count = 0
        np.random.shuffle(list_IDs)
        for IDs in list_IDs:
            user = IDs['user']
            video = IDs['video']
            x_i = IDs['time-stamp']
            # Load the data
            if MODEL_NAME == 'pos_only':
                encoder_pos_inputs_for_batch.append(get_traces(video, user)[x_i - M_WINDOW:x_i])
                decoder_pos_inputs_for_batch.append(get_traces(video, user)[x_i:x_i + 1])
                decoder_outputs_for_batch.append(get_traces(video, user)[x_i + 1:x_i + future_window + 1])
            else:
                raise NotImplementedError()
            count += 1
            if count == BATCH_SIZE:
                count = 0
                if MODEL_NAME == 'pos_only':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
                else:
                    raise NotImplementedError()
                encoder_pos_inputs_for_batch = []
                encoder_sal_inputs_for_batch = []
                decoder_pos_inputs_for_batch = []
                decoder_sal_inputs_for_batch = []
                decoder_outputs_for_batch = []
        if count != 0:
            if MODEL_NAME == 'pos_only':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
            else:
                raise NotImplementedError()


def train() -> None:
    steps_per_ep_train = np.ceil(len(PARTITION['train']) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(PARTITION['test']) / BATCH_SIZE)

    # train
    csv_logger_f = os.path.join(RESULTS_FOLDER, 'results.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
    weights_f = os.path.join(MODELS_FOLDER, 'weights.hdf5')
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        weights_f, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    if MODEL_NAME == 'pos_only':
        MODEL.fit_generator(
            generator=generate_arrays(PARTITION['train'], future_window=H_WINDOW),
            verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS,
            callbacks=[csv_logger, model_checkpoint],
            validation_data=generate_arrays(PARTITION['test'], future_window=H_WINDOW), validation_steps=steps_per_ep_validate
        )
    else:
        raise NotImplementedError()


def evaluate() -> None:

    if MODEL_NAME == "pos_only":
        MODEL.load_weights(os.path.join(MODELS_FOLDER, 'weights.hdf5'))
    else:
        raise NotImplementedError()

    traces_count = 0
    errors_per_video = {}
    errors_per_timestep = {}

    for ID in PARTITION['test']:
        traces_count += 1
        logging.info(f"Progress: {traces_count}/{len(PARTITION['test'])}")

        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        if video not in errors_per_video.keys():
            errors_per_video[video] = {}

        # Load the data
        if MODEL_NAME == 'pos_only':
            encoder_pos_inputs_for_sample = np.array([get_traces(video, user)[x_i - M_WINDOW:x_i]])
            decoder_pos_inputs_for_sample = np.array([get_traces(video, user)[x_i:x_i + 1]])
        else:
            raise NotImplementedError()

        groundtruth = get_traces(video, user)[x_i + 1:x_i + H_WINDOW + 1]

        if MODEL_NAME == 'pos_only':
            model_pred = MODEL.predict([transform_batches_cartesian_to_normalized_eulerian(
                encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
        else:
            raise NotImplementedError()

        for t in range(len(groundtruth)):
            if t not in errors_per_video[video].keys():
                errors_per_video[video][t] = []
            errors_per_video[video][t].append(METRIC(groundtruth[t], model_prediction[t]))
            if t not in errors_per_timestep.keys():
                errors_per_timestep[t] = []
            errors_per_timestep[t].append(METRIC(groundtruth[t], model_prediction[t]))

    for video_name in VIDEOS_TEST:
        for t in range(H_WINDOW):
            logging.info(f"video={video_name} {t} {np.mean(errors_per_video[video_name][t])}")

    avg_error_per_timestep = []
    for t in range(H_WINDOW):
        logging.info(f"Average {t} {np.mean(errors_per_timestep[t])}")
        avg_error_per_timestep.append(np.mean(errors_per_timestep[t]))
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep, label=MODEL_NAME)
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.description = 'train or evaluate users360 models and datasets'
    model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
    dataset_names = ['David_MMSys_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'Xu_CVPR_18', 'Xu_PAMI_18']
    parser.add_argument('-train', action='store_true',
                        help='Flag that tells run the train procedure')
    parser.add_argument('-evaluate', action='store_true',
                        help='Flag that tells run the evaluate procedure')
    parser.add_argument('-gpu_id', nargs='?', type=int, default=0,
                        help='Used cuda gpu')
    parser.add_argument('-model_name', nargs='?',
                        choices=model_names, type=str, default=model_names[0],
                        help='The name of the model used to reference the network structure used')
    parser.add_argument('-dataset_name', nargs='?',
                        choices=dataset_names, type=str, default=dataset_names[0],
                        help='The name of the dataset used to train this network')
    parser.add_argument('-entropy_class', nargs='?',
                        help='Name entropy_class to filter dataset')
    parser.add_argument('-init_window', nargs='?', type=int, default=30,
                        help='Initial buffer to avoid stationary part')
    parser.add_argument('-m_window', nargs='?', type=int, default=5,
                        help='Buffer window in timesteps',)
    parser.add_argument('-h_window', nargs='?', type=int, default=25,
                        help='Forecast window in timesteps used to predict (5 timesteps = 1 second)')

    # vars from argparse
    ARGS = parser.parse_args()
    TRAIN_MODEL = ARGS.train
    EVALUATE_MODEL = ARGS.evaluate
    MODEL_NAME = ARGS.model_name
    DATASET_NAME = ARGS.dataset_name
    ENTROPY_CLASS = ARGS.entropy_class
    INIT_WINDOW = ARGS.init_window
    M_WINDOW = ARGS.m_window
    H_WINDOW = ARGS.h_window
    END_WINDOW = H_WINDOW

    # DATASET_SAMPLED_FOLDER
    DATASET_DIR_HMP = os.path.join('users360', 'head_motion_prediction', DATASET_NAME)
    DATASET_SAMPLED_FOLDER = os.path.join(DATASET_DIR_HMP, 'sampled_dataset')

    # RESULTS_FOLDER, MODELS_FOLDER folders
    DATASET_DIR = os.path.join(DATADIR, DATASET_NAME)
    EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + \
        str(H_WINDOW) + '_end_' + str(END_WINDOW) + ('_' + ENTROPY_CLASS + "_entropy" if ENTROPY_CLASS else '')
    if MODEL_NAME == 'pos_only':
        RESULTS_FOLDER = os.path.join(DATASET_DIR, 'pos_only', 'Results_EncDec_eulerian' + EXP_NAME)
        MODELS_FOLDER = os.path.join(DATASET_DIR, 'pos_only', 'Models_EncDec_eulerian' + EXP_NAME)
    else:
        raise NotImplementedError()
    logging.info(f"MODELS_FOLDER is {MODELS_FOLDER}")
    logging.info(f"RESULTS_FOLDER is {RESULTS_FOLDER}")
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    # prepare partitions for train/evaluate
    if (ARGS.train or ARGS.evaluate):
        logging.info("prepare partitions")
        df = get_df_trajects()
        if ENTROPY_CLASS:
            if 'entropy_class' not in df:
                raise Exception("not df.entropy_class column")
            df = df[df['entropy_class'] == ENTROPY_CLASS]
        X, Y = train_test_split(df, test_size=PERC_TEST, random_state=1)
        PARTITION = {}
        PARTITION['train'] = [{'video': row[1]['ds_video'], 'user': row[1]
                               ['ds_user'], 'time-stamp': tstap}
                              for row in X.iterrows()
                              for tstap in range(INIT_WINDOW, row[1]['traces'].shape[0] - END_WINDOW)]
        PARTITION['test'] = [{'video': row[1]['ds_video'], 'user': row[1]
                              ['ds_user'], 'time-stamp': tstap}
                             for row in Y.iterrows()
                             for tstap in range(INIT_WINDOW, row[1]['traces'].shape[0] - END_WINDOW)]
        VIDEOS_TEST = Y['ds_video'].unique()

        # create model
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if ARGS.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
        logging.info("create_model")
        MODEL = create_model()
    if ARGS.train:
        logging.info("train")
        train()
    if ARGS.evaluate:
        logging.info("evaluate")
        evaluate()
