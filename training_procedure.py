#!env python

import argparse
import logging
import os
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from users360 import *

logging.basicConfig(level=logging.INFO, format='-- training_procedure.py %(levelname)-s %(message)s')


# consts
METRIC = all_metrics['orthodromic']
NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
RATE = 0.2
BATCH_SIZE = 128.0


def create_model() -> keras.models.Model:
    if MODEL_NAME == "pos_only":
        from users360.head_motion_prediction.position_only_baseline import \
            create_pos_only_model
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
    csv_logger_f = os.path.join(MODEL_FOLDER, 'train_results.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_logger_f)
    weights_f = os.path.join(MODEL_FOLDER, 'weights.hdf5')
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
        MODEL.load_weights(os.path.join(MODEL_FOLDER, 'weights.hdf5'))
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
        avg = np.mean(errors_per_timestep[t])
        logging.info(f"avg_error at timestep {t} {avg}")
        avg_error_per_timestep.append(avg)
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep, label=MODEL_NAME)
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    res_file = os.path.join(MODEL_FOLDER, f"results_evaluate_{DATASET_NAME}{DATASET_FILTER}_{str(PERC_TEST).replace('.','_')}")
    np.savetxt(f'{res_file}.csv', avg_error_per_timestep)
    plt.savefig(res_file)
    logging.info(f"avg_error per timestep at {t} {avg}")


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.description = 'train or evaluate users360 models and datasets'
    model_names = ['pos_only', 'TRACK', 'CVPR18', 'MM18', 'most_salient_point']
    entropy_classes = ['all', 'low', 'medium', 'hight']
    dataset_names = ['David_MMSys_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'Xu_CVPR_18', 'Xu_PAMI_18']
    parser.add_argument('-train', action='store_true',
                        help='Flag that tells run the train procedure')
    parser.add_argument('-train_entropy', nargs='?', default=entropy_classes[0],
                        choices=entropy_classes, help='Name of entropy_class to filter training')
    parser.add_argument('-evaluate', action='store_true',
                        help='Flag that tells run the evaluate procedure')
    parser.add_argument('-evaluate_entropy', nargs='?', default=entropy_classes[0],
                        choices=entropy_classes, help='Name of entropy_class to filter evaluation')
    parser.add_argument('-gpu_id', nargs='?', type=int, default=0,
                        help='Used cuda gpu')
    parser.add_argument('-model_name', nargs='?',
                        choices=model_names, default=model_names[0],
                        help='The name of the model used to reference the network structure used')
    parser.add_argument('-dataset_name', nargs='?',
                        choices=dataset_names, default=dataset_names[0],
                        help='The name of the dataset used to train this network')
    parser.add_argument('-init_window', nargs='?', type=int, default=30,
                        help='Initial buffer to avoid stationary part')
    parser.add_argument('-m_window', nargs='?', type=int, default=5,
                        help='Buffer window in timesteps',)
    parser.add_argument('-h_window', nargs='?', type=int, default=25,
                        help='Forecast window in timesteps used to predict (5 timesteps = 1 second)')
    parser.add_argument('-epochs', nargs='?', type=int, default=500,
                        help='epochs numbers (default is 500)')
    parser.add_argument('-perc_test', nargs='?', type=float, default=0.8,
                        help='Test percetage (default is 0.8)')

    # vars from argparse
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    DATASET_NAME = args.dataset_name
    INIT_WINDOW = args.init_window
    M_WINDOW = args.m_window
    H_WINDOW = args.h_window
    END_WINDOW = H_WINDOW
    PERC_TEST = args.perc_test

    # DATASET_SAMPLED_FOLDER
    DATASET_DIR_HMP = os.path.join('users360', 'head_motion_prediction', DATASET_NAME)
    DATASET_SAMPLED_FOLDER = os.path.join(DATASET_DIR_HMP, 'sampled_dataset')

    # MODEL_FOLDER
    WIN_PARAMS = 'init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + \
        str(H_WINDOW) + '_end_' + str(END_WINDOW)
    DATASET_FILTER = f'_{args.train_entropy}_entropy' if args.train_entropy != 'all' else ''
    if MODEL_NAME == 'pos_only':
        MODEL_FOLDER = os.path.join(DATADIR, f'{MODEL_NAME}_{WIN_PARAMS}_{DATASET_NAME}{DATASET_FILTER}')
    else:
        raise NotImplementedError()

    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    logging.info(f"MODEL_FOLDER is {MODEL_FOLDER}")

    # prepare partitions for train/evaluate
    if (args.train or args.evaluate):
        logging.info("")
        logging.info("prepare train/test partitions ...")
        df = get_df_trajects()
        X, Y = train_test_split(df, test_size=PERC_TEST, random_state=1)
        if args.train_entropy != 'all':
            X = X[X['entropy_class'] == args.train_entropy]
        if args.evaluate_entropy != 'all':
            Y = Y[Y['entropy_class'] == args.evaluate_entropy]
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
        USERS_TEST = Y['ds_user'].unique()

        # create model
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        logging.info("")
        logging.info("create_model ...")
        MODEL = create_model()
    if args.train:
        logging.info("")
        logging.info("train ...")
        logging.info(f"EPOCHS is {EPOCHS}")
        logging.info(f"PERC_TRAIN is {1-PERC_TEST}")
        logging.info(f"train_entropy is {args.train_entropy}")
        logging.info(f"X has {len(X)} traces")
        logging.info(f"PARTITION['train'] has {len(PARTITION['train'])} windows")
        train()
    if args.evaluate:
        logging.info("")
        logging.info("evaluate ...")
        logging.info(f"PERC_TRAIN is {PERC_TEST}")
        logging.info(f"evaluate_entropy is {args.evaluate_entropy}")
        logging.info(f"Y has {len(Y)} traces")
        logging.info(f"PARTITION['test'] has {len(PARTITION['test'])} windows")
        evaluate()
