from users360 import *
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
sys.path.insert(0, './')

from users360.head_motion_prediction.position_only_baseline import *
# from users360.head_motion_prediction.Pos_Only_Baseline_3dLoss import *

DATASET_NAME: str
MODEL_NAME: str
M_WINDOW: int
H_WINDOW: int
INIT_WINDOW: int
END_WINDOW: int
METRIC = all_metrics['orthodromic']
np.random.seed(19680801)
EPOCHS = 500
NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216
RATE = 0.2
RESULTS_FOLDER: str
MODELS_FOLDER: str
EXP_NAME: str
PERC_VIDEOS_TRAIN = 0.8
PERC_USERS_TRAIN = 0.5
BATCH_SIZE = 128.0
TRAIN_MODEL = False
EVALUATE_MODEL = False
MODEL = None


def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2])
                         for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
    return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2 * np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)


def parse_args(parser: argparse.ArgumentParser):
    parser.description = 'train or evaluate users360 models and datasets.'
    model_names = ['pos_only']
    # model_names = ['TRACK', 'CVPR18', 'pos_only', 'no_motion', 'most_salient_point', 'true_saliency',
    #                'content_based_saliency', 'CVPR18_orig', 'TRACK_AblatSal', 'TRACK_AblatFuse', 'MM18', 'pos_only_3d_loss']
    dataset_names = ['David_MMSys_18']
    parser.add_argument('--train', action='store_true', help='Flag that tells run the training procedure.')
    parser.add_argument('--evaluate', action='store_true', help='Flag that tells run the evaluate procedure.')
    parser.add_argument('-gpu_id', required=True, action='store', dest='gpu_id',
                        help='The gpu used to train this network.')
    parser.add_argument('-model_name', choices=model_names, required=True, action='store', dest='model_name',
                        help='The name of the model used to reference the network structure used.')
    parser.add_argument('-dataset_name', choices=dataset_names, required=True, action='store', dest='dataset_name',
                        help='The name of the dataset used to train this network.')
    parser.add_argument('-m_window', required=True, action='store',
                        dest='m_window', help='Buffer window in timesteps', type=int)
    parser.add_argument('-i_window', required=True, action='store',
                        dest='i_window', help='Initial buffer to avoid stationary part', type=int)
    parser.add_argument('-h_window', required=True, action='store',
                        dest='h_window', help='Forecast window in timesteps used to predict (5 timesteps = 1 second)', type=int)
    args = parser.parse_args()

    TRAIN_MODEL = args.train
    EVALUATE_MODEL = args.evaluate

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name
    M_WINDOW = args.m_window
    H_WINDOW = args.h_window

    if args.i_window is None:
        INIT_WINDOW = M_WINDOW
    else:
        INIT_WINDOW = args.i_window
    END_WINDOW = H_WINDOW
    EXP_NAME = 'exp_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW)
    RESULTS_FOLDER = os.path.join(DATADIR, 'pos_only/results' + EXP_NAME)
    MODELS_FOLDER = os.path.join(DATADIR, 'pos_only/models/' + EXP_NAME)
    # Create results folder and models folder
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    # if MODEL_NAME == 'pos_only':
        # MODEL = create_pos_only_model(M_WINDOW, H_WINDOW)
    # elif MODEL_NAME == 'pos_only_3d_loss':
    #     obj = Pos_Only_Class(H_WINDOW)
    #     MODEL= obj.get_model()
    MODEL = create_pos_only_model(M_WINDOW, H_WINDOW)


def get_traces(video, user):
    row = Data.singleton().df_trajects.query(f"ds={DATASET_NAME} and ds_user={user} and ds_video={video}")
    return row['traces']


def generate_arrays(list_IDs, future_window):
    while True:
        encoder_pos_inputs_for_batch = []
        encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
        count = 0
        np.random.shuffle(list_IDs)
        for ID in list_IDs:
            user = ID['user']
            video = ID['video']
            x_i = ID['time-stamp']
            # Load the data
            # TODO
            encoder_pos_inputs_for_batch.append(get_traces(video, user)[x_i - M_WINDOW: x_i])
            decoder_pos_inputs_for_batch.append(get_traces(video, user)[x_i: x_i + 1])
            decoder_outputs_for_batch.append(get_traces(video, user)[x_i + 1: x_i + future_window + 1])
            count += 1
            if count == BATCH_SIZE:
                count = 0
                if MODEL_NAME == 'pos_only':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
                # elif MODEL_NAME in ['pos_only_3d_loss']:
                #     yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch)], np.array(decoder_outputs_for_batch))
                #     yield (np.array(encoder_sal_inputs_for_batch), np.array(decoder_outputs_for_batch))
                encoder_pos_inputs_for_batch = []
                encoder_sal_inputs_for_batch = []
                decoder_pos_inputs_for_batch = []
                decoder_sal_inputs_for_batch = []
                decoder_outputs_for_batch = []
        if count != 0:
            if MODEL_NAME == 'pos_only':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
            # elif MODEL_NAME in ['pos_only_3d_loss']:
            #     yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch)], np.array(decoder_outputs_for_batch))


def split_list_by_percentage(the_list, percentage):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Shuffle to select randomly
    np.random.shuffle(the_list)
    num_samples_first_part = int(len(the_list) * percentage)
    train_part = the_list[:num_samples_first_part]
    test_part = the_list[num_samples_first_part:]
    return train_part, test_part


def partition():
    videos_train, videos_test = split_list_by_percentage(videos, PERC_VIDEOS_TRAIN)
    users_train, users_test = split_list_by_percentage(users, PERC_USERS_TRAIN)

    partition = {}
    partition['train'] = []
    partition['test'] = []
    for video in videos_train:
        for user in users_train:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(INIT_WINDOW, trace_length - END_WINDOW):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['train'].append(ID)
    for video in videos_test:
        for user in users_test:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(INIT_WINDOW, trace_length - END_WINDOW):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['test'].append(ID)
    return partition


def train_model():
    # TODO partition
    steps_per_ep_train = np.ceil(len(partition['train']) / BATCH_SIZE)
    steps_per_ep_validate = np.ceil(len(partition['test']) / BATCH_SIZE)
    csv_logger = keras.callbacks.CSVLogger(RESULTS_FOLDER + '/results.csv')
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        MODELS_FOLDER + '/weights.hdf5', save_best_only=True, save_weights_only=True, mode='auto', period=1)
    model.fit_generator(
        generator=generate_arrays(partition['train'], future_window=H_WINDOW),
        verbose=1, steps_per_epoch=steps_per_ep_train, epochs=EPOCHS,
        callbacks=[csv_logger, model_checkpoint],
        validation_data=generate_arrays(partition['test'], future_window=H_WINDOW), validation_steps=steps_per_ep_validate
    )


def evaluate_model():
    traces_count = 0
    errors_per_video = {}
    errors_per_timestep = {}

    for ID in partition['test']:
        traces_count += 1
        logging.info('Progress:', traces_count, '/', len(partition['test']))

        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        if video not in errors_per_video.keys():
            errors_per_video[video] = {}

        # TODO get_traces(        encoder_pos_inputs_for_sample, np, rray([get_traces(video,user, x_i - M_WINDOW:x_i]])
        decoder_pos_inputs_for_sample = np.array(get_traces(video, user)[x_i:x_i + 1])

        groundtruth = get_traces(video, user)[x_i + 1: x_i + H_WINDOW + 1]

        if MODEL_NAME == 'pos_only':
            model_pred = model.predict([transform_batches_cartesian_to_normalized_eulerian(
                encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
            model_prediction = transform_normalized_eulerian_to_cartesian(model_pred)
        # elif MODEL_NAME == 'pos_only_3d_loss':
        #     model_pred = model.predict([np.array(encoder_pos_inputs_for_sample),
        #                                 np.array(decoder_pos_inputs_for_sample)])[0]

        for t in range(len(groundtruth)):
            if t not in errors_per_video[video].keys():
                errors_per_video[video][t] = []
            errors_per_video[video][t].append(METRIC(groundtruth[t], model_prediction[t]))
            if t not in errors_per_timestep.keys():
                errors_per_timestep[t] = []
            errors_per_timestep[t].append(METRIC(groundtruth[t], model_prediction[t]))

    for video_name in videos_test:
        for t in range(H_WINDOW):
            logging.info(video_name, t, np.mean(errors_per_video[video_name][t]), end=';')
        logging.info()

    import matplotlib.pyplot as plt
    avg_error_per_timestep = []
    for t in range(H_WINDOW):
        logging.info('Average', t, np.mean(errors_per_timestep[t]), end=';')
        avg_error_per_timestep.append(np.mean(errors_per_timestep[t]))
    logging.info("--")
    plt.plot(np.arange(H_WINDOW) + 1 * RATE, avg_error_per_timestep, label=MODEL_NAME)
    met = 'orthodromic'
    plt.title('Average %s in %s dataset using %s model' % (met, DATASET_NAME, MODEL_NAME))
    plt.ylabel(met)
    plt.xlabel('Prediction step s (sec.)')
    plt.legend()
    plt.show()


def make_dataset():
    logging.info(f"load_dataset")
    Data.singleton().load_dataset()
    df = Data.singleton().df_trajects
    logging.info(f"df_trajects.size={df.size}")
    logging.info(f"calc_trajects_entropy")
    calc_trajects_entropy()
    Data.singleton().save()


if __name__ == "__main__":
    if not exists("data/Data.pickle"):
        logging.error("data/Data.pickle does not exist. run make_dataset.py.")
    parser = argparse.ArgumentParser()
    parse_args(parser)
    if TRAIN_MODEL:
        logging.info("train ended")
        train_model()
    if EVALUATE_MODEL:
        logging.info("evaluation ended")
        evaluate_model()
