from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Input, Lambda
from tensorflow import keras

from predict360user.model_config import Config
from predict360user.utils.math360 import (
    cartesian_to_eulerian,
    eulerian_to_cartesian,
    metric_orth_dist_eulerian,
)


# This way we ensure that the network learns to predict the delta angle
def delta_angle_from_ori_mag_dir(values):
    orientation = values[0]
    magnitudes = values[1] / 2.0
    directions = values[2]
    # The network returns values between 0 and 1, we force it to be between -2/5 and 2/5
    motion = magnitudes * directions

    yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
    pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

    cond_above = tf.cast(tf.greater(pitch_pred_wo_corr, 1.0), tf.float32)
    cond_correct = tf.cast(
        tf.logical_and(
            tf.less_equal(pitch_pred_wo_corr, 1.0),
            tf.greater_equal(pitch_pred_wo_corr, 0.0),
        ),
        tf.float32,
    )
    cond_below = tf.cast(tf.less(pitch_pred_wo_corr, 0.0), tf.float32)

    pitch_pred = (
        cond_above * (1.0 - (pitch_pred_wo_corr - 1.0))
        + cond_correct * pitch_pred_wo_corr
        + cond_below * (-pitch_pred_wo_corr)
    )
    yaw_pred = tf.math.mod(
        cond_above * (yaw_pred_wo_corr - 0.5)
        + cond_correct * yaw_pred_wo_corr
        + cond_below * (yaw_pred_wo_corr - 0.5),
        1.0,
    )
    return tf.concat([yaw_pred, pitch_pred], -1)


def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch) -> np.array:
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [
        [cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch]
        for batch in positions_in_batch
    ]
    eulerian_batches = np.array(eulerian_batches) / np.array([2 * np.pi, np.pi])
    return eulerian_batches


def transform_normalized_eulerian_to_cartesian(positions) -> np.array:
    positions = positions * np.array([2 * np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)


class PosOnly(keras.Model):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        
        # Defining model structure
        encoder_inputs = Input(shape=(self.cfg.m_window, 2))
        decoder_inputs = Input(shape=(1, 2))

        # TODO: try tf.compat.v1.keras.layers.CuDNNLSTM
        lstm_layer = LSTM(1024, return_sequences=True, return_state=True)
        # print(type(lstm_layer))
        # print(isinstance(lstm_layer, LSTM))
        # print(isinstance(lstm_layer, CuDNNLSTM))
        decoder_dense_mot = Dense(2, activation="sigmoid")
        decoder_dense_dir = Dense(2, activation="tanh")
        To_Position = Lambda(delta_angle_from_ori_mag_dir)

        # Encoding
        _, state_h, state_c = lstm_layer(encoder_inputs)
        states = [state_h, state_c]

        # Decoding
        all_outputs = []
        inputs = decoder_inputs
        for _ in range(self.cfg.h_window):
            # # Run the decoder on one timestep
            decoder_pred, state_h, state_c = lstm_layer(inputs, initial_state=states)
            outputs_delta = decoder_dense_mot(decoder_pred)
            outputs_delta_dir = decoder_dense_dir(decoder_pred)
            outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
            # Store the current prediction (we will concantenate all predictions later)
            all_outputs.append(outputs_pos)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = outputs_pos
            states = [state_h, state_c]

        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        # decoder_outputs = all_outputs

        # Define and compile model
        super().__init__(
            inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs
        )
        model_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.lr)
        self.compile(optimizer=model_optimizer, loss=metric_orth_dist_eulerian)

    def generate_batch(
        self, traces_l: list[np.array], x_i_l: list
    ) -> Tuple[list, list]:
        encoder_pos_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_outputs_for_batch = []
        for traces, x_i in zip(traces_l, x_i_l):
            encoder_pos_inputs_for_batch.append(traces[x_i - self.cfg.m_window : x_i])
            decoder_pos_inputs_for_batch.append(traces[x_i : x_i + 1])
            decoder_outputs_for_batch.append(
                traces[x_i + 1 : x_i + self.cfg.h_window + 1]
            )
        return (
            [
                transform_batches_cartesian_to_normalized_eulerian(
                    encoder_pos_inputs_for_batch
                ),
                transform_batches_cartesian_to_normalized_eulerian(
                    decoder_pos_inputs_for_batch
                ),
            ],
            transform_batches_cartesian_to_normalized_eulerian(
                decoder_outputs_for_batch
            ),
        )

    def predict_for_sample(self, traces: np.array, x_i: int) -> np.array:
        encoder_pos_inputs_for_sample = np.array(
            [traces[x_i - self.cfg.m_window : x_i]]
        )
        decoder_pos_inputs_for_sample = np.array([traces[x_i : x_i + 1]])
        inputs = [
            transform_batches_cartesian_to_normalized_eulerian(
                encoder_pos_inputs_for_sample
            ),
            transform_batches_cartesian_to_normalized_eulerian(
                decoder_pos_inputs_for_sample
            ),
        ]
        model_pred = super().predict(inputs, verbose=0)[0]
        return transform_normalized_eulerian_to_cartesian(model_pred)
