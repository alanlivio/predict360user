from typing import Tuple

import numpy as np
from keras import backend as K
from keras.layers import LSTM, Dense, Input, Lambda
from keras.models import Model
from tensorflow import keras

from predict360user.models.base_model import (BaseModel,
                                              delta_angle_from_ori_mag_dir,
                                              metric_orth_dist_eulerian)
from predict360user.utils import (cartesian_to_eulerian, eulerian_to_cartesian,
                                  rotationBetweenVectors)


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

class PosOnly(BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    encoder_pos_inputs_for_batch = []
    decoder_pos_inputs_for_batch = []
    decoder_outputs_for_batch = []
    for traces, x_i in zip(traces_l, x_i_l):
      encoder_pos_inputs_for_batch.append(traces[x_i - self.m_window:x_i])
      decoder_pos_inputs_for_batch.append(traces[x_i:x_i + 1])
      decoder_outputs_for_batch.append(traces[x_i + 1:x_i + self.h_window + 1])
    return ([
        transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),
        transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)
    ], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    encoder_pos_inputs_for_sample = np.array([traces[x_i - self.m_window:x_i]])
    decoder_pos_inputs_for_sample = np.array([traces[x_i:x_i + 1]])
    inputs = [
        transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
        transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)
    ]
    model_pred = super().predict(inputs, verbose=0)[0]
    return transform_normalized_eulerian_to_cartesian(model_pred)

  def __init__(self, m_window: int, h_window: int) -> None:
    self.m_window, self.h_window = m_window, h_window

    # Defining model structure
    encoder_inputs = Input(shape=(self.m_window, 2))
    decoder_inputs = Input(shape=(1, 2))

    # TODO: try tf.compat.v1.keras.layers.CuDNNLSTM
    lstm_layer = LSTM(1024, return_sequences=True, return_state=True)
    # print(type(lstm_layer))
    # print(isinstance(lstm_layer, LSTM))
    # print(isinstance(lstm_layer, CuDNNLSTM))
    decoder_dense_mot = Dense(2, activation='sigmoid')
    decoder_dense_dir = Dense(2, activation='tanh')
    To_Position = Lambda(delta_angle_from_ori_mag_dir)

    # Encoding
    _, state_h, state_c = lstm_layer(encoder_inputs)
    states = [state_h, state_c]

    # Decoding
    all_outputs = []
    inputs = decoder_inputs
    for _ in range(self.h_window):
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
    super().__init__([encoder_inputs, decoder_inputs], decoder_outputs)
    model_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    self.compile(optimizer=model_optimizer, loss=metric_orth_dist_eulerian)

class NoMotion(BaseModel):

  def __init__(self, h_window) -> None:
    self.h_window = h_window

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    model_pred = np.repeat(traces[x_i:x_i + 1], self.h_window, axis=0)
    return model_pred

class Interpolation(BaseModel):

  def __init__(self, h_window) -> None:
    self.h_window = h_window

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    rotation = rotationBetweenVectors(traces[-2], traces[-1])
    return [rotation.rotate(trace) for trace in traces[x_i:x_i + self.h_window]]

class Regression(Model):
  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError