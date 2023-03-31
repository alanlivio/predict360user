import numpy as np
import pandas as pd
import os
from contextlib import redirect_stderr
from os.path import exists
from abc import ABC
from typing import Tuple

with redirect_stderr(open(os.devnull, 'w')):
  import tensorflow as tf
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  from tensorflow import keras
  from keras.metrics import mean_squared_error as mse
  from keras import backend as K
  from keras.layers import (LSTM, Concatenate, ConvLSTM2D, Convolution2D, Dense, Flatten, Input,
                            Lambda, MaxPooling2D, Reshape, TimeDistributed)
  # from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from .utils.fov import (eulerian_to_cartesian, cartesian_to_eulerian, calc_actual_entropy)
from .dataset import get_class_name
from . import config


def metric_orth_dist(true_position, pred_position) -> float:
  yaw_true = (true_position[:, :, 0:1] - 0.5) * 2 * np.pi
  pitch_true = (true_position[:, :, 1:2] - 0.5) * np.pi
  # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
  yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2 * np.pi
  pitch_pred = (pred_position[:, :, 1:2] - 0.5) * np.pi
  # Finally compute orthodromic distance
  delta_long = tf.abs(tf.atan2(tf.sin(yaw_true - yaw_pred), tf.cos(yaw_true - yaw_pred)))
  numerator = tf.sqrt(
      tf.pow(tf.cos(pitch_pred) * tf.sin(delta_long), 2.0) + tf.pow(
          tf.cos(pitch_true) * tf.sin(pitch_pred) -
          tf.sin(pitch_true) * tf.cos(pitch_pred) * tf.cos(delta_long), 2.0))
  denominator = tf.sin(pitch_true) * tf.sin(pitch_pred) + tf.cos(pitch_true) * tf.cos(
      pitch_pred) * tf.cos(delta_long)
  great_circle_distance = tf.abs(tf.atan2(numerator, denominator))
  return great_circle_distance


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


class ModelABC(ABC):

  model: keras.models.Model

  def build(self) -> keras.Model:
    raise NotImplementedError

  def load(self, model_file: str) -> keras.Model:
    raise NotImplementedError

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError


co_metric = {'metric_orth_dist': metric_orth_dist}


class PosOnly(ModelABC):
  def __init__(self, m_window: int, h_window: int) -> None:
    self.m_window, self.h_window = m_window, h_window

  def load(self, model_file: str) -> keras.Model:
    assert exists(model_file)
    self._model = keras.models.load_model(model_file, custom_objects=co_metric)

  # This way we ensure that the network learns to predict the delta angle
  def toPosition(self, values):
    orientation = values[0]
    magnitudes = values[1] / 2.0
    directions = values[2]
    # The network returns values between 0 and 1, we force it to be between -2/5 and 2/5
    motion = magnitudes * directions

    yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
    pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

    cond_above = tf.cast(tf.greater(pitch_pred_wo_corr, 1.0), tf.float32)
    cond_correct = tf.cast(
        tf.logical_and(tf.less_equal(pitch_pred_wo_corr, 1.0),
                       tf.greater_equal(pitch_pred_wo_corr, 0.0)), tf.float32)
    cond_below = tf.cast(tf.less(pitch_pred_wo_corr, 0.0), tf.float32)

    pitch_pred = cond_above * (
        1.0 - (pitch_pred_wo_corr - 1.0)) + cond_correct * pitch_pred_wo_corr + cond_below * (
            -pitch_pred_wo_corr)
    yaw_pred = tf.math.mod(
        cond_above * (yaw_pred_wo_corr - 0.5) + cond_correct * yaw_pred_wo_corr + cond_below *
        (yaw_pred_wo_corr - 0.5), 1.0)
    return tf.concat([yaw_pred, pitch_pred], -1)

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

  def build(self) -> keras.models.Model:
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
    To_Position = Lambda(self.toPosition)

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
    self._model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    self._model.compile(optimizer=model_optimizer, loss=metric_orth_dist)
    return self._model

  def predict(self, traces: np.array, x_i) -> np.array:
    encoder_pos_inputs_for_sample = np.array([traces[x_i - self.m_window:x_i]])
    decoder_pos_inputs_for_sample = np.array([traces[x_i:x_i + 1]])
    batchs = [
        transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample),
        transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)
    ]
    return self._model.predict(batchs,verbose=0)[0]


class PosOnly_Auto(PosOnly):
  def load_models(self, model_file_low: str, model_file_medium: str, model_file_hight: str,
                  threshold_medium, threshold_hight) -> None:
    assert exists(model_file_low)
    assert exists(model_file_medium)
    assert exists(model_file_hight)
    self.model_low = keras.models.load_model(model_file_low)
    self.model_medium = keras.models.load_model(model_file_medium)
    self.model_hight = keras.models.load_model(model_file_hight)
    self.threshold_medium, self.threshold_hight = threshold_medium, threshold_hight

  def predict(
      self,
      train_entropy: str,
      traces: np.array,
      x_i,
  ) -> np.array:
    if train_entropy == 'auto':
      window = traces
    elif train_entropy == 'auto_m_window':
      window = traces[x_i - self.m_window:x_i]
    elif train_entropy == 'auto_since_start':
      window = traces[0:x_i]
    else:
      raise RuntimeError()
    inputs, _ = self.get_inputs_outputs_per_sample(traces, x_i)
    a_ent = calc_actual_entropy(window)
    actS_c = get_class_name(a_ent, self.threshold_medium, self.threshold_hight)
    if actS_c == 'low':
      model_pred = self.model_low.predict(inputs)[0]
    elif actS_c == 'medium':
      model_pred = self.model_medium.predict(inputs)[0]
    elif actS_c == 'hight':
      model_pred = self.model_hight.predict(inputs)[0]
    else:
      raise RuntimeError()
    return transform_normalized_eulerian_to_cartesian(model_pred)


class NoMotion(ModelABC):
  def predict(self, traces: np.array, x_i) -> np.array:
    model_pred = np.repeat(traces[x_i:x_i + 1], self.h_window, axis=0)
    return model_pred


class PosOnly3D(PosOnly):

  # This way we ensure that the network learns to predict the delta angle
  def toPosition(self, values):
    orientation = values[0]
    delta = values[1]
    return orientation + delta

  def loss_function(self, x_true, x_pred):
    xent_loss = mse(x_true, x_pred)
    unitary_loss = K.square((K.sqrt(K.sum(K.square(x_pred), axis=-1))) - 1.0)
    loss = xent_loss + unitary_loss
    return loss

  def build(self, h_window) -> keras.models.Model:
    # Defining model structure
    encoder_inputs = Input(shape=(None, 3))
    decoder_inputs = Input(shape=(1, 3))

    sense_pos_1 = TimeDistributed(Dense(256))
    sense_pos_2 = TimeDistributed(Dense(256))
    sense_pos_3 = TimeDistributed(Dense(256))
    lstm_layer_enc = LSTM(1024, return_sequences=True, return_state=True)
    lstm_layer_dec = LSTM(1024, return_sequences=True, return_state=True)
    decoder_dense_1 = Dense(256)
    decoder_dense_2 = Dense(256)
    decoder_dense_3 = Dense(3)
    To_Position = Lambda(self.toPosition)

    # Encoding
    encoder_outputs = sense_pos_1(encoder_inputs)
    encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_outputs)
    states = [state_h, state_c]

    # Decoding
    all_outputs = []
    inputs = decoder_inputs
    for _ in range(h_window):
      # # Run the decoder on one timestep
      inputs_1 = sense_pos_1(inputs)
      inputs_2 = sense_pos_2(inputs_1)
      inputs_3 = sense_pos_3(inputs_2)
      decoder_pred, state_h, state_c = lstm_layer_dec(inputs_3, initial_state=states)
      outputs_delta = decoder_dense_1(decoder_pred)
      outputs_delta = decoder_dense_2(outputs_delta)
      outputs_delta = decoder_dense_3(outputs_delta)
      outputs_pos = To_Position([inputs, outputs_delta])
      # Store the current prediction (we will concantenate all predictions later)
      all_outputs.append(outputs_pos)
      # Reinject the outputs as inputs for the next loop iteration as well as update the states
      inputs = outputs_pos
      states = [state_h, state_c]
    if h_window == 1:
      decoder_outputs = outputs_pos
    else:
      # Concatenate all predictions
      decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # Define and compile model
    self._model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model_optimizer = keras.optmizers.Adam(lr=0.0005)
    self._model.compile(optimizer=model_optimizer,
                        loss=self.loss_function,
                        metrics=[metric_orth_dist])
    return self._model

  def predict(self, pos_inputs) -> np.array:
    pred = self._model.predict([np.array([pos_inputs[:-1]]), np.array([pos_inputs[-1:]])])
    norm_factor = np.sqrt(pred[0, :, 0] * pred[0, :, 0] + pred[0, :, 1] * pred[0, :, 1] +
                          pred[0, :, 2] * pred[0, :, 2])
    data = {
        'x': pred[0, :, 0] / norm_factor,
        'y': pred[0, :, 1] / norm_factor,
        'z': pred[0, :, 2] / norm_factor
    }
    return pd.DataFrame(data)