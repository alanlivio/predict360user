from typing import Tuple

import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.layers import (LSTM, Concatenate, Dense, Flatten, Input, Lambda,
                          Reshape, TimeDistributed)

from predict360user.models.base_model import (BaseModel,
                                              delta_angle_from_ori_mot,
                                              metric_orth_dist_cartesian,
                                              selectImageInModel)


class TRACK(keras.Model, BaseModel):

  def generate_batch(self, traces_l: list[np.array], x_i_l: list) -> Tuple[list, list]:
    raise NotImplementedError

  def predict_for_sample(self, traces: np.array, x_i) -> np.array:
    raise NotImplementedError


  def __init__(self, m_window: int, h_window: int, num_tiles_height: int, num_tiles_width: int) -> None:
    self.m_window, self.h_window = m_window, h_window
    # Defining model structure
    encoder_position_inputs = Input(shape=(m_window, 3))
    encoder_saliency_inputs = Input(shape=(m_window, num_tiles_height, num_tiles_width, 1))
    decoder_position_inputs = Input(shape=(1, 3))
    decoder_saliency_inputs = Input(shape=(h_window, num_tiles_height, num_tiles_width, 1))

    sense_pos_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_enc')

    sense_sal_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_enc')

    fuse_1_enc = LSTM(units=256, return_sequences=True, return_state=True)

    sense_pos_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_dec')

    sense_sal_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_dec')

    fuse_1_dec = LSTM(units=256, return_sequences=True, return_state=True)

    fuse_2 = Dense(units=256)

    # Act stack
    fc_layer_out = Dense(3)
    To_Position = Lambda(delta_angle_from_ori_mot)

    # Encoding
    out_enc_pos, state_h_1, state_c_1 = sense_pos_enc(encoder_position_inputs)
    states_1 = [state_h_1, state_c_1]

    out_flat_enc = TimeDistributed(Flatten())(encoder_saliency_inputs)
    out_enc_sal, state_h_2, state_c_2 = sense_sal_enc(out_flat_enc)
    states_2 = [state_h_2, state_c_2]

    conc_out_enc = Concatenate(axis=-1)([out_enc_sal, out_enc_pos])

    fuse_out_enc, state_h_fuse, state_c_fuse = fuse_1_enc(conc_out_enc)
    states_fuse = [state_h_fuse, state_c_fuse]

    # Decoding
    all_pos_outputs = []
    inputs = decoder_position_inputs
    for curr_idx in range(h_window):
      out_enc_pos, state_h_1, state_c_1 = sense_pos_dec(inputs, initial_state=states_1)
      states_1 = [state_h_1, state_c_1]

      selected_timestep_saliency = Lambda(selectImageInModel, arguments={'curr_idx': curr_idx})(decoder_saliency_inputs)
      flatten_timestep_saliency = Reshape((1, num_tiles_width * num_tiles_height))(selected_timestep_saliency)
      out_enc_sal, state_h_2, state_c_2 = sense_sal_dec(flatten_timestep_saliency, initial_state=states_2)
      states_2 = [state_h_2, state_c_2]

      conc_out_dec = Concatenate(axis=-1)([out_enc_sal, out_enc_pos])

      fuse_out_dec_1, state_h_fuse, state_c_fuse = fuse_1_dec(conc_out_dec, initial_state=states_fuse)
      fuse_out_dec_2 = TimeDistributed(fuse_2)(fuse_out_dec_1)

      outputs_delta = fc_layer_out(fuse_out_dec_2)

      decoder_pred = To_Position([inputs, outputs_delta])

      all_pos_outputs.append(decoder_pred)
      # Reinject the outputs as inputs for the next loop iteration as well as update the states
      inputs = decoder_pred

    # Concatenate all predictions
    decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(all_pos_outputs)

    # Define and compile model
    super().__init__(inputs=[encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs], outputs=decoder_outputs_pos)
    model_optimizer = keras.optimizers.Adam(lr=0.0005)
    self.compile(optimizer=model_optimizer, loss='mean_squared_error', metrics=[metric_orth_dist_cartesian])
