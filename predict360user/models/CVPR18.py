from typing import Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (
    LSTM,
    Concatenate,
    Dense,
    Input,
    Lambda,
    Reshape,
    TimeDistributed,
)

from predict360user.base_model import KerasBaseModel, RunConfig
from predict360user.models.pos_only_3d import delta_angle_from_ori_mot
from predict360user.utils.math360 import metric_orth_dist_cartesian


def selectImageInModel(input_to_selector, curr_idx):
    selected_image = input_to_selector[:, curr_idx : curr_idx + 1]
    return selected_image


def add_timestep_axis(input):
    return tf.expand_dims(input, 1)


class CVPR18(KerasBaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.model: keras.Model = self.build()

    def generate_batch(
        self, traces_l: list[np.ndarray], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
        raise NotImplementedError

    def build(self) -> keras.Model:
        self.m_window, self.h_window = self.cfg.m_window, self.cfg.h_window
        # Defining model structure
        encoder_position_inputs = Input(shape=(self.cfg.m_window, 3))
        decoder_saliency_inputs = Input(
            shape=(
                self.cfg.h_window,
                self.cfg.num_tiles_height,
                self.cfg.num_tiles_width,
                1,
            )
        )
        decoder_position_inputs = Input(shape=(1, 3))

        # Propioception stack
        sense_pos_1_enc = LSTM(
            units=256, return_sequences=True, return_state=True, name="prop_lstm_1_enc"
        )
        sense_pos_2_enc = LSTM(
            units=256, return_sequences=False, return_state=True, name="prop_lstm_2_enc"
        )

        sense_pos_1_dec = LSTM(
            units=256, return_sequences=True, return_state=True, name="prop_lstm_1_dec"
        )
        sense_pos_2_dec = LSTM(
            units=256, return_sequences=False, return_state=True, name="prop_lstm_2_dec"
        )

        # Fuse stack
        fuse_1 = Dense(units=256)
        fuse_2 = Dense(units=256)

        # Act stack
        fc_layer_out = Dense(3)
        To_Position = Lambda(delta_angle_from_ori_mot)

        prop_out_enc_1, state_h_1, state_c_1 = sense_pos_1_enc(encoder_position_inputs)
        states_1 = [state_h_1, state_c_1]
        prop_out_enc_2, state_h_2, state_c_2 = sense_pos_2_enc(prop_out_enc_1)
        states_2 = [state_h_2, state_c_2]

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.cfg.h_window):
            selected_timestep_saliency = Lambda(
                selectImageInModel, arguments={"curr_idx": curr_idx}
            )(decoder_saliency_inputs)
            flatten_timestep_saliency = Reshape(
                (1, self.cfg.num_tiles_width * self.cfg.num_tiles_height)
            )(selected_timestep_saliency)
            prop_out_dec_1, state_h_1, state_c_1 = sense_pos_1_dec(
                inputs, initial_state=states_1
            )
            states_1 = [state_h_1, state_c_1]
            prop_out_dec_2, state_h_2, state_c_2 = sense_pos_2_dec(
                prop_out_dec_1, initial_state=states_2
            )
            states_2 = [state_h_2, state_c_2]
            prop_out_dec_2_timestep = Lambda(add_timestep_axis)(prop_out_dec_2)

            conc_out_dec = Concatenate(axis=-1)(
                [flatten_timestep_saliency, prop_out_dec_2_timestep]
            )

            fuse_out_1_dec = TimeDistributed(fuse_1)(conc_out_dec)
            fuse_out_2_dec = TimeDistributed(fuse_2)(fuse_out_1_dec)

            outputs_delta = fc_layer_out(fuse_out_2_dec)
            decoder_pred = To_Position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(
            all_pos_outputs
        )
        # decoder_outputs_img = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model
        model = keras.Model(
            inputs=[
                encoder_position_inputs,
                decoder_position_inputs,
                decoder_saliency_inputs,
            ],
            outputs=decoder_outputs_pos,
        )

        model_optimizer = keras.optimizers.Adam(lr=0.0005)
        model.compile(
            optimizer=model_optimizer,
            loss="mean_squared_error",
            metrics=[metric_orth_dist_cartesian],
        )
        return model
