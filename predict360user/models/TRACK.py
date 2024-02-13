from typing import Tuple

import numpy as np
from keras import backend as K
from keras.layers import (
    LSTM,
    Concatenate,
    Dense,
    Flatten,
    Input,
    Lambda,
    Reshape,
    TimeDistributed,
)
from tensorflow import keras

from predict360user.base_model import KerasBaseModel, RunConfig
from predict360user.models.CVPR18 import selectImageInModel
from predict360user.models.pos_only_3d import delta_angle_from_ori_mot
from predict360user.utils.math360 import metric_orth_dist_cartesian

N_TILES_W = 384
N_TILES_H = 216


class TRACK(KerasBaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        self.m_window, self.h_window = (
            cfg.m_window,
            cfg.h_window,
        )

        # Defining model structure
        encoder_position_inputs = Input(shape=(self.m_window, 3))
        encoder_saliency_inputs = Input(shape=(self.m_window, N_TILES_H, N_TILES_W, 1))
        decoder_position_inputs = Input(shape=(1, 3))
        decoder_saliency_inputs = Input(shape=(self.h_window, N_TILES_H, N_TILES_W, 1))

        sense_pos_enc = LSTM(
            units=256, return_sequences=True, return_state=True, name="prop_lstm_1_enc"
        )

        sense_sal_enc = LSTM(
            units=256, return_sequences=True, return_state=True, name="prop_lstm_2_enc"
        )

        fuse_1_enc = LSTM(units=256, return_sequences=True, return_state=True)

        sense_pos_dec = LSTM(
            units=256, return_sequences=True, return_state=True, name="prop_lstm_1_dec"
        )

        sense_sal_dec = LSTM(
            units=256, return_sequences=True, return_state=True, name="prop_lstm_2_dec"
        )

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
        for curr_idx in range(self.h_window):
            out_enc_pos, state_h_1, state_c_1 = sense_pos_dec(
                inputs, initial_state=states_1
            )
            states_1 = [state_h_1, state_c_1]

            selected_timestep_saliency = Lambda(
                selectImageInModel, arguments={"curr_idx": curr_idx}
            )(decoder_saliency_inputs)
            flatten_timestep_saliency = Reshape((1, N_TILES_W * N_TILES_H))(
                selected_timestep_saliency
            )
            out_enc_sal, state_h_2, state_c_2 = sense_sal_dec(
                flatten_timestep_saliency, initial_state=states_2
            )
            states_2 = [state_h_2, state_c_2]

            conc_out_dec = Concatenate(axis=-1)([out_enc_sal, out_enc_pos])

            fuse_out_dec_1, state_h_fuse, state_c_fuse = fuse_1_dec(
                conc_out_dec, initial_state=states_fuse
            )
            fuse_out_dec_2 = TimeDistributed(fuse_2)(fuse_out_dec_1)

            outputs_delta = fc_layer_out(fuse_out_dec_2)

            decoder_pred = To_Position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(
            all_pos_outputs
        )

        # Define and compile model
        model = keras.Model(
            inputs=[
                encoder_position_inputs,
                encoder_saliency_inputs,
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

    def generate_batch(
        self, traces_l: list[np.ndarray], x_i_l: list
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def predict_for_sample(self, traces: np.ndarray, x_i) -> np.ndarray:
        raise NotImplementedError
