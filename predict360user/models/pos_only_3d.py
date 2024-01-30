import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import LSTM, Dense, Input, Lambda, TimeDistributed
from keras.metrics import mean_squared_error as mse
from tensorflow import keras

from predict360user.base_model import KerasModel, Config
from predict360user.utils.math360 import metric_orth_dist_cartesian


# This way we ensure that the network learns to predict the delta angle
def delta_angle_from_ori_mot(values):
    orientation = values[0]
    # The network returns values between 0 and 1, we force it to be between -1/2 and 1/2
    motion = values[1]
    return orientation + motion


class PosOnly3D(keras.Model, KerasModel):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model: keras.Model = self.build()

    def build(self) -> keras.Model:
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
        To_Position = Lambda(delta_angle_from_ori_mot)

        # Encoding
        encoder_outputs = sense_pos_1(encoder_inputs)
        encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_outputs)
        states = [state_h, state_c]

        # Decoding
        all_outputs = []
        inputs = decoder_inputs
        for _ in range(self.cfg.h_window):
            # # Run the decoder on one timestep
            inputs_1 = sense_pos_1(inputs)
            inputs_2 = sense_pos_2(inputs_1)
            inputs_3 = sense_pos_3(inputs_2)
            decoder_pred, state_h, state_c = lstm_layer_dec(
                inputs_3, initial_state=states
            )
            outputs_delta = decoder_dense_1(decoder_pred)
            outputs_delta = decoder_dense_2(outputs_delta)
            outputs_delta = decoder_dense_3(outputs_delta)
            outputs_pos = To_Position([inputs, outputs_delta])
            # Store the current prediction (we will concantenate all predictions later)
            all_outputs.append(outputs_pos)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = outputs_pos
            states = [state_h, state_c]
        if self.cfg.h_window == 1:
            decoder_outputs = outputs_pos
        else:
            # Concatenate all predictions
            decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model
        model = keras.Model(
            inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs
        )
        model_optimizer = keras.optimizers.Adam(lr=0.0005)
        model.compile(
            optimizer=model_optimizer,
            loss=self.loss_function,
            metrics=[metric_orth_dist_cartesian],
        )
        return model

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

    def predict_for_sample(self, pos_inputs) -> np.array:
        pred = self.model.predict(
            [np.array([pos_inputs[:-1]]), np.array([pos_inputs[-1:]])]
        )
        norm_factor = np.sqrt(
            pred[0, :, 0] * pred[0, :, 0]
            + pred[0, :, 1] * pred[0, :, 1]
            + pred[0, :, 2] * pred[0, :, 2]
        )
        data = {
            "x": pred[0, :, 0] / norm_factor,
            "y": pred[0, :, 1] / norm_factor,
            "z": pred[0, :, 2] / norm_factor,
        }
        return pd.DataFrame(data)
