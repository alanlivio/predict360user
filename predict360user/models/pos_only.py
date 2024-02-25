import logging
import os
from contextlib import suppress
from typing import Sequence, Tuple

import absl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Input, Lambda
from sklearn.utils.validation import check_is_fitted
from tensorflow import keras
from wandb.keras import WandbCallback

import wandb
from predict360user.base_model import BaseModel, batch_generator_fn
from predict360user.run_config import RunConfig
from predict360user.utils.math360 import (
    cartesian_to_eulerian,
    eulerian_to_cartesian,
    metric_orth_dist_eulerian,
)

log = logging.getLogger()


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


def batch_cartesian_to_normalized_eulerian(
    positions_in_batch: np.ndarray,
) -> np.ndarray:
    eulerian_batch = [
        [cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch]
        for batch in positions_in_batch
    ]
    eulerian_batch = np.array(eulerian_batch) / np.array([2 * np.pi, np.pi])
    return eulerian_batch


def batch_normalized_eulerian_to_cartesian(
    positions_in_batch: np.ndarray,
) -> np.ndarray:
    positions_in_batch = positions_in_batch * np.array([2 * np.pi, np.pi])
    cartesian_batch = [
        [eulerian_to_cartesian(pos[0], pos[1]) for pos in batch]
        for batch in positions_in_batch
    ]
    return cartesian_batch


def transform_normalized_eulerian_to_cartesian(positions) -> np.ndarray:
    positions = positions * np.array([2 * np.pi, np.pi])
    cartesian = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(cartesian)


class PosOnly(BaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    def get_model(self) -> keras.Model:
        # Defining model structure
        encoder_inputs = Input(shape=(self.cfg.m_window, 2))
        decoder_inputs = Input(shape=(1, 2))

        lstm_layer = LSTM(1024, return_sequences=True, return_state=True)
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
        model = keras.Model(
            inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs
        )
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr)
        model.compile(optimizer=model_optimizer, loss=metric_orth_dist_eulerian)
        return model

    def fit(self, df_wins: pd.DataFrame) -> BaseModel:
        log.info("fit ...")
        
        absl.logging.set_verbosity(absl.logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        self.model = self.get_model()
        initial_epoch = 0
        if wandb.run.resumed:
            try: 
                log.info("restoring fit from preivous interrupeted.")
                self.model.load_weights(wandb.restore("model-best.h5").name)
                initial_epoch = wandb.run.step
            except:
                log.error("restoring fit failed. starting new fit.") 

        if self.cfg.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)
            log.info(f"set visible cpu to {self.cfg.gpu_id}")

        # fit data
        train_wins = df_wins[df_wins["partition"] == "train"]
        val_wins = df_wins[df_wins["partition"] == "val"]
        steps_per_ep_train = np.ceil(len(train_wins) / self.cfg.batch_size)
        steps_per_ep_validate = np.ceil(len(val_wins) / self.cfg.batch_size)

        def get_fit_data(df_wins: pd.DataFrame) -> Tuple[list, list]:
            encoder_pos_inputs = df_wins["m_window"].values
            decoder_pos_inputs = df_wins["trace"].values
            decoder_outputs = df_wins["h_window"].values
            return (
                [
                    batch_cartesian_to_normalized_eulerian(encoder_pos_inputs),
                    batch_cartesian_to_normalized_eulerian(decoder_pos_inputs),
                ],
                batch_cartesian_to_normalized_eulerian(decoder_outputs),
            )

        self.model.fit_generator(
            generator=batch_generator_fn(self.cfg.batch_size, train_wins, get_fit_data),
            validation_data=batch_generator_fn(
                self.cfg.batch_size, val_wins, get_fit_data
            ),
            steps_per_epoch=steps_per_ep_train,
            validation_steps=steps_per_ep_validate,
            epochs=self.cfg.epochs,
            initial_epoch=initial_epoch,
            callbacks=[WandbCallback(save_model=True, monitor="loss")],
            verbose=2,
        )
        self.is_fitted_ = True
        return self

    def predict(self, df_wins: pd.DataFrame) -> Sequence:
        log.info("predict ...")
        check_is_fitted(self)

        # convert to model expected input
        encoder_pos_inputs = df_wins["m_window"].values
        decoder_pos_inputs = df_wins["trace"].values
        predict_data = [
            batch_cartesian_to_normalized_eulerian(encoder_pos_inputs),
            batch_cartesian_to_normalized_eulerian(decoder_pos_inputs),
        ]
        # predict
        pred = self.model.predict(predict_data, verbose=2)
        # convert bacth to cartesian
        return batch_normalized_eulerian_to_cartesian(pred)
