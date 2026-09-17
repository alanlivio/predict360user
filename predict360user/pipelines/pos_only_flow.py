import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import BaseTrainEvalFlow, _TF_PACKAGES

log = logging.getLogger("pos_only_flow")


class PosOnlyFlow(BaseTrainEvalFlow):
    model = "pos_only"

    def build_model(self, cfg):
        import os
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix and "XLA_FLAGS" not in os.environ:
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={conda_prefix}"
        from typing import Sequence, Tuple
        import numpy as np
        import pandas as pd
        import absl.logging
        import tensorflow as tf
        from keras.layers import Concatenate, Dense, Input, Lambda, LSTM
        from sklearn.utils.validation import check_is_fitted
        from tensorflow import keras

        from predict360user.base_model import BaseModel, batch_generator_fn
        from predict360user.utils.math360 import (
            cartesian_to_eulerian,
            eulerian_to_cartesian,
            metric_orth_dist_eulerian,
        )

        def delta_angle_from_ori_mag_dir(values):
            orientation = values[0]
            magnitudes = values[1] / 2.0
            directions = values[2]
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

        def batch_cartesian_to_norm_eulerian(positions_in_batch: np.ndarray) -> np.ndarray:
            eulerian_batch = [
                [cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch]
                for batch in positions_in_batch
            ]
            return np.array(eulerian_batch) / np.array([2 * np.pi, np.pi])

        def batch_norm_eulerian_to_cartesian(positions_in_batch: np.ndarray) -> list:
            positions_in_batch = positions_in_batch * np.array([2 * np.pi, np.pi])
            return [
                [eulerian_to_cartesian(pos[0], pos[1]) for pos in batch]
                for batch in positions_in_batch
            ]

        class PosOnly(BaseModel):
            def __init__(self, config) -> None:
                super().__init__(config)
                self.model = self._get_model()

            def _get_model(self):
                encoder_inputs = Input(shape=(self.cfg.m_window, 2))
                decoder_inputs = Input(shape=(1, 2))

                lstm_layer = LSTM(1024, return_sequences=True, return_state=True)
                decoder_dense_mot = Dense(2, activation="sigmoid")
                decoder_dense_dir = Dense(2, activation="tanh")
                To_Position = Lambda(delta_angle_from_ori_mag_dir)

                _, state_h, state_c = lstm_layer(encoder_inputs)
                states = [state_h, state_c]

                all_outputs = []
                inputs = decoder_inputs
                for _ in range(self.cfg.h_window):
                    decoder_pred, state_h, state_c = lstm_layer(inputs, initial_state=states)
                    outputs_delta = decoder_dense_mot(decoder_pred)
                    outputs_delta_dir = decoder_dense_dir(decoder_pred)
                    outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
                    all_outputs.append(outputs_pos)
                    inputs = outputs_pos
                    states = [state_h, state_c]

                decoder_outputs = Concatenate(axis=1)(all_outputs)
                model = keras.Model(
                    inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs
                )
                model_optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr)
                model.compile(optimizer=model_optimizer, loss=metric_orth_dist_eulerian)
                return model

            def fit(self, df: pd.DataFrame) -> BaseModel:
                absl.logging.set_verbosity(absl.logging.ERROR)
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

                if self.cfg.gpu_id:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)

                train_wins = df[df["partition"] == "train"]
                val_wins = df[df["partition"] == "val"]
                steps_per_ep_train = int(np.ceil(len(train_wins) / self.cfg.batch_size))
                steps_per_ep_validate = int(np.ceil(len(val_wins) / self.cfg.batch_size))

                def get_fit_data(df_part: pd.DataFrame) -> Tuple[tuple, np.ndarray]:
                    return (
                        (
                            batch_cartesian_to_norm_eulerian(df_part["m_window"].values),
                            batch_cartesian_to_norm_eulerian(df_part["trace"].values),
                        ),
                        batch_cartesian_to_norm_eulerian(df_part["h_window"].values),
                    )

                history_cb = keras.callbacks.History()
                fit_fn = getattr(self.model, "fit_generator", self.model.fit)
                fit_fn(
                    batch_generator_fn(self.cfg.batch_size, train_wins, get_fit_data),
                    validation_data=batch_generator_fn(
                        self.cfg.batch_size, val_wins, get_fit_data
                    ),
                    steps_per_epoch=steps_per_ep_train,
                    validation_steps=steps_per_ep_validate,
                    epochs=self.cfg.epochs,
                    initial_epoch=self.cfg.initial_epoch,
                    callbacks=[history_cb],
                    verbose=2,
                )
                self.history = history_cb.history
                self.is_fitted_ = True
                return self

            def predict(self, df: pd.DataFrame) -> Sequence:
                check_is_fitted(self)
                predict_data = (
                    batch_cartesian_to_norm_eulerian(df["m_window"].values),
                    batch_cartesian_to_norm_eulerian(df["trace"].values),
                )
                pred = self.model.predict(predict_data, verbose=2)
                return batch_norm_eulerian_to_cartesian(pred)

        return PosOnly(cfg)

    @conda(
        python="3.11",
        packages=_TF_PACKAGES,
    )
    @step
    def train_and_eval_run(self):
        import os
        import sys

        os.environ["PATH"] = f"{sys.prefix}/bin:" + os.environ.get("PATH", "")
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={sys.prefix}"

        from predict360user.run_config import set_random_seed

        self.seed = self.input
        set_random_seed(self.seed)

        cfg = self.build_run_config(self.seed)
        model = self.build_model(cfg)
        model.fit(self.df)

        loss_curve = getattr(model, "history", {}).get("loss", [0.0] * self.epochs)
        val_loss_curve = getattr(model, "history", {}).get("val_loss", [0.0] * self.epochs)
        self.record_run_results(model, loss_curve, val_loss_curve)
        self.next(self.join_runs)


if __name__ == "__main__":
    PosOnlyFlow()
