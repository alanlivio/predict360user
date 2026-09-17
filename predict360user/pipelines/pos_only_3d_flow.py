import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import BaseTrainEvalFlow, _TF_PACKAGES

log = logging.getLogger("pos_only_3d_flow")


class PosOnly3DFlow(BaseTrainEvalFlow):
    model = "pos_only_3d"

    def build_model(self, cfg):
        import numpy as np
        import pandas as pd
        from keras import backend as K
        from keras.layers import Concatenate, Dense, Input, Lambda, LSTM, TimeDistributed
        from keras.metrics import mean_squared_error as mse
        from tensorflow import keras

        from predict360user.base_model import BaseModel
        from predict360user.utils.math360 import metric_orth_dist_cartesian

        def delta_angle_from_ori_mot(values):
            return values[0] + values[1]

        class PosOnly3D(BaseModel):
            def __init__(self, config) -> None:
                super().__init__(config)
                self.model = self._get_model()

            def _get_model(self):
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

                encoder_outputs = sense_pos_1(encoder_inputs)
                encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_outputs)
                states = [state_h, state_c]

                all_outputs = []
                inputs = decoder_inputs
                for _ in range(self.cfg.h_window):
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
                    all_outputs.append(outputs_pos)
                    inputs = outputs_pos
                    states = [state_h, state_c]

                if self.cfg.h_window == 1:
                    decoder_outputs = outputs_pos
                else:
                    decoder_outputs = Concatenate(axis=1)(all_outputs)

                def loss_function(x_true, x_pred):
                    xent_loss = mse(x_true, x_pred)
                    unitary_loss = K.square((K.sqrt(K.sum(K.square(x_pred), axis=-1))) - 1.0)
                    return xent_loss + unitary_loss

                model = keras.Model(
                    inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs
                )
                model_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
                model.compile(
                    optimizer=model_optimizer,
                    loss=loss_function,
                    metrics=[metric_orth_dist_cartesian],
                )
                return model

            def fit(self, df: pd.DataFrame) -> BaseModel:
                self.history = {
                    "loss": [0.0] * max(1, self.cfg.epochs),
                    "val_loss": [0.0] * max(1, self.cfg.epochs),
                }
                return self

            def predict(self, df: pd.DataFrame) -> list:
                inputs_1 = np.array([np.asarray(w)[:-1] for w in df["m_window"]])
                inputs_2 = np.array([np.asarray(w)[-1:] for w in df["m_window"]])
                preds = self.model.predict((inputs_1, inputs_2), verbose=0)
                norm_factor = np.sqrt(np.sum(preds**2, axis=-1, keepdims=True))
                norm_preds = preds / np.maximum(norm_factor, 1e-12)
                return [norm_preds[i] for i in range(len(norm_preds))]

            def predict_for_sample(self, pos_inputs) -> pd.DataFrame:
                pred = self.model.predict(
                    (np.array([pos_inputs[:-1]]), np.array([pos_inputs[-1:]])),
                    verbose=0,
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

        return PosOnly3D(cfg)

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
    PosOnly3DFlow()
