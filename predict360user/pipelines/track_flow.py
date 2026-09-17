import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import BaseTrainEvalFlow, _TF_PACKAGES

log = logging.getLogger("track_flow")


class TRACKFlow(BaseTrainEvalFlow):
    model = "track"

    def build_model(self, cfg):
        from typing import Tuple
        import numpy as np
        import tensorflow as tf
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

        from predict360user.base_model import BaseModel
        from predict360user.utils.math360 import metric_orth_dist_cartesian

        n_tiles_w = 384
        n_tiles_h = 216

        def select_image(input_to_selector, curr_idx):
            return input_to_selector[:, curr_idx : curr_idx + 1]

        def to_pos(values):
            return values[0] + values[1]

        class TRACK(BaseModel):
            def __init__(self, config) -> None:
                super().__init__(config)
                self.model = self._get_model()

            def _get_model(self):
                encoder_position_inputs = Input(shape=(self.cfg.m_window, 3))
                encoder_saliency_inputs = Input(shape=(self.cfg.m_window, n_tiles_h, n_tiles_w, 1))
                decoder_position_inputs = Input(shape=(1, 3))
                decoder_saliency_inputs = Input(shape=(self.cfg.h_window, n_tiles_h, n_tiles_w, 1))

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

                fc_layer_out = Dense(3)
                to_pos_layer = Lambda(to_pos)

                out_enc_pos, state_h_1, state_c_1 = sense_pos_enc(encoder_position_inputs)
                states_1 = [state_h_1, state_c_1]

                out_flat_enc = TimeDistributed(Flatten())(encoder_saliency_inputs)
                out_enc_sal, state_h_2, state_c_2 = sense_sal_enc(out_flat_enc)
                states_2 = [state_h_2, state_c_2]

                conc_out_enc = Concatenate(axis=-1)([out_enc_sal, out_enc_pos])
                fuse_out_enc, state_h_fuse, state_c_fuse = fuse_1_enc(conc_out_enc)
                states_fuse = [state_h_fuse, state_c_fuse]

                all_pos_outputs = []
                inputs = decoder_position_inputs
                for curr_idx in range(self.cfg.h_window):
                    out_enc_pos, state_h_1, state_c_1 = sense_pos_dec(
                        inputs, initial_state=states_1
                    )
                    states_1 = [state_h_1, state_c_1]

                    selected_timestep_saliency = Lambda(
                        select_image, arguments={"curr_idx": curr_idx}
                    )(decoder_saliency_inputs)
                    flatten_timestep_saliency = Reshape((1, n_tiles_w * n_tiles_h))(
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
                    decoder_pred = to_pos_layer([inputs, outputs_delta])

                    all_pos_outputs.append(decoder_pred)
                    inputs = decoder_pred

                decoder_outputs_pos = Concatenate(axis=1)(all_pos_outputs)
                model = keras.Model(
                    inputs=[
                        encoder_position_inputs,
                        encoder_saliency_inputs,
                        decoder_position_inputs,
                        decoder_saliency_inputs,
                    ],
                    outputs=decoder_outputs_pos,
                )
                model_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
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

        return TRACK(cfg)

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
    TRACKFlow()
