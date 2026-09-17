import logging
import sys
from pathlib import Path

from metaflow import conda, step

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from predict360user.pipelines.base_train_eval_flow import BaseTrainEvalFlow, _TF_PACKAGES

log = logging.getLogger("cvpr18_flow")


class CVPR18Flow(BaseTrainEvalFlow):
    model = "cvpr18"

    def build_model(self, cfg):
        from typing import Tuple
        import numpy as np
        import tensorflow as tf
        from keras.layers import (
            LSTM,
            Concatenate,
            Dense,
            Input,
            Lambda,
            Reshape,
            TimeDistributed,
        )
        from tensorflow import keras

        from predict360user.base_model import BaseModel
        from predict360user.utils.math360 import metric_orth_dist_cartesian

        def select_image(input_to_selector, curr_idx):
            return input_to_selector[:, curr_idx : curr_idx + 1]

        def add_time_axis(input_tensor):
            return tf.expand_dims(input_tensor, 1)

        def to_position(values):
            return values[0] + values[1]

        class CVPR18(BaseModel):
            def __init__(self, config) -> None:
                super().__init__(config)
                self.model = self._get_model()

            def _get_model(self):
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

                fuse_1 = Dense(units=256)
                fuse_2 = Dense(units=256)
                fc_layer_out = Dense(3)
                to_pos_layer = Lambda(to_position)

                prop_out_enc_1, state_h_1, state_c_1 = sense_pos_1_enc(encoder_position_inputs)
                states_1 = [state_h_1, state_c_1]
                prop_out_enc_2, state_h_2, state_c_2 = sense_pos_2_enc(prop_out_enc_1)
                states_2 = [state_h_2, state_c_2]

                all_pos_outputs = []
                inputs = decoder_position_inputs
                for curr_idx in range(self.cfg.h_window):
                    selected_timestep_saliency = Lambda(
                        select_image, arguments={"curr_idx": curr_idx}
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
                    prop_out_dec_2_timestep = Lambda(add_time_axis)(prop_out_dec_2)

                    conc_out_dec = Concatenate(axis=-1)(
                        [flatten_timestep_saliency, prop_out_dec_2_timestep]
                    )

                    fuse_out_1_dec = TimeDistributed(fuse_1)(conc_out_dec)
                    fuse_out_2_dec = TimeDistributed(fuse_2)(fuse_out_1_dec)

                    outputs_delta = fc_layer_out(fuse_out_2_dec)
                    decoder_pred = to_pos_layer([inputs, outputs_delta])

                    all_pos_outputs.append(decoder_pred)
                    inputs = decoder_pred

                decoder_outputs_pos = Concatenate(axis=1)(all_pos_outputs)
                model = keras.Model(
                    inputs=[
                        encoder_position_inputs,
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

        return CVPR18(cfg)

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
    CVPR18Flow()
