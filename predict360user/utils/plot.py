import os
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from colour import Color
from numpy.random import randint
from plotly.subplots import make_subplots

from predict360user.utils.math360 import cartesian_to_eulerian, fov_points
from predict360user.utils.tileset360 import (
    TILESET_DEFAULT,
    TileSet,
    TileSetVoro,
    tile_points,
)


class Plot:
    def __init__(self, tileset=TILESET_DEFAULT, thumb_path: str = None) -> None:
        self.tileset = tileset
        self.thumb_path = thumb_path
        self.traces = []
        self.predictions = {}
        self.title = "Traces in Projection"
        
        self.dft_start_c = Color("DarkBlue").hex
        self.dft_end_c = Color("SkyBlue").hex
        self.prediction_start_c = Color("DarkGreen").hex
        self.prediction_end_c = Color("LightGreen").hex

    def add_traces(
        self, traces: np.ndarray, start_c=None, end_c=None
    ) -> None:
        if start_c is None:
            start_c = self.dft_start_c
        if end_c is None:
            end_c = self.dft_end_c
        self.traces.append((traces, start_c, end_c))

    def add_predictions(self, predictions: dict) -> None:
        self.predictions = predictions

    def _add_traces_data(self, traces: np.ndarray, start_c, end_c, visible=True) -> list:
        thetas = []
        phis = []
        for pt in traces:
            t, p = cartesian_to_eulerian(pt[0], pt[1], pt[2])
            thetas.append(t)
            phis.append(p)
            
        data = []
        # Start marker
        data.append(
            go.Scatter(
                x=[thetas[0]],
                y=[phis[0]],
                mode="markers",
                visible=visible,
                marker=dict(size=8, color=start_c),
                showlegend=False,
            )
        )
        # End marker
        data.append(
            go.Scatter(
                x=[thetas[-1]],
                y=[phis[-1]],
                mode="markers",
                visible=visible,
                marker=dict(size=8, color=end_c),
                showlegend=False,
            )
        )
        
        n = len(traces)
        if n > 1:
            colors = [x.hex for x in list(Color(start_c).range_to(Color(end_c), n))]
            for index in range(n - 1):
                data.append(
                    go.Scatter(
                        x=thetas[index : index + 2],
                        y=phis[index : index + 2],
                        visible=visible,
                        hovertext=f"trace[{index}]",
                        hoverinfo="text",
                        mode="lines",
                        line=dict(width=5, color=colors[index]),
                        showlegend=False,
                    )
                )
        return data

    def show(self) -> None:
        # Load background image
        img = None
        if self.thumb_path and os.path.exists(self.thumb_path):
            img = cv2.imread(self.thumb_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            # Neutral gray background
            img = np.full((480, 960, 3), 240, dtype=np.uint8)

        x_coords = np.linspace(0, 2 * np.pi, img.shape[1])
        y_coords = np.linspace(0, np.pi, img.shape[0])

        fig = px.imshow(img, x=x_coords, y=y_coords)

        # 1. Base traces
        base_data = []
        for traces, start_c, end_c in self.traces:
            base_data.extend(self._add_traces_data(traces, start_c, end_c, visible=True))
            
        for d in base_data:
            fig.add_trace(d)
            
        # 2. Predictions
        if self.predictions:
            n_pre = len(fig.data)
            steps = []
            
            all_pred_data = []
            for i, (_, pred_traces) in enumerate(self.predictions.items()):
                pred_data = self._add_traces_data(pred_traces, self.prediction_start_c, self.prediction_end_c, visible=False)
                all_pred_data.append((len(pred_data), pred_data))
                
            for _, pred_data in all_pred_data:
                for d in pred_data:
                    fig.add_trace(d)
                    
            n_end = len(fig.data)
            
            curr_idx = n_pre
            for i, (len_pred, _) in enumerate(all_pred_data):
                step = dict(
                    method="update",
                    args=[{"visible": [True] * n_pre + [False] * (n_end - n_pre)}],
                )
                step["args"][0]["visible"][curr_idx : curr_idx + len_pred] = [True] * len_pred
                steps.append(step)
                curr_idx += len_pred
                
            fig.update_layout(
                sliders=[dict(active=0, currentvalue={"prefix": "Prediction at: "}, steps=steps)]
            )

        fig.update_layout(
            title=self.title,
            xaxis=dict(title="Yaw (Theta)", range=[0, 2 * np.pi]),
            yaxis=dict(title="Pitch (Phi)", range=[np.pi, 0]),
            width=960,
            height=480,
            showlegend=False,
        )

        fig.show()

    def show_fov(self, trace) -> None:
        assert len(trace) == 3

        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "image"}]]
        )

        img = None
        if self.thumb_path and os.path.exists(self.thumb_path):
            img = cv2.imread(self.thumb_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        if img is None:
            img = np.full((480, 960, 3), 240, dtype=np.uint8)

        x_coords = np.linspace(0, 2 * np.pi, img.shape[1])
        y_coords = np.linspace(0, np.pi, img.shape[0])

        bg_fig = px.imshow(img, x=x_coords, y=y_coords)
        fig.add_trace(bg_fig.data[0], row=1, col=1)

        t_theta, t_phi = cartesian_to_eulerian(*trace)
        fig.add_trace(
            go.Scatter(
                x=[t_theta],
                y=[t_phi],
                mode="markers",
                marker=dict(size=8, color="red"),
                showlegend=False,
            ),
            row=1, col=1
        )

        points = fov_points(*trace)
        fov_thetas = []
        fov_phis = []
        for pt in points:
            t, p = cartesian_to_eulerian(*pt)
            fov_thetas.append(t)
            fov_phis.append(p)
        fov_thetas.append(fov_thetas[0])
        fov_phis.append(fov_phis[0])

        fig.add_trace(
            go.Scatter(
                x=fov_thetas,
                y=fov_phis,
                mode="lines",
                line=dict(color="blue", width=3),
                showlegend=False,
            ),
            row=1, col=1
        )

        heatmap = self.tileset.request(trace)
        if isinstance(self.tileset, TileSetVoro):
            heatmap = np.reshape(heatmap, self.tileset.shape)
        x = [str(x) for x in range(1, heatmap.shape[1] + 1)]
        y = [str(y) for y in range(1, heatmap.shape[0] + 1)]
        erp_heatmap = px.imshow(heatmap, text_auto=True, x=x, y=y)
        for t in erp_heatmap["data"]:
            fig.add_trace(t, row=1, col=2)

        fig.update_xaxes(range=[0, 2 * np.pi], row=1, col=1)
        fig.update_yaxes(range=[np.pi, 0], row=1, col=1)

        title = f"trace_[{trace[0]:.2},{trace[1]:.2},{trace[2]:.2}]_{self.tileset.prefix}"
        fig.update_layout(width=1000, height=500, showlegend=False, title_text=title)
        fig.show()
