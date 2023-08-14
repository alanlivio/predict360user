import numpy as np
import pandas as pd
from predict360user.utils import show_or_save, DEFAULT_SAVEDIR
from predict360user.trainer import Dataset
from predict360user.tileset import TileSet, TileSetVoro, TileCover

def calc_tileset_reqs_metrics(df: pd.DataFrame, tileset_l: list[TileSet]) -> None:
    if len(df) >= 4:
        log.info("df.size >= 4, it will take for some time")

    def _trace_mestrics_np(trace, tileset) -> np.array:
        heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
        return np.array([np.sum(heatmap), vp_quality, area_out])

    def _traject_metrics_np(traces, tileset) -> np.array:
        return np.apply_along_axis(_trace_mestrics_np, 1, traces, tileset=tileset)

    for tileset in tileset_l:
        column_name = f"metrics_{tileset.title}"
        metrics_np = df["traces"].progress_apply(
            _traject_metrics_np, tileset=tileset
        )
        df[column_name] = metrics_np
        assert not df[column_name].empty

def show_tileset_reqs_metrics(df: pd.DataFrame) -> None:
    # check
    columns = [
        column for column in df.columns if column.startswith("metrics_")
    ]
    assert len(columns), "run calc_tileset_reqs_metrics"
    # create dftmp
    data = []
    for name in [
        column for column in df.columns if column.startswith("metrics_")
    ]:
        avg_reqs = float(
            df[name].apply(lambda traces: np.sum(traces[:, 0])).mean()
        )
        avg_qlt = df[name].apply(lambda traces: np.sum(traces[:, 1])).mean()
        avg_lost = df[name].apply(lambda traces: np.sum(traces[:, 2])).mean()
        score = avg_qlt / avg_lost
        data.append(
            (name.removeprefix("metrics_"), avg_reqs, avg_qlt, avg_lost, score)
        )
    assert len(data) > 0
    columns = ["tileset", "avg_reqs", "avg_qlt", "avg_lost", "score"]
    dftmp = pd.DataFrame(data, columns=columns)
    # show dftmp
    fig = make_subplots(
        rows=4, cols=1, subplot_titles=columns[1:], shared_yaxes=True
    )
    trace = go.Bar(y=dftmp["tileset"], x=dftmp["avg_reqs"], orientation="h")
    fig.add_trace(trace, row=1, col=1)
    trace = go.Bar(y=dftmp["tileset"], x=dftmp["avg_lost"], orientation="h")
    fig.add_trace(trace, row=2, col=1)
    trace = go.Bar(y=dftmp["tileset"], x=dftmp["avg_qlt"], orientation="h")
    fig.add_trace(trace, row=3, col=1)
    trace = go.Bar(y=dftmp["tileset"], x=dftmp["score"], orientation="h")
    fig.add_trace(trace, row=4, col=1)
    fig.update_layout(
        width=500,
        showlegend=False,
        barmode="stack",
    )
    show_or_save(fig, savedir=DEFAULT_SAVEDIR, title="tileset_metrics")

if __name__ == "__main__":
    tileset_variations = [
        TileSet(4, 6, TileCover.CENTER),
        TileSet(4, 6, TileCover.ANY),
        TileSet(4, 6, TileCover.ONLY20PERC),
        TileSetVoro(14, TileCover.CENTER),
        TileSetVoro(14, TileCover.ANY),
        TileSetVoro(14, TileCover.ONLY20PERC),
        TileSetVoro(24, TileCover.CENTER),
        TileSetVoro(24, TileCover.ANY),
        TileSetVoro(24, TileCover.ONLY20PERC)
    ]
    df = Dataset(dataset_name="david").df[:2]
    calc_tileset_reqs_metrics(df, tileset_variations)
    show_tileset_reqs_metrics(df, tileset_variations)