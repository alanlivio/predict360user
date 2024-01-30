import numpy as np
import pandas as pd
import plotly.express as px
import sys
import logging
import IPython
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
import datetime
from os.path import abspath, isabs, join
import plotly.express as px
import plotly.graph_objs as go

from predict360user.data_ingestion import get_class_name, get_class_thresholds, DEFAULT_SAVEDIR
from predict360user.utils.plot360 import Plot360
from predict360user.utils.tileset360 import TILESET_DEFAULT

ENTROPY_CLASS_COLORS = {"low": "blue", "medium": "green", "high": "red"}

log = logging.getLogger()

def show_or_save(output, savedir=DEFAULT_SAVEDIR, title="") -> None:
    if "ipykernel" in sys.modules:
        IPython.display.display(output)
    else:
        if not title:
            if isinstance(output, go.Figure):
                title = output.layout.title.text
            else:
                title = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        html_file = join(savedir, title + ".html")
        if isinstance(output, go.Figure):
            output.write_html(html_file)
        else:
            output.to_html(html_file)
        if not isabs(html_file):
            html_file = abspath(html_file)
        log.info(f"compare_train saved on {html_file}")

def calc_traces_hmps(df: pd.DataFrame) -> None:
    df.drop(["traces_hmps"], axis=1, errors="ignore", inplace=True)

    def _calc_traject_hmp(traces) -> np.array:
        return np.apply_along_axis(TILESET_DEFAULT.request, 1, traces)

    tqdm.pandas(desc=f"calc traces_hmps")
    np_hmps = df["traces"].progress_apply(_calc_traject_hmp)
    df["traces_hmps"] = pd.Series(np_hmps)
    assert not df["traces_hmps"].isnull().any()


def calc_traces_poles_prc(df: pd.DataFrame) -> None:
    def _calc_poles_prc(traces) -> float:
        return np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)

    df.drop(["poles_prc", "poles_prc_c"], axis=1, errors="ignore", inplace=True)
    tqdm.pandas(desc=f"calc poles_prc")
    df["poles_prc"] = pd.Series(
        df["traces"].progress_apply(_calc_poles_prc).astype(float)
    )
    threshold_medium, threshold_high = get_class_thresholds(df, "poles_prc")
    df["poles_prc_c"] = (
        df["poles_prc"]
        .apply(get_class_name, args=(threshold_medium, threshold_high))
        .astype("string")
    )
    assert not df["poles_prc_c"].isna().any()


def show_traject(row: pd.Series, title=None) -> None:
    assert "traces" in row.index
    traces = row["traces"]
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]]
    )

    # add traces
    plot = Plot360()
    plot.add_traces(traces)
    for d in plot.data:  # load all data from the df: pd.DataFrame
        fig.append_trace(d, row=1, col=1)

    # add hmps_sum
    traces_hmps = np.apply_along_axis(TILESET_DEFAULT.request, 1, row["traces"])
    hmps_sum = np.sum(traces_hmps, axis=0)
    x = [str(x) for x in range(1, hmps_sum.shape[1] + 1)]
    y = [str(y) for y in range(1, hmps_sum.shape[0] + 1)]
    erp_heatmap = px.imshow(hmps_sum, text_auto=True, x=x, y=y)
    erp_heatmap.update_layout(width=100, height=100)

    # show fig
    fig.append_trace(erp_heatmap.data[0], row=1, col=2)
    # fix given phi 0 being the north pole at cartesian_to_eulerian
    fig.update_yaxes(autorange="reversed")
    title = title if title else f"trajec_{row.name}_[{TILESET_DEFAULT.prefix}]"
    fig.update_layout(width=800, showlegend=False, title_text=title)
    fig.show()


def show_trajects_representative(df: pd.DataFrame) -> None:
    show_traject(
        df[df["actS_c"] == "low"].nlargest(1, "actS").iloc[0], "trajec top low"
    )
    show_traject(
        df[df["actS_c"] == "medium"].nlargest(1, "actS").iloc[0], "trajec top medium"
    )
    show_traject(
        df[df["actS_c"] == "high"].nlargest(1, "actS").iloc[0], "trajec top hight"
    )
    
# def show_trajects_representative_predictions(df: pd.DataFrame) -> None:
    # visualize some predictions
    # # Case "most low with low"
    # one_traject_low = ds.df[ds.df['actS_c'] == 'low'].nsmallest(1,'actS')
    # plot = Plot360()
    # plot.add_traces(one_traject_low['traces'].iloc[0])
    # plot.add_predictions(one_traject_low['pos_only_low_entropy'].iloc[0])
    # plot.show()

    # # Case "most low with low"
    # one_traject_low = ds.df[ds.df['actS_c'] == 'low'].nsmallest(1,'actS')
    # plot = Plot360()
    # plot.add_traces(one_traject_low['traces'].iloc[0])
    # plot.add_predictions(one_traject_low['pos_only'].iloc[0])
    # plot.show()

    # # Case "most medium with medium"
    # one_traject_medium = ds.df[ds.df['actS_c'] == 'medium'].nsmallest(1,'actS')
    # plot = Plot360()
    # plot.add_traces(one_traject_medium['traces'].iloc[0])
    # plot.add_predictions(one_traject_medium['pos_only_medium_entropy'].iloc[0])
    # plot.show()

    # # Case "most medium with all"
    # one_traject_medium = ds.df[ds.df['actS_c'] == 'medium'].nsmallest(1,'actS')
    # plot = Plot360()
    # plot.add_traces(one_traject_medium['traces'].iloc[0])
    # plot.add_predictions(one_traject_medium['pos_only'].iloc[0])
    # plot.show()

    # # Case "most high with high"
    # one_traject_high = ds.df[ds.df['actS_c'] == 'high'].nlargest(1,'actS')
    # plot = Plot360()
    # plot.add_traces(one_traject_high['traces'].iloc[0])
    # plot.add_predictions(one_traject_high['pos_only_high_entropy'].iloc[0])
    # plot.show()

    # # Case "most high with all
    # one_traject_high = ds.df[ds.df['actS_c'] == 'high'].nlargest(1,'actS')
    # plot = Plot360()
    # plot.add_traces(one_traject_medium['traces'].iloc[0])
    # plot.add_predictions(one_traject_medium['pos_only'].iloc[0])
    # plot.show()

def show_entropy_histogram(df: pd.DataFrame) -> None:
    assert "actS" in df.columns
    px.histogram(
        df.dropna(),
        x="actS",
        color="actS_c",
        color_discrete_map=ENTROPY_CLASS_COLORS,
        width=900,
        category_orders={"actS": ["low", "medium", "high"]},
    ).show()


def show_entropy_histogram_per_partition(df: pd.DataFrame) -> None:
    assert "partition" in df.columns
    px.histogram(
        df.dropna(),
        x="actS",
        color="actS_c",
        facet_col="partition",
        color_discrete_map=ENTROPY_CLASS_COLORS,
        category_orders={
            "actS": ["low", "medium", "high"],
            "partition": ["train", "val", "test"],
        },
        width=900,
    ).show()
