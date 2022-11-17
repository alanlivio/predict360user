from os.path import exists

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats
import swifter
from plotly.subplots import make_subplots

from .data import *
from .utils.tileset import *

_CLASS_COR = {"hight": "red", "medium": "green", "low": "blue"}


def calc_trajects_hmps(tileset=TILESET_DEFAULT) -> None:
    df = Data.instance().df_trajects
    f_trace = lambda trace: tileset.request(trace)
    f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
    logging.info("calculating heatmaps ...")
    df['hmps'] = pd.Series(df['traces'].swifter.apply(f_traject))


def calc_trajects_entropy() -> None:
    df = Data.instance().df_trajects
    if 'hmps' not in df.columns:
        calc_trajects_hmps()
    # calc entropy
    f_entropy = lambda x: scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
    logging.info("calculating trajects entropy ...")
    df['entropy'] = df['hmps'].swifter.apply(f_entropy)
    # calc class
    idxs_sort = df['entropy'].argsort()
    trajects_len = len(df['entropy'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df['entropy'][idx_threshold_medium]
    threshold_hight = df['entropy'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df['entropy_class'] = df['entropy'].apply(f_threshold)


def calc_trajects_entropy_users() -> None:
    df = Data.instance().df_trajects
    # df_users = Data.instance().df_users
    if 'hmps' not in df.columns:
        calc_trajects_hmps()
    # calc entropy
    logging.info("calculating users entropy ...")
    def f_entropy_user(same_user_rows) -> np.ndarray:
        same_user_rows_np = same_user_rows['hmps'].to_numpy()
        hmps_sum = sum(np.sum(x, axis=0) for x in same_user_rows_np)
        entropy = scipy.stats.entropy(hmps_sum.reshape((-1)))
        return entropy
    df_users = df.groupby(['ds_user']).apply(f_entropy_user)
    df_users = df_users.reset_index()
    df_users.columns = ['ds_user', 'entropy']
    assert not df_users['entropy'].isna().all()
    # calc class
    idxs_sort = df_users['entropy'].argsort()
    trajects_len = len(df_users['entropy'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df_users['entropy'][idx_threshold_medium]
    threshold_hight = df_users['entropy'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df_users['entropy_class'] = df_users['entropy'].apply(f_threshold)
    Data.instance().df_users = df_users


def calc_trajects_poles_prc() -> None:
    df = Data.instance().df_trajects
    f_traject = lambda traces: np.count_nonzero(abs(traces[:, 2]) > 0.7) / len(traces)
    df['poles_prc'] = pd.Series(df['traces'].apply(f_traject))
    idxs_sort = df['poles_prc'].argsort()
    trajects_len = len(df['poles_prc'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df['poles_prc'][idx_threshold_medium]
    threshold_hight = df['poles_prc'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df['poles_class'] = df['poles_prc'].apply(f_threshold)


def show_trajects_poles_prc() -> None:
    df = Data.instance().df_trajects
    assert (not df.empty)
    if not 'poles_prc' in df.columns:
        calc_trajects_poles_prc()
    fig = px.scatter(df, y='ds_user', x='poles_prc', color='poles_class',
                     color_discrete_map=_CLASS_COR,
                     hover_data=[df.index], title='trajects poles_perc', width=600)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker_size=2)
    fig.show()


def show_trajects_entropy() -> None:
    df = Data.instance().df_trajects
    assert (not df.empty)
    if not 'entropy' in df.columns:
        calc_trajects_entropy()
    df = Data.instance().df_trajects
    fig = px.scatter(df, y='ds_user', x='entropy', color='entropy_class',
                     color_discrete_map=_CLASS_COR,
                     title='trajects entropy', width=600)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker_size=2)
    fig.show()


def show_trajects_entropy_users() -> None:
    df_users = Data.instance().df_users
    assert (not df_users.empty)
    if (not 'entropy' in df_users.columns) or (df_users['entropy'].isna().all()):
        calc_trajects_entropy_users()
    assert ('entropy_class' in df_users.columns)
    fig = px.scatter(df_users, y='ds_user', x='entropy', color='entropy_class',
                     color_discrete_map=_CLASS_COR, title='users entropy', width=600)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker_size=2)
    fig.show()


def calc_tileset_reqs_metrics(tileset_l: list[TileSetIF], n_trajects=None) -> None:
    assert (not get_df_trajects().empty)
    if n_trajects:
        df = get_df_trajects()[:n_trajects]
    else:
        df = get_df_trajects()

    def create_tsdf(ts_idx) -> pd.DataFrame:
        tileset = tileset_l[ts_idx]
        def f_trace(trace) -> tuple[int, float, float]:
            heatmap, vp_quality, area_out = tileset.request(trace, return_metrics=True)
            return (int(np.sum(heatmap)), vp_quality, area_out)
        f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
        tmpdf = pd.DataFrame(df['traces'].swifter.apply(f_traject))
        tmpdf.columns = [tileset.title]
        return tmpdf
    df_tileset_metrics = pd.concat(map(create_tsdf, range(len(tileset_l))), axis=1)
    Data.instance().df_tileset_metrics = df_tileset_metrics


def show_tileset_reqs_metrics() -> None:
    df_tileset_metrics = Data.instance().df_tileset_metrics
    assert (not df_tileset_metrics.empty)

    # calc tileset metrics
    f_traject_reqs = lambda traces: np.sum(traces[:, 0])
    f_traject_qlt = lambda traces: np.mean(traces[:, 1])
    f_traject_lost = lambda traces: np.mean(traces[:, 2])
    data = {'tileset': [], 'avg_reqs': [], 'avg_qlt': [], 'avg_lost': []}
    for name in df_tileset_metrics.columns:
        dfts = df_tileset_metrics[name]
        data['tileset'].append(name)
        data['avg_reqs'].append(dfts.apply(f_traject_reqs).mean())
        data['avg_qlt'].append(dfts.apply(f_traject_qlt).mean())
        data['avg_lost'].append(dfts.apply(f_traject_lost).mean())
        data['score'] = data['avg_qlt'][-1] / data['avg_lost'][-1]
    df = pd.DataFrame(data)

    fig = make_subplots(rows=1, cols=4, subplot_titles=("avg_reqs", "avg_lost",
                        "avg_qlt", "score=avg_qlt/avg_lost"), shared_yaxes=True)
    y = df_tileset_metrics.columns
    trace = go.Bar(y=y, x=df['avg_reqs'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=1)
    trace = go.Bar(y=y, x=df['avg_lost'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=2)
    trace = go.Bar(y=y, x=df['avg_qlt'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=3)
    trace = go.Bar(y=y, x=df['score'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=4)
    fig.update_layout(width=1500, showlegend=False, barmode="stack",)
    fig.show()
