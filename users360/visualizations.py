from .data import *
from .tileset import *
from .projection import *
import plotly.graph_objs as go
import plotly.express as px
import plotly
from plotly.subplots import make_subplots

class_color_map = {"hight": "red", "medium": "green", "low": "blue"}


def show_fov(trace, tileset=TileSet.default(), to_html=False):
    assert len(trace) == 3  # cartesian

    # default Projection
    project = Projection(tileset)
    project.add_trace(trace)
    project.add_vp(trace)

    # erp heatmap
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])
    for t in project.data:
        fig.append_trace(t, row=1, col=1)

    heatmap = tileset.request(trace)
    if isinstance(tileset, TileSetVoro):
        heatmap = np.reshape(heatmap, tileset.shape)
    erp_heatmap = px.imshow(heatmap, text_auto=True,
                            x=[str(x) for x in range(1, heatmap.shape[1] + 1)],
                            y=[str(y) for y in range(1, heatmap.shape[0] + 1)])
    for t in erp_heatmap["data"]:
        fig.append_trace(t, row=1, col=2)

    title = f"trace_[{trace[0]:.2},{trace[1]:.2},{trace[2]:.2}] {tileset.str_hmp_sum([heatmap])}"
    if isinstance(tileset, TileSet):
        # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
        fig.update_yaxes(autorange="reversed")
    if to_html:
        output_folder = pathlib.Path(__file__).parent.parent / 'output'
        plotly.offline.plot(fig, filename=f'{output_folder}/{title}.html', auto_open=False)
    else:
        fig.update_layout(width=800, showlegend=False, title_text=title)
        fig.show()


def show_trajects(df: pd.DataFrame, tileset=TileSet.default(), to_html=False):
    assert('hmps' in df)

    # subplot two figures https://stackoverflow.com/questions/67291178/how-to-create-subplots-using-plotly-express
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "image"}]])

    # default Projection
    project = Projection(tileset)
    data = project.data

    # add traces to row=1,col=1
    def f_traces(traces):
        scatter = go.Scatter3d(x=traces[:, 0], y=traces[:, 1], z=traces[:, 2],
                               mode='lines', line={'width': 3, 'color': 'blue'}, showlegend=False)
        data.append(scatter)
    df['traces'].apply(f_traces)
    for trace in data:
        fig.append_trace(trace, row=1, col=1)

    # add erp_heatmap row=1, col=2
    hmp_sums = df['hmps'].apply(lambda traces: np.sum(traces, axis=0))
    if isinstance(tileset, TileSetVoro):
        hmp_sums = np.reshape(hmp_sums, tileset.shape)
    hmp_final = np.sum(hmp_sums, axis=0)
    erp_heatmap = px.imshow(hmp_final, text_auto=True,
                            x=[str(x) for x in range(1, hmp_final.shape[1] + 1)],
                            y=[str(y) for y in range(1, hmp_final.shape[0] + 1)])
    for data in erp_heatmap["data"]:
        fig.append_trace(data, row=1, col=2)

    title = f"trajects_{str(df.shape[0])} {tileset.str_hmp_sum(hmp_final)}"

    if isinstance(tileset, TileSet):
        # fix given phi 0 being the north pole at Utils.cartesian_to_eulerian
        fig.update_yaxes(autorange="reversed")
    if to_html:
        output_folder = pathlib.Path(__file__).parent.parent / 'output'
        plotly.offline.plot(fig, filename=f'{output_folder}/{title}.html', auto_open=False)
    else:
        fig.update_layout(width=800, showlegend=False, title_text=title)
        fig.show()


def show_poles_prc():
    df = Data.singleton().df_trajects
    assert(not df.empty)
    if not 'poles_prc' in df.columns:
        calc_poles_prc()
    px.scatter(df, x='user', y='poles_prc', color='poles_class',
               color_discrete_map=class_color_map,
               hover_data=[df.index], title='trajects poles_perc', width=600).show()


def show_entropy():
    df = Data.singleton().df_trajects
    if not 'entropy' in df.columns:
        calc_entropy()

    px.scatter(df, x='user', y='entropy', color='entropy_class',
               color_discrete_map=class_color_map, hover_data=[df.index],
               title='trajects entropy', width=600).show()


def show_entropy_users():
    df_users = Data.singleton().df_users
    if not 'entropy' in df_users.columns:
        calc_entropy_users()
    assert('entropy_class' in df_users.columns)
    px.scatter(df_users, x='user', y='entropy', color='entropy_class',
               color_discrete_map=class_color_map, title='users entropy', width=600).show()


def show_tileset_metrics():
    assert(Data.singleton().df_tileset_metrics)
    df_tileset_metrics = Data.singleton().df_tileset_metrics
    fig = make_subplots(rows=1, cols=4, subplot_titles=("avg_reqs", "avg_lost",
                        "avg_qlt", "score=avg_qlt/avg_lost"), shared_yaxes=True)
    y = df_tileset_metrics['scheme']
    trace = go.Bar(y=y, x=df_tileset_metrics['avg_reqs'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=1)
    trace = go.Bar(y=y, x=df_tileset_metrics['avg_lost'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=2)
    trace = go.Bar(y=y, x=df_tileset_metrics['avg_qlt'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=3)
    trace = go.Bar(y=y, x=df_tileset_metrics['score'], orientation='h', width=0.3)
    fig.add_trace(trace, row=1, col=4)
    fig.update_layout(width=1500, showlegend=False, barmode="stack",)
    fig.show()
