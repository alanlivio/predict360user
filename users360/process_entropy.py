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


def calc_trajects_hmps(tileset=TILESET_DEFAULT):
    df = get_df_trajects()
    f_trace = lambda trace: tileset.request(trace)
    f_traject = lambda traces: np.apply_along_axis(f_trace, 1, traces)
    df['hmps'] = pd.Series(df['traces'].swifter.apply(f_traject))


def calc_trajects_entropy():
    df = get_df_trajects()
    if 'hmps' not in df.columns:
        calc_trajects_hmps()
    # calc entropy
    f_entropy = lambda x: scipy.stats.entropy(np.sum(x, axis=0).reshape((-1)))
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


def calc_trajects_entropy_users():
    df = get_df_trajects()
    df_users = Data.singleton().df_users
    if 'hmps' not in df.columns:
        calc_trajects_hmps()
    # calc entropy
    def f_entropy_user(trajects):
        hmps_sum = np.sum(np.sum(trajects['hmps'].to_numpy(), axis=0), axis=0)
        return scipy.stats.entropy(hmps_sum.reshape((-1)))
    df_users['entropy'] = df.groupby(['user']).apply(f_entropy_user)
    # calc class
    idxs_sort = df_users['entropy'].argsort()
    trajects_len = len(df_users['entropy'])
    idx_threshold_medium = idxs_sort[int(trajects_len * .60)]
    idx_threshold_hight = idxs_sort[int(trajects_len * .90)]
    threshold_medium = df_users['entropy'][idx_threshold_medium]
    threshold_hight = df_users['entropy'][idx_threshold_hight]
    f_threshold = lambda x: 'low' if x < threshold_medium else ('medium' if x < threshold_hight else 'hight')
    df_users['entropy_class'] = df_users['entropy'].apply(f_threshold)

def calc_trajects_poles_prc():
    df = get_df_trajects()
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
    

def show_trajects_poles_prc():
    df = get_df_trajects()
    assert (not df.empty)
    if not 'poles_prc' in df.columns:
        calc_trajects_poles_prc()
    px.scatter(df, x='user', y='poles_prc', color='poles_class',
               color_discrete_map=_CLASS_COR,
               hover_data=[df.index], title='trajects poles_perc', width=600).show()

def show_trajects_entropy():
    df = get_df_trajects()
    assert (not df.empty)
    if not 'entropy' in df.columns:
        calc_trajects_entropy()
    px.scatter(df, x='user', y='entropy', color='entropy_class',
               color_discrete_map=_CLASS_COR, hover_data=[df.index],
               title='trajects entropy', width=600).show()


def show_trajects_entropy_users():
    df_users = Data.singleton().df_users
    assert (not df_users.empty)
    if not 'entropy' in df_users.columns:
        calc_trajects_entropy_users()
    assert ('entropy_class' in df_users.columns)
    px.scatter(df_users, x='user', y='entropy', color='entropy_class',
               color_discrete_map=_CLASS_COR, title='users entropy', width=600).show()
