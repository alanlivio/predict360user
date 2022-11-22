from os.path import exists

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

from .data import *
from .entropy import *


def get_train_test_split(train_entropy, test_entropy, perc_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = Data.instance().df_trajects
    if (train_entropy != 'all') and (test_entropy != 'all'):
        assert not df['traject_entropy_class'].empty, "dataset has no 'traject_entropy_class' collumn, run -calc_trajects_entropy"
        X_train, X_test = train_test_split(df, test_size=perc_test, random_state=1)
    else:
        if train_entropy == 'all':
            X_train, _ = train_test_split(df, test_size=perc_test, random_state=1)
        else:
            X_train, _ = train_test_split(df[df['traject_entropy_class'] == train_entropy],
                                          test_size=perc_test, random_state=1)
        if test_entropy == 'all':
            _, X_test = train_test_split(df, test_size=perc_test, random_state=1)
        else:
            _, X_test = train_test_split(df[df['traject_entropy_class'] == test_entropy],
                                         test_size=perc_test, random_state=1)
    return X_train, X_test


def show_train_test_split(train_entropy, test_entropy, perc_test) -> None:
    X_train, X_test = get_train_test_split(train_entropy, test_entropy, perc_test)
    X_train['partition'] = 'train'
    X_test['partition'] = 'test'
    df_trajects = pd.concat([X_train, X_test])

    fig = px.scatter(df_trajects, y='ds', x='traject_entropy',
                     color='traject_entropy_class',
                     facet_col="partition", color_discrete_map=ENTROPY_CLASS_COLORS,
                     title=f'trajects_entropy by ds', width=600)
    fig.update_traces(marker_size=2)
    fig.show()
    fig = px.scatter(df_trajects, y='ds_user', x='traject_entropy',
                     color='traject_entropy_class',
                     facet_col="partition", color_discrete_map=ENTROPY_CLASS_COLORS,
                     title=f'trajects_entropy by ds_user', width=600)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker_size=2)
    fig.show()
    fig = px.scatter(df_trajects, y='ds_video', x='traject_entropy',
                     color='traject_entropy_class',
                     facet_col="partition", color_discrete_map=ENTROPY_CLASS_COLORS,
                     title=f'trajects_entropy by ds_video', width=600)
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker_size=2)
    fig.show()
    fig = px.scatter(df_trajects, y='user_entropy_class', x='traject_entropy',
                     color='traject_entropy_class',
                     facet_col="partition",
                     color_discrete_map=ENTROPY_CLASS_COLORS,
                     title='trajects_entropy by user_entropy_class', width=600)
    fig.update_traces(marker_size=2)
    fig.show()
