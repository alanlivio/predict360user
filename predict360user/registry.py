import os
from os.path import isdir, join

import pandas as pd
import plotly.express as px

from predict360user.utils.utils import *


def compare_train_results(savedir="saved", model_filter=[]) -> None:
    # find results_csv files
    csv_df_l = [
        (dir_name, pd.read_csv(join(savedir, dir_name, file_name)))
        for dir_name in os.listdir(savedir)
        if isdir(join(savedir, dir_name))
        for file_name in os.listdir(join(savedir, dir_name))
        if file_name == TRAIN_RES_CSV
    ]
    csv_df_l = [df.assign(model_name=dir_name) for (dir_name, df) in csv_df_l]
    assert csv_df_l, f"no <savedir>/<model>/{TRAIN_RES_CSV} files"
    df_compare = pd.concat(csv_df_l, ignore_index=True)
    if model_filter:
        df_compare = df_compare.loc[df_compare["model_name"].isin(model_filter)]

    # plot
    fig = px.line(
        df_compare,
        x="epoch",
        y="loss",
        color="model_name",
        title="compare_loss",
        width=800,
    )
    show_or_save(fig, savedir, "compare_loss")
    fig = px.line(
        df_compare,
        x="epoch",
        y="val_loss",
        color="model_name",
        title="compare_val_loss",
        width=800,
    )
    show_or_save(fig, savedir, "compare_val_loss")


def compare_eval_results(
    savedir="saved", model_filter=[], entropy_filter=[]
) -> None:
    # find results_csv files
    csv_df_l = [
        pd.read_csv(join(savedir, dir_name, file_name))
        for dir_name in os.listdir(savedir)
        if isdir(join(savedir, dir_name))
        for file_name in os.listdir(join(savedir, dir_name))
        if file_name == EVAL_RES_CSV
    ]
    assert csv_df_l, f"no <savedir>/<model>/{EVAL_RES_CSV} files"
    df_compare = pd.concat(csv_df_l, ignore_index=True)

    # create vis table
    t_range = [c for c in df_compare.columns if c.isnumeric()]
    props = "text-decoration: underline"
    if model_filter:
        df_compare = df_compare.loc[df_compare["model_name"].isin(model_filter)]
    if entropy_filter:
        df_compare = df_compare.loc[df_compare["actS_c"].isin(entropy_filter)]
    output = (
        df_compare.sort_values(by=t_range)
        .style.background_gradient(axis=0, cmap="coolwarm")
        .highlight_min(subset=t_range, props=props)
        .highlight_max(subset=t_range, props=props)
    )
    show_or_save(output, savedir, "compare_evaluate")
