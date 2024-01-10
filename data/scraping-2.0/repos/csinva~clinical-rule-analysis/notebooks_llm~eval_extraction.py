import pathlib
import re
from typing import Dict, List
import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import os.path
from tqdm import tqdm
import json
import os
import numpy as np
import openai
from os.path import dirname

path_to_file = dirname(__file__)
path_to_repo = dirname(path_to_file)
papers_dir = join(path_to_repo, "papers")


def compute_metrics_within_1(
    df,
    preds_col_to_gt_col_dict={
        "num_male": "num_male_corrected",
        "num_female": "num_female_corrected",
        "num_total": "num_total_corrected",
    },
) -> pd.DataFrame:
    d = defaultdict(list)
    one_perc = (df["participants___total"].astype(float) / 100).apply(np.ceil)
    for k in df.columns:
        # if k.startswith('num_') and k + '_corrected' in df.columns:

        # print(one_perc)
        if k in preds_col_to_gt_col_dict:
            gt_col = preds_col_to_gt_col_dict[k]
            # print(df.columns, gt_col)
            idxs_with_labels = df[gt_col].notnull() & ~(df[gt_col].isin({-1}))
            gt = df[gt_col][idxs_with_labels].astype(int)
            pred = df[k].apply(cast_int)[idxs_with_labels].astype(int)
            pred = pred.apply(lambda x: x if x >= 0 else np.nan)
            # print('preds', (pred >= 0).sum())
            # print('gt', gt)

            d["target"].append(gt_col)
            d["n_gt"].append(len(gt))
            # print(df[k])
            # d['n_pred'].append(df[k].notna().sum())
            d["n_pred"].append((pred.notna() & (pred >= 0)).sum())
            # print((gt - pred).values.tolist())
            # d["n_correct_within_1"].append((np.abs(gt - pred) <= 1).sum())
            d["n_correct_1_perc"].append(
                (np.abs(gt - pred) <= one_perc[idxs_with_labels]).sum()
            )
            # d['n_predicted'].append(df[k].notnull().sum())
            # count number of values which contain a number
    metrics = pd.DataFrame.from_dict(d)
    metrics["recall"] = metrics["n_correct_1_perc"] / metrics["n_gt"]
    metrics["precision"] = metrics["n_correct_1_perc"] / metrics["n_pred"]

    return metrics.round(2)


def convert_percentages_when_total_is_known(num, tot):
    if tot is not None and isinstance(tot, str):
        tot = tot.replace(",", "").replace(" ", "")
    if (
        str_contains_number(num)
        and str_is_percentage(num)
        and str_contains_number(tot)
        and not str_is_percentage(tot)
    ):
        num = percentage_to_num(num)
        tot = int(tot)
        num = round(num * tot / 100)
    return num


def cast_int(x):
    try:
        return int(x)
    except:
        return -1


def int_or_empty(x):
    try:
        return int(x)
    except:
        return ""


def int_or_neg1(x):
    try:
        return int(x)
    except:
        return -1


def str_is_parsable(x):
    """Check that string only contains numbers, percent, or periods"""
    return x is not None and all(
        char.isdigit() or char in [".", "%", " ", ","] for char in str(x)
    )


def str_contains_number(x):
    return (
        x is not None
        and any(char.isdigit() for char in str(x))
        and not any(char.isalpha() for char in str(x))
    )


def str_is_percentage(s):
    return "%" in s or "." in s


def percentage_to_num(s):
    if "%" in s:
        s = s.replace("%", "")
    return float(s)
