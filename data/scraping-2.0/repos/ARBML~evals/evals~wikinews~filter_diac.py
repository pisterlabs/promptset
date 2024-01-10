import os
import json
import openai
import numpy as np
from tqdm import tqdm
from pyarabic.araby import strip_tashkeel
from torch.utils.data import Dataset, DataLoader
from wikinews import read_file, read_json, write_file

DOMAIN = "sports"

if __name__ == "__main__":

    dirpath = "/Users/bkhmsi/Desktop/WikiNews"
    pred_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.0.pred")
    grnd_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.0.grnd")
    usage_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.0.usage.json")

    usage = read_json(usage_path)
    preds = read_file(pred_path)
    grndt = read_file(grnd_path)

    filtered_preds = []
    filtered_grndt = []

    for i, row in enumerate(usage):
        if row["completion_tokens"] >= 512:
            print(i)
        else:
            filtered_preds += [preds[i]]
            filtered_grndt += [grndt[i]]

    fpred_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.0.fpred")
    fgrnd_path = os.path.join(dirpath, f"WikiNews.{DOMAIN}.0.fgrnd")

    write_file(fpred_path, filtered_preds)
    write_file(fgrnd_path, filtered_grndt)