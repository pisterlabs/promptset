import os
from pathlib import Path
import shutil
import pandas as pd
import json
import numpy as np
import soundfile as sf
import librosa
import cv2
import torch
import openai

from constants import *

def clear_folder_content(folder_path: Path, including_folder: bool = False):
    if not os.path.exists(folder_path):
        return
    print("Deleting folder: ", folder_path)
    for file_path in folder_path.iterdir():
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    if including_folder:
        ensure_dir(folder_path)
        os.rmdir(folder_path)
    print("Folder deleted: ", folder_path)


def ensure_dir(folder_path: Path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_to_file(to_save, file_path: Path, additional_info=None):
    ensure_dir(file_path.parent)
    if type(to_save) == pd.DataFrame:
        file_path = file_path if file_path.name.endswith(".csv") else f"{file_path}.csv"
        to_save.to_csv(file_path)
    elif type(to_save) == dict:
        file_path = file_path if file_path.name.endswith(".txt") else f"{file_path}.txt"
        with open(file_path, 'w') as file:
            file.write(json.dumps(to_save, cls=NpEncoder))
    elif type(to_save) == np.ndarray and additional_info["type"] == "audio":
        sf.write(file_path, to_save, additional_info["sample_rate"])
    elif type(to_save) == torch.Tensor:
        torch.save(to_save, file_path)
    elif type(to_save) == np.ndarray and additional_info["type"] == "image":
        file_path = file_path.as_posix() if file_path.name.endswith(".png") else f"{file_path}.png"
        cv2.imwrite(file_path, to_save)
    else:
        raise NotImplementedError(f"Saving of this type: {type(to_save)} isn't implemented yet. File: {file_path} .")


def load_file(file_path: Path):
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path, index_col=0)
    elif file_path.suffix == ".txt":
        with open(file_path) as f:
            return json.load(f)
    elif file_path.suffix == ".wav":
        data, sample_rate = librosa.load(file_path, sr=None)
        return sample_rate, data
    elif file_path.suffix == ".pt":
        return torch.load(file_path)
    elif file_path.suffix == ".png":
        return cv2.imread(file_path.as_posix())
    else:
        raise NotImplementedError(f"Reading files of this object type isn't implemented yet. File: {file_path} .")


def ensure_open_ai_api():
    if not os.environ.get("OPENAI_API_KEY"):
        if not os.path.exists(OTHER_FOLDER / "openai_api_key.txt"):
            api_key = input("Please enter your openai api key: ")
            with open(OTHER_FOLDER / "openai_api_key.txt", "w") as file:
                file.write(api_key)

        with open(OTHER_FOLDER / "openai_api_key.txt") as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()
    openai.api_key = os.environ["OPENAI_API_KEY"]


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    resampled_data = librosa.resample(y=audio, orig_sr=original_rate, target_sr=target_rate, res_type="linear",
                                      fix=True,
                                      scale=False)
    return resampled_data
