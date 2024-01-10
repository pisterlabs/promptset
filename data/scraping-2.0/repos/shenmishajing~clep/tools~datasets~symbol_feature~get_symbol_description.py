import json
import os
import pickle
from time import sleep

import pandas as pd
import torch
from openai import InvalidRequestError
from tqdm import tqdm

from datasets.tianchi.tianchi import TianChiDataset
from utils import openai


def get_llm_results(messages, model="gpt-3.5-turbo", temperature=0):
    try_num = 0
    while True:
        try:
            return openai.ChatCompletion.create(
                messages=messages, model=model, temperature=temperature
            )
        except InvalidRequestError as e:
            if e.code == "context_length_exceeded":
                if model == "gpt-3.5-turbo":
                    model = "gpt-3.5-turbo-16k"
                    continue
                else:
                    print(f"context length exceeded, skip, error: {e}")
            else:
                raise e
        except Exception as e:
            try_num += 1
            print(f"error: {e}, try num: {try_num}, retry after {try_num} min")
            sleep(try_num * 60)


def get_symbol_raw_description_per_wave(data_root, leads, symbols):
    if not os.path.exists(os.path.join(data_root, "symbol_description_raw.pkl")):
        print("get symbol description raw")
        data = {}
        for lead in tqdm(leads, desc="lead", position=0, leave=False):
            data[lead] = {}
            for symbol, disease in tqdm(
                symbols.items(), desc="symbol", position=1, leave=False
            ):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"In ecg, what does a {disease} look like in lead {lead}? Describe it in P wave, QRS wave and T wave respectively, starting with '<wave_name> wave: '. For example, describe P wave starting with P wave: . If there is no special characteristic for some wave in such a lead, just say None.",
                    },
                ]
                data[lead][symbol] = get_llm_results(messages)

        pickle.dump(
            data, open(os.path.join(data_root, "symbol_description_raw.pkl"), "wb")
        )


def get_symbol_description_per_wave(data_root):
    print("get symbol description")
    data = pickle.load(
        open(os.path.join(data_root, "symbol_description_raw.pkl"), "rb")
    )
    for lead in tqdm(data, desc="lead", position=0, leave=False):
        for symbol in tqdm(data[lead], desc="symbol", position=1, leave=False):
            data[lead][symbol] = data[lead][symbol]["choices"][0]["message"]["content"]

    json.dump(
        data,
        open(os.path.join(data_root, "symbol_description.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )


def get_symbol_sentence_per_wave(data_root):
    print("get symbol sentence")
    data = json.load(open(os.path.join(data_root, "symbol_description.json")))
    for lead in tqdm(data, desc="lead", position=0, leave=False):
        for symbol in tqdm(data[lead], desc="symbol", position=1, leave=False):
            res = {}
            sentences = data[lead][symbol].split("\n")
            for wave in ["P", "QRS", "T"]:
                for sentence in sentences:
                    ind = sentence.find(f"{wave} wave: ")
                    if ind != -1:
                        res[wave] = sentence[ind + len(f"{wave} wave: ") :]
                        break
                if wave in res and "None" in res[wave]:
                    res.pop(wave)
            if "QRS" in res:
                res["R"] = res.pop("QRS")
            data[lead][symbol] = res

    json.dump(
        data,
        open(os.path.join(data_root, "symbol_sentence.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    try_num = 0
    while True:
        try:
            return openai.Embedding.create(input=[text], model=model)
        except Exception as e:
            try_num += 1
            print(f"error: {e}, try num: {try_num}, retry after {try_num+10} min")
            sleep((try_num + 10) * 60)


def get_symbol_raw_embedding_per_wave(data_root):
    if not os.path.exists(os.path.join(data_root, "symbol_embedding_raw.pkl")):
        print("get symbol raw embedding")
        data = json.load(open(os.path.join(data_root, "symbol_sentence.json")))
        for lead in tqdm(data, desc="lead", position=0, leave=False):
            for symbol in tqdm(data[lead], desc="symbol", position=1, leave=False):
                for wave in tqdm(
                    data[lead][symbol], desc="wave", position=2, leave=False
                ):
                    data[lead][symbol][wave] = get_embedding(data[lead][symbol][wave])

        pickle.dump(
            data, open(os.path.join(data_root, "symbol_embedding_raw.pkl"), "wb")
        )


def get_symbol_embedding_per_wave(data_root):
    print("get symbol embedding")
    data = pickle.load(open(os.path.join(data_root, "symbol_embedding_raw.pkl"), "rb"))

    for lead in tqdm(data, desc="lead", position=0, leave=False):
        for symbol in tqdm(data[lead], desc="symbol", position=1, leave=False):
            for wave in tqdm(data[lead][symbol], desc="wave", position=2, leave=False):
                data[lead][symbol][wave] = torch.tensor(
                    data[lead][symbol][wave]["data"][0]["embedding"]
                )

    pickle.dump(data, open(os.path.join(data_root, "symbol_embedding.pkl"), "wb"))


def get_symbol_per_wave(info):
    get_symbol_raw_description_per_wave(
        info["data_root"], info["leads"], info["symbols"]
    )
    get_symbol_description_per_wave(info["data_root"])
    get_symbol_sentence_per_wave(info["data_root"])
    get_symbol_raw_embedding_per_wave(info["data_root"])
    get_symbol_embedding_per_wave(info["data_root"])


def get_symbol_raw_description(data_root, symbols):
    if not os.path.exists(os.path.join(data_root, "symbol_description_raw.pkl")):
        print("get symbol description raw")
        data = {}
        for symbol, disease in tqdm(symbols.items(), desc="symbol"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"What does an ecg look like for a patient with {disease} simultaneously? Please describe it in all relevant aspects, including but not limited to P wave, QRS wave, T wave, hearte rate and all other related aspects, respectively. Answer this in English.",
                },
            ]
            data[symbol] = get_llm_results(messages)

        pickle.dump(
            data, open(os.path.join(data_root, "symbol_description_raw.pkl"), "wb")
        )


def get_symbol_description(data_root):
    print("get symbol description")
    data = pickle.load(
        open(os.path.join(data_root, "symbol_description_raw.pkl"), "rb")
    )
    for symbol in tqdm(data, desc="symbol"):
        data[symbol] = (
            data[symbol]["choices"][0]["message"]["content"]
            + f" This ecg contains disease {symbol}."
        )

    json.dump(
        data,
        open(os.path.join(data_root, "symbol_description.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )


def get_symbol(info):
    get_symbol_raw_description(info["data_root"], info["symbols"])
    get_symbol_description(info["data_root"])


def main():
    data_info = {
        "mit-bih_per_wave": {
            "data_root": "data/mit-bih-arrhythmia-database-1.0.0/symbols_chatgpt_per_wave",
            "leads": ["MLII", "V1", "V2", "V4", "V5"],
            "symbols": {
                "N": "Normal beat",
                "SVEB": "Supraventricular ectopic beat",
                "VEB": "Ventricular ectopic beat",
                "F": "Fusion beat",
            },
            # "symbols": {
            #     # N
            #     "N": "Normal beat",
            #     "L": "Left bundle branch block beat",
            #     "R": "Right bundle branch block beat",
            #     "e": "Atrial escape beat",
            #     "j": "Nodal (junctional) escape beat",
            #     # SVEB
            #     "A": "Atrial premature beat",
            #     "a": "Aberrated atrial premature beat",
            #     "J": "Nodal (junctional) premature beat",
            #     "S": "Supraventricular premature",
            #     # VE
            #     "V": "Premature ventricular contraction",
            #     "E": "Ventricular escape beat",
            #     # F
            #     "F": "Fusion of ventricular and normal beat",
            #     # Q
            #     "f": "Fusion of paced and normal beat",
            #     # "Q": "Unclassifiable beat",
            #     # # other
            #     # "[": "Start of ventricular flutter/fibrillation",
            #     # "!": "Ventricular flutter wave",
            #     # "]": "End of ventricular flutter/fibrillation",
            #     # "/": "Paced beat",
            #     # "x": "Non-conducted P-wave (blocked APC)",
            #     # "|": "Isolated QRS-like artifact",
            # },
        },
        "tianchi_per_wave": {
            "data_root": "data/tianchi/symbols_chatgpt_per_wave",
            "leads": TianChiDataset.SignalNames,
            "symbols": None,
        },
        "tianchi": {
            "data_root": "data/tianchi/symbols_chatgpt",
            "leads": TianChiDataset.SignalNames,
            "symbols": None,
            "func": get_symbol,
        },
    }

    data = pd.read_csv("data/tianchi/ann/round_all/all.csv")

    data_info["tianchi_per_wave"]["symbols"] = data_info["tianchi"]["symbols"] = {
        d: d.replace("_", ", ") for d in set(data["disease"])
    }

    data = [
        ("ST段改变_一度房室传导阻滞_二联律_室性早搏_左心室肥大_窦性心律", 2059),
        ("QRS低电压_T波改变_室性早搏_心房颤动_快心室率_起搏心律", 1372),
        ("QRS低电压_T波改变_一度房室传导阻滞_房性早搏_窦性心律", 1287),
        ("右束支传导阻滞_完全性右束支传导阻滞_左前分支传导阻滞_房性早搏_窦性心动过速", 903),
        ("右束支传导阻滞_完全性右束支传导阻滞_左心室高电压_房性早搏_窦性心动过速", 874),
    ]
    data = [d[0] for d in data]
    data_info["tianchi_per_wave"]["symbols"] = data_info["tianchi"]["symbols"] = {
        d: d.replace("_", ", ") for d in data
    }

    for name in data_info:
        data_info[name].setdefault("func", get_symbol_per_wave)

    names = ["tianchi"]

    if names is None:
        names = data_info
    for name in names:
        info = data_info[name]
        print(f"process {name}")

        os.makedirs(info["data_root"], exist_ok=True)
        info["func"](info)


if __name__ == "__main__":
    main()
