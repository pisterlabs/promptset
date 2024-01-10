import json
import os
import pickle
from time import sleep

import torch

from utils import openai


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


def cache_embedding(data_root, symbols, waves="PRT"):
    if os.path.exists(os.path.join(data_root, "symbol_embedding_raw.pkl")):
        data = pickle.load(
            open(os.path.join(data_root, "symbol_embedding_raw.pkl"), "rb")
        )
    else:
        data = json.load(
            open(os.path.join(data_root, "symbol_sentence_processed.json"))
        )
        for lead in data:
            for symbol in data[lead]:
                for wave in data[lead][symbol]:
                    data[lead][symbol][wave] = get_embedding(data[lead][symbol][wave])

        pickle.dump(
            data, open(os.path.join(data_root, "symbol_embedding_raw.pkl"), "wb")
        )

    for lead in data:
        for symbol in data[lead]:
            for wave in data[lead][symbol]:
                data[lead][symbol][wave] = torch.tensor(
                    data[lead][symbol][wave]["data"][0]["embedding"]
                )

    for lead in data:
        if lead == "common":
            continue
        for symbol in data[lead]:
            for wave in waves:
                if wave not in data[lead][symbol]:
                    data[lead][symbol][wave] = data["common"][symbol][wave]
            data[lead][symbol] = torch.cat([data[lead][symbol][w] for w in waves])
        data[lead] = torch.stack([data[lead][symbol] for symbol in symbols])

    data.pop("common", None)

    pickle.dump(
        data, open(os.path.join(data_root, "symbol_embedding.pkl"), "wb")
    )


def cache_processed_sentence(data_root, leads, waves="PRT"):
    data = json.load(open(os.path.join(data_root, "symbel_sentence.json")))

    res = {lead: {} for lead in leads + ["common"]}
    for symbol in data:
        for lead in ["common", *leads]:
            res[lead][symbol] = {}
            if "common" in data[symbol]:
                common = data[symbol]["common"] + " "
            else:
                common = ""

            for wave in waves:
                if lead == "common" and common:
                    write = True
                else:
                    write = False

                cur_res = common
                if wave in data[symbol]:
                    if isinstance(data[symbol][wave], str):
                        data[symbol][wave] = {"common": data[symbol][wave]}

                    if (
                        isinstance(data[symbol][wave], dict)
                        and lead in data[symbol][wave]
                    ):
                        if lead != "common" and "common" in data[symbol][wave]:
                            cur_res += data[symbol][wave]["common"] + " "
                        cur_res += data[symbol][wave][lead]
                        write = True

                if write and cur_res:
                    res[lead][symbol][wave] = cur_res.strip()

    json.dump(
        res,
        open(os.path.join(data_root, "symbol_sentence_processed.json"), "w"),
        indent=4,
    )


def main():
    data_root = "data/mit-bih-arrhythmia-database-1.0.0/symbols"
    leads = ["MLII", "V1", "V2", "V4", "V5"]
    symbols = ["N", "L", "R", "V", "A"]
    waves = "PRT"

    cache_processed_sentence(data_root, leads, waves)
    cache_embedding(data_root, symbols, waves)


if __name__ == "__main__":
    main()
