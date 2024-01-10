import openai
import json
from tqdm import tqdm
import webvtt

f = open("/shared/medhini/COIN/COIN.json")
data = json.load(f)["database"]
dic = {}
category = []
for item in data:
    if data[item]["class"] not in dic:
        dic[data[item]["class"]] = {}
        dic[data[item]["class"]]["ids"] = []
    dic[data[item]["class"]]["ids"].append(item)
for category in dic:
    dic[category]["ids"]  = ",".join(dic[category]["ids"])

with open("./COIN/base/coin_categories.json","w") as f:
    json.dump(dic, f)