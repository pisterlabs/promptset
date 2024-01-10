# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import datasets
from datasets import load_dataset,load_from_disk
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
openai.api_key="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"

def get_ubuntu(path):
    with open(path,"r") as f:
        data = f.readlines()
    print(len(data))
    print(data[0])
    # groups = []

    # for row in data:
    #     group = []
            

if __name__ == "__main__":
    path = "/mmu_nlp/wuxing/suzhenpeng/SimpleReDial-v1/MyReDial/data/ubuntu_old/test.txt" 
    get_ubuntu(path)