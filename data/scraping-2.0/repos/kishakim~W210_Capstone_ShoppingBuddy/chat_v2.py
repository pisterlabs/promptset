"""## 1.Import Libraries"""
import torch
import clip


import os.path as osp
import pickle


from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import openai

# import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

import os
import openai

"""## 2.Set Variables"""

load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

"""### 3.1 Clip

## INSTRUCTION
### to run the model
### 1. download pickle.zip and target_images.zip
### https://drive.google.com/drive/folders/1BgfQoRVKosqvPssa5V-sdCo9X1lLeRgx
### 2. unzip the data and put into folder demo_data
### 3. create an access key for chatGPT (only need to put in $5)
### https://platform.openai.com/account/api-keys
"""

import torch
import torch.nn as nn
import clip
from PIL import Image
import pandas as pd
import requests
import os.path as osp
import pickle
import random
import numpy as np
from pathlib import Path
import sys
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import shutil


def read_pickle(dir):
    with open(dir, "rb") as handle:
        b = pickle.load(handle)
    return b


def write_pickle(dir, data):
    with open(dir, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Timer:
    def __init__(self):

        self.t1 = None

    @staticmethod
    def delta_to_string(td):

        res_list = []

        def format():
            return ", ".join(reversed(res_list)) + " elapsed."

        seconds = td % 60
        td //= 60
        res_list.append(f"{round(seconds,3)} seconds")

        if td <= 0:
            return format()

        minutes = td % 60
        td //= 60
        res_list.append(f"{minutes} minutes")

        if td <= 0:
            return format()

        hours = td % 24
        td //= 24
        res_list.append(f"{hours} hours")

        if td <= 0:
            return format()

        res_list.append(f"{td} days")

        return format()

    def __enter__(self):

        self.t1 = time.time()

    def __exit__(self, *args, **kwargs):

        t2 = time.time()
        td = t2 - self.t1

        print(self.delta_to_string(td))


def top_n(input_dict, n):
    return dict(sorted(input_dict.items(), key=itemgetter(1), reverse=True)[:n])


# def find_products(text_input, category_df, title_df):

#     text_input = [text_input]

#     # stage one, compare categories
#     category_df = category_df[~category_df["encoded_category"].isna()]
#     categories = list(category_df["category"].values)

#     categories_features = torch.stack(list(category_df["encoded_category"].values))
#     encoded_texts = clip.tokenize(text_input).to(device)

#     with torch.no_grad():

#         text_features = model.encode_text(encoded_texts)

#         categories_features /= categories_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         similarity =  100 * categories_features @ text_features.T

#     res = dict(zip(categories, similarity.reshape(-1).tolist()))

#     res = sorted(res.items(), key=itemgetter(1), reverse=True)

#     n = 100
#     res_list = []
#     res_count = 0
#     for r in res:
#         cur_res = meta_df[meta_df["combined_category"] == r[0]]
#         res_list.append(cur_res)
#         res_count += len(cur_res)
#         if res_count >= n:
#             break

#     # stage two, compare titles
#     res = pd.concat(res_list, axis=0)
#     res = res.title.values

#     title_df = title_df[title_df.title.isin(res)]
#     titles = list(title_df["title"].values)

#     title_features = torch.stack(list(title_df["encoded_title"].values))

#     with torch.no_grad():

#         title_features /= title_features.norm(dim=-1, keepdim=True)
#         similarity =  100 * title_features @ text_features.T

#     res = dict(zip(titles, similarity.reshape(-1).tolist()))

#     res = sorted(res.items(), key=itemgetter(1), reverse=True)

#     n = 5
#     res = res[:n]
#     res_set = set([r[0] for r in res])
#     res = meta_df[meta_df["title"].isin(res_set)]["uniq_id"].values

#     return res


def show_images(res):
    n = len(res)
    fig, ax = plt.subplots(1, n)

    fig.set_figheight(5)
    fig.set_figwidth(5 * n)

    for i, image in enumerate(res):
        img_path = image_path(image)
        img = mpimg.imread(img_path)
        ax[i].imshow(img)
        ax[i].axis("off")
        # ax[i].set_title(get_label(image), fontsize=8)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()


def image_path(uid):
    return osp.join(image_storage, f"{uid}.jpg")


def load_data(pickle_path):
    category_df = read_pickle(osp.join(pickle_path, "categories.pkl"))
    title_df = read_pickle(osp.join(pickle_path, "titles.pkl"))
    meta_df = read_pickle(osp.join(pickle_path, "meta_data.pkl"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    return device, model, preprocess, category_df, title_df, meta_df


image_storage = "data/image"
pickle_path = "data/pickle"

"bolt://44.215.124.63:7687"
"neo4j+s://493c5686.databases.neo4j.io"
"bolt://localhost:7687"

# this worked - https://github.com/neo4j/NaLLM
from neo4j import GraphDatabase

host = "neo4j+s://demo.neo4jlabs.com"
user = "companies"
password = "companies"
driver = GraphDatabase.driver(host, auth=(user, password))


def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return result.to_df()


system_prompt = """
You are an assistant that helps intake user input and describe a furniture that a user is looking for with explicit metadata like color, material, price and etc.
The latest prompt contains the information, and you need to generate a human readable description based on the given information.
Make the description sound as a detailed information.
Do not add any additional information that is not explicitly provided in the latest prompt.
I repeat, do not add any information that is not explicitly given.
"""


def generate_user_prompt(question, context):
    return f"""
   The user input is {question}
   Provide a description of what the user is looking for by using the provided information:
   {context}
   """


def retrieve_context(question, k=3):
    data = run_query(
        """
    // retrieve the embedding of the question
    CALL apoc.ml.openai.embedding([$question], $apiKey) YIELD embedding
    // match relevant movies
    MATCH (m:Movie)
    WITH m, gds.similarity.cosine(embedding, m.embedding) AS score
    ORDER BY score DESC
    // limit the number of relevant documents
    LIMIT toInteger($k)
    // retrieve graph context
    MATCH (m)--()--(m1:Movie)
    WITH m,m1, count(*) AS count
    ORDER BY count DESC
    WITH m, apoc.text.join(collect(m1.title)[..3], ", ") AS similarMovies
    MATCH (m)-[r:ACTED_IN|DIRECTED]-(t)
    WITH m, similarMovies, type(r) as type, collect(t.name) as names
    WITH m, similarMovies, type+": "+reduce(s="", n IN names | s + n + ", ") as types
    WITH m, similarMovies, collect(types) as contexts
    WITH m, "Movie title: "+ m.title + " year: "+coalesce(m.released,"") +" plot: "+ coalesce(m.tagline,"")+"\n" +
          reduce(s="", c in contexts | s + substring(c, 0, size(c)-2) +"\n") + "similar movies:" + similarMovies + "\n" as context
    RETURN context
  """,
        {"question": question, "k": k, "apiKey": openai.api_key},
    )
    return data["context"].to_list()


def generate_description_for_clip_KG(question):
    # Retrieve context
    context = retrieve_context(question)
    # Print context
    for c in context:
        print(c)
    # Generate answer
    response = run_query(
        """
  CALL apoc.ml.openai.chat([{role:'system', content: $system},
                      {role: 'user', content: $user}], $apiKey) YIELD value
  RETURN value.choices[0].message.content AS answer
  """,
        {
            "system": system_prompt,
            "user": generate_user_prompt(question, context),
            "apiKey": openai.api_key,
        },
    )
    return response["answer"][0]


messages = []

res_list = []

prefix = (
    "considering what the user asked before, what is the user looking for with the following request."
    " Only respond with the product description no more than 30 words:"
)


def Chat(message):
    if message == "quit":
        exit()

    if message:
        print(f"System msg - User entered: {message}")
        messages.append(
            {"role": "user", "content": f"{prefix} {message}"},
        )

        needs = message
        missing_metadata = []
        for i in ["color", "price", "material", "room"]:

            messages2 = []
            prefix_question = (
                "Does the following text contain information on "
                + str(i)
                + " of a furniture item?"
                + "Respond in yes or no"
            )

            print("needs: ", needs)
            needs = "text: " + needs
            messages2.append(
                {"role": "user", "content": f"{prefix_question} {needs}"},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages2
            )
            reply2 = chat.choices[0].message.content
            print("reply2: ", reply2)
            print("System msg - Did it have information on " + str(i) + " ?", reply2)
            if "no" in str.lower(reply2):
                print
                print("System msg - No info on " + str(i) + " was provided")
                # data_asset = "Is there a specific " + str(i) + " you are looking for?"
                data_asset = input(
                    "Is there a specific " + str(i) + " you are looking for? :"
                )
                needs = str(needs) + " in " + str(i) + " " + str(data_asset)
                print("System msg - ", needs)
                messages.append(
                    {"role": "user", "content": f"{prefix} {needs}"},
                )
            else:
                print("System msg - Info on " + str(i) + " was provided")
                print("System msg - ", needs)
                messages.append(
                    {"role": "user", "content": f"{prefix} {needs}"},
                )
        # using all info collected, generate description for clip, instead of baseline openai.ChatCompletion.create function
        print("System msg - FINAL ", needs)
        reply = generate_description_for_clip_KG(needs)

        print("Description to put to Clip:", reply)
        return reply

while True:
    message = input("user: ")
    if message == "quit":
        exit()
    else:
        Chat(message)