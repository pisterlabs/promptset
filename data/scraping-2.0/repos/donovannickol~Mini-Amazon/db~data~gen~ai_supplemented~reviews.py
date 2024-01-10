import os
import openai

import pandas as pd
import numpy as np
import random
import openai_keys as keys
from time import perf_counter


#SETUP
openai.organization = keys.config['ORG_KEY']
openai.api_key = os.getenv(keys.config['SECRET_KEY'])
openai.api_key=keys.config['SECRET_KEY']
model=keys.model
max_tokens=keys.max_tokens
temp=keys.temp


row_num = 0

review_map = {1: "very negative", 2: "negative", 3:"neutral", 4:"positive", 5:"very positive"}

def gen_reviews():

    global row_num

    t1_start = perf_counter()

    df = pd.read_csv("complete/products.csv", sep="^")

    df["Rating"] = 0
    df["Review"] = 0
    df = df[["Product_ID", "Rating","Title", "Review"]]

    num_products = df.shape[0]


    print("Generating Product Review Distribution...", end=" ", flush=True)
    
    df = df.apply(gen_review_distribution, raw=True, axis=1)

    ## Expand out the list distribution so that each index is its own row
    df = df.explode("Rating")

    row_num = 0

    print("\nGenerating Product Reviews...", end=" ", flush=True)

    df = df.apply(gen_review, raw=True, axis=1)

    t2_stop = perf_counter()

    print("\nGenerated", row_num, "reviews for", num_products, "products in", t2_stop-t1_start, "seconds.")

    df = df.drop(columns=["Title"])
    # df.to_csv("complete/reviews.csv", sep="^")


def gen_review_distribution(row):
    global row_num
    if(row_num % 100 == 0):
        print(row_num, end=" ", flush=True)
    row_num += 1
    row[1] = [random.randint(1,5) for i in range(20)]
    return row

def gen_review(row):
    global row_num
    prompt = "Write a " + review_map[row[1]] + " review about \"" + row[2] + "\"" 
    tokens = max_tokens - len(prompt) - 1
    row[3] = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=tokens,
        temperature=temp
    )["choices"][0]['text']
    if(row_num % 100 == 0):
        print(row_num, end=" ", flush=True)
        if(row_num % 500 == 0) :
            print("Sample review:", row[3])
    row_num += 1
    return row


# gen_reviews()