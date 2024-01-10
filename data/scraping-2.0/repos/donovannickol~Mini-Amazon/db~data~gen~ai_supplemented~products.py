import os
import openai

import pandas as pd
import numpy as np
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

def gen_products():

    t1_start = perf_counter()

    print("Generating Product Descritptions...", flush=True)
    df = pd.read_csv("sources/cleaned_data/combined.csv", sep="^")

    df["Product_ID"] = df.index

    df = df[["Product_ID", "Title", "Product_Description", "Image_URL_1", "Product_Category", "Price"]]

    df.rename(columns={"Image_URL_1":"Image_URL"})
    
    df = df.apply(gen_description, raw=True, axis=1)

    t2_stop = perf_counter()

    print("\nGenerated", row_num, "desriptions for", df.shape[0], "products in", t2_stop-t1_start, "seconds.")


    print(df.columns)
    df.to_csv("complete/products.csv", sep="^")

def gen_description(row):
    global row_num
    if(type(row[2]) == float or len(row[2]) == 0):
        prompt = "Write a detailed description about \"" + row[1] + "\"" 
        tokens = max_tokens - len(prompt) - 1
        output = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=tokens,
            temperature=temp
        )
        row[2] = output["choices"][0]['text']
        if(row_num % 100 == 0):
            print(row_num, end=" ", flush=True)
            if(row_num % 500 == 0) :
                print("Sample description:", row[2])
        row_num += 1

    return row


# gen_products()