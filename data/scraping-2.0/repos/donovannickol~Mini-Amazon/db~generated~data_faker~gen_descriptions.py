import os
import openai
import datagenconfig as dgc
import pandas as pd
import numpy as np


#SETUP
openai.organization = dgc.config['ORG_KEY']
openai.api_key = os.getenv(dgc.config['SECRET_KEY'])
openai.api_key=dgc.config['SECRET_KEY']


def gen_fake_data():
    df = pd.read_csv("scraped_data/combined.csv")
    
    # df = gen_fake_descriptions(df)
    
    df.to_csv("ai_generated/products.csv")



def gen_fake_descriptions(df):

    mod="text-ada-001"
    max_token=750
    temp=0.7
    count = 0
    print(df['Product_Description'].head(5))
    for i in range(0,df.shape[0]):
        if type(df.at[i,"Product_Description"]) == float:
            title = df.at[i,"Title"]
            promp = "Describe \'" + title + "\'"
            output= openai.Completion.create(
                model=mod,
                prompt=promp,
                max_tokens=max_token,
                temperature=temp
            )
            df.at[i,"Product_Description"] = output['choices'][0]['text']
    return df


    # df.to_csv('ai_generated/filled_reviews.csv')


gen_fake_data()