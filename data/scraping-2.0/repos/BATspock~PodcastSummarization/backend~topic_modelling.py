#Import necessary packages
import os
import openai as ai
import json
import requests
import numpy as np
import pandas as pd

#Set API key - Need to do it from environment
#ai.api_key = os.environ['OPENAI_API_KEY']
ai.api_key = os.getenv('OPENAI_API_KEY')

#Function to generate subtitle topic from text chunk
def generate_subtitle_topic(text_chunk: str)-> str:

    #Creating prompt from text chunk
    prompt = "This is a subtitle topic generator. The input is a text chunk. The output is a subtitle topic. \n\nText chunk: " + text_chunk + "\n\nSubtitle topic:"

    #Call openai api to generate subtitle topic
    response = ai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 500,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0,
        stop = ["\n"]
    )
    #Return subtitle topic
    return response['choices'][0]['text']

#Read dataframe with speakers and text chunks
chunk_df = pd.read_csv("chunk_df.csv")

#Getting subtitle topics for each text chunk
chunk_df['subtitle_topic'] = chunk_df['text_chunk'].apply(lambda x: generate_subtitle_topic(x))

#Write output to csv
chunk_df.to_csv("Subtitle_Topic.csv", index = False)