#!/usr/bin/env python                                                                                                       
'''                                                                                                                        
Functions to load and call OpenAI APIs.

Copyright 2023, Arno Klein, MIT License

'''
import os
import openai
import json
from pathlib import Path
from io_files import json2png

#-----------------------------------------------------------------------------                                              
# Load GPT                                                                                     
#-----------------------------------------------------------------------------                                              
gpt_model = "gpt-4"  #"gpt-3.5-turbo" #"text-davinci-003"
if gpt_model == "gpt-4":
    openai.api_key = os.getenv("OPENAI_API_KEY")
elif gpt_model == "gpt-3.5-turbo":  #"text-davinci-003"
    openai.api_key = os.getenv("OPENAI3_API_KEY")


#-----------------------------------------------------------------------------                                              
# Functions                                                                                                  
#-----------------------------------------------------------------------------                                              


def get_gpt_response(prompt, model=gpt_model):
    completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
    response = completion.choices[0].message.content

    return response


def call_dalle(prompt, size="256x256", output_format="png", output_dir="tmp", verbose=False):

    if output_format == "url":
        response_format = "url"
    elif output_format == "png":
        response_format = "b64_json"

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size,
        response_format=response_format
    )

    if output_format == "url":
        url = response["data"][0]["url"]
        if verbose:
            print(url)

        return url

    elif output_format == "png":

        json_file = os.path.join(output_dir, f"{prompt[:5]}-{response['created']}.json")
        with open(json_file, mode="w", encoding="utf-8") as file:
            json.dump(response, file)
            #if verbose:
            #    print(response["data"][0]["b64_json"][:50])
        
        png_file = json2png(json_file, output_dir)
        if verbose:
            print(png_file)

        return png_file


#output = openai.ChatCompletion.create(
#  model="gpt-4",
#  messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "Who won the world series in 2020?"},
#        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#        {"role": "user", "content": "Where was it played?"}
#    ]
#)
#print(output)

