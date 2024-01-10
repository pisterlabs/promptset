from typing import Union
import instaloader
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from revChatGPT.V1 import Chatbot
import os
import openai
from dotenv import load_dotenv
import glob
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import openai
import requests
import json
from twilio.rest import Client
import datetime

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")

def main():
    vertex_captions = open("./captions.txt", "r").readlines()
    # print(vertex_captions)
    # print(type(vertex_captions))
    for idx, caption in enumerate(vertex_captions):
        print(caption)
        print(type(caption))
        print(caption[-2:])
        if caption[-2:]=="\n":
            vertex_captions[idx]=caption[:-2]
            print(caption[:-2])
    print("Loop Ended")
    # vertex_captions=['a model walks down the runway wearing a pink and white saree','a man wearing a denim jacket is walking down the street','a woman in a blue dress is standing in front of a couch',
    # 'a man is wearing a black coat and grey pants','a man wearing a black shirt and green cargo pants','a man wearing glasses and a colorful jacket is laughing']
    trend_messages=[{
        'role' : 'system',
        'content' : f'''you work for a fashion company and your task is to read from an array of strings in which each string is desciption of an image of a fashion influencer\n
        and your task is to understand about what new types of clothes they are wearing and what fashion trends is being followed.you are given this array = < {vertex_captions} >. \n
        take time to think & analyze while iterating through this array to understand what people are wearing and what is or can be name of those fashion clothes and what trends are these. you only speak JSON and output should strictly be like \n
        {{"fashion_trends" : ["array items should be name of fashion clothes currently trending after your analysis of this array.]}} \n
        and there should not be any extra text in your response'''
    }]
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=trend_messages)
    print(chat_completion)
    gpt_response = json.loads(chat_completion['choices'][0]['message']['content'])
    with open ("./trends.txt", "w") as file:
        for trend in gpt_response['fashion_trends']:
            file.write(trend + "\n")
    print(gpt_response)

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcpconfig.json"
    # if len(sys.argv) != 4:
    #     print("Usage: python generate.py <folder_path> <google_credentials>")
    # else:
    main()