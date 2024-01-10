from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai
import os
from pytube import YouTube
from pydub import AudioSegment
from pkg_resources import load_entry_point
import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError
from pydotenvs import load_env
import codecs
import openai
import requests
import json
import time
from PIL import Image

#load local environment
load_env()

#set up Youtube API key and credentials
SERVICE_ACCOUNT_JSON = os.environ.get('SERVICE_ACCOUNT_JSON')   #path to credentials json file
api_key = os.environ.get('YOUTUBE_KEY')
scopes = ['https://www.googleapis.com/auth/youtube.force-ssl']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=scopes)

#set up OpenAI API key
openai.api_key = os.environ.get('OPENAI_KEY')

#set up AWS credentials
aws_access_key = os.environ.get('AWS_KEY')
aws_secret_key = os.environ.get('AWS_SECRET')
user_bucket = os.environ.get('USER_BUCKET')

#set up YouTube Data API client
youtube = build('youtube', 'v3', developerKey=api_key, credentials=credentials)

#authenticate S3 resource with your user credentials that are stored in your .env config file
s3resource = boto3.resource('s3',
                        region_name='us-east-1',
                        aws_access_key_id = aws_access_key,
                        aws_secret_access_key = aws_secret_key
                        )

def generate_prompt(category_selected):
    """Function that takes in the physiotherapy category user selected by user & creates the prompt for OpenAI's GPT 
    chat completion model. The prompt is created using the JSON file containing YouTube videos scripts.
    -----
    Input parameters:
    category_selected: string
        This is the category selection through streamlit
    -----
    Returns:
    prompt: string
        This is the prompt created
    """

    prompt = ""
    for title in json_content:  #traverse through video titles in the JSON
        if (json_content[title]['category']==category_selected):    #check if video category matches selected category
            prompt += "###\nTitle: " + title +"\nText: " + json_content[title]['transcription'] #generate prompt with title & transcription of video
    return prompt

#set up streamlit app
icon = Image.open('workout-app.jpeg')  #for icon of the streamlit website tab
st.set_page_config(page_title="Fit Finder App", page_icon=icon, layout="wide")
image = Image.open('fitfinder-icon.png')
st.image(image, width=300)
st.markdown("<h1 style='color: #746E9E;'>FitFinder</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #746E9E;'>Find you stride</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #746E9E;'>FitFinder gives you refined search capabilities to find videos consisting the exercise you need</h3>", unsafe_allow_html=True)
st.write("Note: Capabilities of this app is limited to Physiotherapy as of now.")

content_object = s3resource.Object(user_bucket, "yt_json.json") #get JSON file from S3 bucket
file_content = content_object.get()['Body'].read().decode('utf-8')  
json_content = json.loads(file_content) #load contents from JSON file

categories = []
for title in json_content:
    categories.append(json_content[title]['category'])  #find all categories from JSON

category_selection = st.selectbox('Select a category:', set(categories))    #display unique categories using set()
if (category_selection != ""):  #if some category is selected
    cateogry_prompt = generate_prompt(category_selection) #call function to generate prompt
    
    body_part = st.text_input("Enter the body part/area you wish to focus on:") #ask user inputs to refine search results
    exercise_name = st.text_input("Are you looking for a particular exercise, enter name:") #ask user inputs to refine search results
    
    ask_btn = st.button("Ask")
    if ask_btn:
        if body_part.strip() == "":
            st.error("Please specify body part/area")   #body area/part to focus on is required
            st.stop()

        if exercise_name != "": #if exercise name is mentioned, add that to user question given as prompt for chat completion
            user_question = f"Which of these scripts include exercises that involve {body_part} & has exercise {exercise_name}? Please give the title"

        else:
            user_question = f"Which of these scripts include exercises that involve {body_part}? Please give the title"
        
        with st.spinner('Refining search...'):
            #system prompt for chat completion
            system_prompt = "Your task is to find one which of the three scripts given about exercises best contains the information that the user asks"

            try:
                #call GPT model through API on the prompt & context we have
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                                    messages=[
                                                                        {"role": "system", "content": system_prompt},
                                                                        {"role": "user", "content": "Here are the 3 video scripts with titles for each"},
                                                                        {"role": "user", "content": cateogry_prompt},
                                                                        {"role": "user", "content": user_question},
                                                                        ],
                                                                    temperature=0,
                                                                    max_tokens=100,
                                                                    top_p=1,
                                                                    frequency_penalty=0,
                                                                    presence_penalty=0)

                response = response.choices[0].message.content.strip()  #store response
                # print(response)

            #in case gpt model is at capacity
            except openai.error.RateLimitError: 
                st.error("Sorry, the system is at full capacity. Please try again later.")
                st.stop()

            flag = True #to check if a relevant video exists in our context or not
            for title in json_content:  #to provide user with video link, find the video link from JSON based on gpt response
                if (title in response):
                    flag = False    #set flag to false since a video is found
                    yt_link = json_content[title]["link"]
                    yt_script = json_content[title]['transcription']

                    try:
                    #call GPT chat completion API to get details about exercises mentioned
                        new_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                                            messages=[
                                                                                {"role": "user", "content": f"Here is a physiotherapy related video script: {yt_script}"},
                                                                                {"role": "user", "content": "List the exercises mentioned in this video"}
                                                                                ],
                                                                            temperature=0,
                                                                            max_tokens=200,
                                                                            top_p=1,
                                                                            frequency_penalty=0,
                                                                            presence_penalty=0)

                        new_response = new_response.choices[0].message.content.strip()  #store response
                        # print(new_response)

                        st.video(yt_link) #show refined results
                        st.write("Here are the exercises in this video: \n" + new_response)  #write the response about exercising mentioned
                    
                    except openai.error.RateLimitError: 
                        st.error("Sorry, the system is at full capacity. Please try again later.")
                        st.stop()

            if (flag):  #in case the video based on the selections is not available 
                st.info("FitFinder has limited capabailities right now, we are unable to provide you with answers for these selections")