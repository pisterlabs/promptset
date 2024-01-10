####################################
############ FUNCTIONS #############
####################################
import openai
import os
from dotenv import load_dotenv # pip3 install python-dotenv For loading the environment variables (API keys and playlist URI's)
import whisper
import tqdm # progress bar when transcribing with whisper

# WHISPER #####################################
# add verbose=True to see the progress
def import_whisper(audio_file_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file_path, verbose = False) # verbose = True to see the progress of transcribing
    text = result["text"]
    return text

# OPENAI #######################################
def set_api_key():
    print("Setting API key...")
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Keeps the full length of the article
def keep_full_length(text):
    print("Keeping full length...")
    return text

# Shortens the article with GPT-3
def shorten_with_gpt(text):
    print("Shortening with GPT-3...")
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "Please provide a concise summary of the following text, condensing it into a clear and coherent response. Ensure the summary is concise and informative, providing key insights and points from the original text. Limit the response to a length of approximately 15-20 sentences."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

# Getting main points of article with GPT-3
def main_points_with_gpt(text):
    print("Getting main points with GPT-3...")
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "Please present the main points from the following text as bullet points. Ensure that the bullet points are clear, concise, and capture the key insights and information from the text. Provide at least 20 points."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

# Changes the tone of the article with GPT-3
def change_tone_elim5(text):
    print("Changing tone with GPT-3...")
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "You are a helpful assistant. Explain the following text like I'm 5."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

# Structure the article with GPT-3
def restructure_gpt3(text):
    print("restructuring the text with GPT-3...")
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "Please reorganize the content from the following text into a concise and coherent format. Utilize tables, bullet points but mainly and text to present the information clearly and effectively, ensuring that the key points are highlighted for easy understanding. Output should be approximately 20-40 sentences."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content'] 