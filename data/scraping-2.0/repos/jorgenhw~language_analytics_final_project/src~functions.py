################################
# Functions for the main.py file
################################

# IMPORTS
import requests
from bs4 import BeautifulSoup
import openai
import textdescriptives as td
import numpy as np
import os
from gtts import gTTS
from dotenv import load_dotenv # pip3 install python-dotenv For loading the environment variables (API keys and playlist URI's)
from src.classes import bcolors

# FUNCTIONS

# Set the OpenAI API key
def set_api_key():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Ask the user for a URL from the Guardian website
def ask_for_url():
    return input("Enter a URL to an article from The Guardian website: ")

# retrives the most recent briefing URL from the Guardian website
def get_most_recent_briefing():
    url = 'https://www.theguardian.com/world/series/guardian-morning-briefing'

    # Send a request to the URL and parse the response
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the link to the most recent briefing article
    article_link = soup.find('a', {'data-link-name': 'article'})
    return article_link['href'] if article_link else None

# Get the article text from the URL
def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    article = soup.find('div', {'class': 'article-body-commercial-selector'})

    for unwanted_element in article(['script', 'style', 'aside', 'figure']):
        unwanted_element.decompose()

    text = ' '.join([paragraph.text for paragraph in article.find_all('p')])

    return text

# Ask the user for their choice of options
def process_user_choice():
    while True:
        print("#############################################")
        print(f"{bcolors.OKCYAN}Choose an option (1 / 2):{bcolors.ENDC}")
        print("#############################################")
        print("1. Enter your own URL to an article from The Guardian website")
        print("2. Get the latest The Guardian Briefing")
        
        choice = input("Enter the number of your choice: ")

        if choice == '1':
            url = ask_for_url()
            return url
        elif choice == '2':
            url = get_most_recent_briefing()
            return url
        else:
            print(f"{bcolors.WARNING}Invalid choice. Please choose option 1 or 2.{bcolors.ENDC}")

# Keeps the full length of the article
def keep_full_length(text):
    return text

# Shortens the article with GPT-3
def shorten_with_gpt(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant. Summarize the following text."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

# Changes the tone of the article with GPT-3
def change_tone_elim5(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant. Explain the following text like I'm 5."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

# Getting text characteristics with textdescriptives package
def get_text_characteristics(text):
    df = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["readability", "coherence"])
    flesch_reading_ease = f"Reading score: {round(float(df['flesch_reading_ease'][0]),1)} (0 is reasy, 100 is hard, 60-70 is normal)"
    return flesch_reading_ease

# Getting main points of article with GPT-3
def main_points_with_gpt(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide the main points of the following text as bullet points."},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

# Converts the text into audio and saves the audio file in the audio_output folder
def text_to_audio(text, language='en', output_folder='audio_output', output_file='output.mp3'):
    """
    Converts input_text to audio and saves it as an MP3 file.

    Args:
        input_text (str): The text to convert to audio.
        language (str): The language in which to convert the text. Default is 'en' (English).
        output_folder (str): The folder in which to save the output audio file. Default is 'audio_output'.
        output_file (str): The name of the output MP3 file. Default is 'output.mp3'.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, output_file)
    myobj = gTTS(text=text, lang=language, slow=False)
    
    myobj.save(output_path)
    
    print(f"Audio version of the text saved as {output_path}")

# The different options for text formatting the user can chose from
def options(article_text):
    print("#############################################")
    print(f"{bcolors.OKCYAN}Choose how you would like to preprocess the article:{bcolors.ENDC}")
    print("1. Shorten")
    print("2. ELIM5")
    print("3. Characteristics")
    print("4. Main points")
    print("5. Keep full length")
    
    option = input("Enter the number of your choice: ")

    if option == '1':
        result = shorten_with_gpt(article_text)
    elif option == '2':
        result = change_tone_elim5(article_text)
    elif option == '3':
        result = get_text_characteristics(article_text)
    elif option == '4':
        result = main_points_with_gpt(article_text)
    elif option == '5':
        result = keep_full_length(article_text)
    else:
        result = f"{bcolors.WARNING}Invalid option: Enter a number between 1 and 5.{bcolors.ENDC}"
    return result

# Asks the user if they want the text read aloud or just get the text
def audio_or_text(result):
    while True:
        print("#############################################")
        audio_option = input(f"{bcolors.OKCYAN}Would you like the text read aloud? (yes/no) {bcolors.ENDC}").strip().lower()
        
        if audio_option in ['yes', 'y']:
            text_to_audio(result)
            break
        elif audio_option in ['no', 'n']:
            print("#############################################")
            save_option = input(f"{bcolors.OKCYAN}Do you want to save the text to a file? Otherwise it will be printed in the console. (yes/no) {bcolors.ENDC}").strip().lower()
            if save_option in ['yes', 'y']:
                if not os.path.exists('text_output'):
                    os.makedirs('text_output')
                output_path = os.path.join('text_output', 'output.txt')
                with open(output_path, 'w') as file:
                    file.write(result)
                print(f"Text saved as {output_path}")
            else:
                print(result)
            
            break
        else:
            print(f"{bcolors.WARNING}Please either answer 'yes' or 'no'. You can also use 'y' or 'n'.{bcolors.ENDC}")