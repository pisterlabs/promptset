# Trabslate the headkines for newspapers in different countries
####################################### IMPORTS ###############################################
import openai
import logging
import os

from dotenv import load_dotenv

import requests
import bs4

##################################### CONSTANTS ###############################################

DATA_DIRECTORY ='dev'

# load environment variables from .env file
load_dotenv()

############################################ Data #############################################

country_newspaper_dict = {"united states": "The New York Times",
                            "united kingdom": "The Guardian",
                            "australia": "The Sydney Morning Herald",
                            "canada": "The Globe and Mail",
                            "india": "The Times of India",
                            "ireland": "The Irish Times",
                            "new zealand": "The New Zealand Herald",
                            "pakistan": "Dawn",
                            "singapore": "The Straits Times",
                            "south africa": "The Mail & Guardian",
                            "spain": "El País",
                            "france": "Le Monde"}

newspaper_url_dict = {"The New York Times": "https://www.nytimes.com/",
                        "The Guardian": "https://www.theguardian.com/international",
                        "The Sydney Morning Herald": "https://www.smh.com.au/",
                        "The Globe and Mail": "https://www.theglobeandmail.com/",
                        "The Times of India": "https://timesofindia.indiatimes.com/",
                        "The Irish Times": "https://www.irishtimes.com/",
                        "The New Zealand Herald": "https://www.nzherald.co.nz/",
                        "Dawn": "https://www.dawn.com/",
                        "The Straits Times": "https://www.straitstimes.com/",
                        "The Mail & Guardian": "https://mg.co.za/",
                        "El País": "https://elpais.com/",
                        "Le Monde": "https://www.lemonde.fr/"}


################################## HELPER FUCTIONS #############################################

# Use chat completion
def get_chat_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

# Standard completion
def get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=300, stop="\"\"\""):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop)

    return response.choices[0].text


####################################### LOGGING ################################################

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging


####################################### START #################################################
logging.info('Start of program')

# Get the current DATA directory
home = os.getcwd()
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(data_dir)

os.chdir(data_dir)

# Authenticate with OpenAI                             
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key


####################################### MAIN ##################################################

country = input("What country are you interested in for news? ")
newspaper = country_newspaper_dict[country.lower()]
url = newspaper_url_dict[newspaper]
logging.info(url)
result = requests.get(url)
logging.info(result.status_code)
soup = bs4.BeautifulSoup(result.text, "html.parser")

logging.info(soup.title.text)
headings = soup.find_all({"h1", "h2", "h3"})[:3]
for heading in headings:
    logging.info(heading.text.strip())

# Translate the headlines into english
prompt = "Translate the following headlines into English:\n"
for heading in headings:
    prompt += heading.text.strip() + "\n"
prompt += "\n\nTranslation:"

translation = get_completion(prompt, 
                             model="gpt-3.5-turbo-instruct", 
                             temperature=0,
                             max_tokens=300)

logging.info(translation)