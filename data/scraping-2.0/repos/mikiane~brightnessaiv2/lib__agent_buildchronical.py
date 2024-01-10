
# ----------------------------------------------------------------------------
# Project: Brightnessaiv2
# File:    lib__agent_buildchronical.py
#  Set of functions to build a chronic based on feeds
# 
# Author:  Michel Levy Provencal
# Brightness.ai - 2023 - contact@brightness.fr
# ----------------------------------------------------------------------------

import feedparser
from random import randint
from elevenlabs import set_api_key
from dotenv import load_dotenv
from lib__script_tasks import truncate_strings, request_llm
import os
from urllib.parse import unquote
from queue import Queue
from pydub import AudioSegment
from moviepy.editor import *
from datetime import date
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
import mimetypes
import requests
from bs4 import BeautifulSoup
from datetime import date
from pydub import AudioSegment
from random import randint
from elevenlabs import set_api_key
from dotenv import load_dotenv
import os
from moviepy.editor import *

import json
from PIL import Image, ImageDraw, ImageFont
from num2words import num2words
import re
from lib__env import *
from openai import OpenAI
import xml.etree.ElementTree as ET





model="gpt-4"
load_dotenv(DOTENVPATH)
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PODCASTS_PATH = os.environ.get("PODCASTS_PATH")
SENDGRID_KEY = os.environ.get("SENDGRID_KEY")




def replace_numbers_with_text(input_string):
    ################################################################################################################
    """
    This function, replace_numbers_with_text, converts numbers and percentages 
    in a given string to their corresponding text representation in French. 
    It first finds percentages in the string (like 50%), 
    converts the numerical part to French words (like "cinquante"), 
    and replaces the original percentage with this text followed by pour cent. 
    Next, it finds standalone numbers, converts each to French words, 
    and replaces the original numbers with their text equivalents. 
    The function returns the modified string with these conversions applied.
    """
    ################################################################################################################
    # Remplacer les pourcentages
    percentages = re.findall(r'\d+%', input_string)
    for percentage in percentages:
        number = percentage[:-1]
        number_in_words = num2words(number, lang='fr')
        input_string = input_string.replace(percentage, f"{number_in_words} pour cent")
    
    # Remplacer les nombres
    numbers = re.findall(r'\b\d+\b', input_string)
    for number in numbers:
        number_in_words = num2words(number, lang='fr')
        input_string = input_string.replace(number, number_in_words)
    return input_string




def split_text(text, limit=1000):
    ################################################################################################################
    """
    This function splits the text into chunks of around 1000 characters. \n
    It splits before a newline character.
    """
    ################################################################################################################
    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) <= limit:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks





def texttospeech(text, voice_id, filename):
    ##################################################################
    """
    Function to convert text to speech with Eleven Labs API
    """
    ##################################################################
    try:
        set_api_key(str(ELEVENLABS_API_KEY))
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
        }

        data = {
        "text": text,
        "model_id": "eleven_multilingual_v1",
        "voice_settings": {
            "stability": 0.95,
            "similarity_boost": 1
            }
        }

        response = requests.post(url, json=data, headers=headers)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    
    except requests.RequestException as e:
        print(f"Failed to convert text to speech: {e}")
        return
    
    
    
def convert_and_merge(text, voice_id, final_filename):
    ##################################################################
    """
    This function splits the text, converts each chunk to speech and merges all the resulting audio files.
    """
    ##################################################################
    chunks = split_text(text)
    filenames = []

    # Add intro sequence to the beginning
    combined = AudioSegment.from_mp3(str(LOCALPATH) + "sounds/intro.mp3")
    #combined = AudioSegment.from_mp3("/home/michel/extended_llm/sounds/intro.mp3")

    for i, chunk in enumerate(chunks):
        filename = f"{i}.mp3"
        filenames.append(filename)
        texttospeech(chunk, voice_id, filename)
        
        # Concatenate each audio segment
        audio_segment = AudioSegment.from_mp3(filename)
        combined += audio_segment

    # Add outro sequence to the end
    #combined += AudioSegment.from_mp3("/home/michel/extended_llm/sounds/outro.mp3")
    combined += AudioSegment.from_mp3(str(LOCALPATH) + "sounds/outro.mp3")

    # Save the final concatenated audio file
    combined.export(final_filename, format='mp3')

    # Delete temporary audio files
    for filename in filenames:
        os.remove(filename)




def mailaudio(title, audio, text, email):
    ##################################################################
    """
    Fonction pour envoyer un e-mail avec une pièce jointe via SendGrid.
    
    Args:
        audio (str): Le chemin vers le fichier à joindre.
        image (str) : Le chemin vers le fichier image à joindre.
        destinataire (str): L'adresse e-mail du destinataire.
        message (str, optional): Un message à inclure dans l'e-mail. Par défaut, le message est vide.
    """
    ##################################################################
    # Création de l'objet Mail
    message = Mail(
        from_email='contact@brightness.fr',
        to_emails=email,
        subject=title,
        plain_text_content=text)
    
    # Ajout des destinataires en BCC
    # for email in destinataires:
    message.add_bcc('contact@mikiane.com')
        
    # Lecture du fichier audio à joindre
    with open(audio, 'rb') as f:
        data_audio = f.read()

    # Encodage du fichier audio en base64
    encoded_audio = base64.b64encode(data_audio).decode()
    
    # Détermination du type MIME du fichier audio
    mime_type_audio = mimetypes.guess_type(audio)[0]
    
    # Création de l'objet Attachment pour l'audio
    attachedFile_audio = Attachment(
        FileContent(encoded_audio),
        FileName(audio),
        FileType(mime_type_audio),
        Disposition('attachment')
    )
    message.add_attachment(attachedFile_audio)

    # Tentative d'envoi de l'e-mail via SendGrid
    try:
        sg = SendGridAPIClient(SENDGRID_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)
        print("\n")
        print(str(e))



def mail_nofile(title, text, email):
    ##################################################################
    """
    Fonction pour envoyer un e-mail sans pièce jointe via SendGrid.
    """
    ##################################################################

    # Création de l'objet Mail
    message = Mail(
        from_email='contact@brightness.fr',
        to_emails=email,
        subject=title,
        plain_text_content=text)
    
    # Ajout des destinataires en BCC
    # for email in destinataires:
    message.add_bcc('contact@mikiane.com')
        
    # Tentative d'envoi de l'e-mail via SendGrid
    try:
        sg = SendGridAPIClient(SENDGRID_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)
        print("\n")
        print(str(e))


def mail_html(title, text, email):
    ################################################################## 
    """
    Fonction pour envoyer un e-mail sans pièce jointe via SendGrid.
    """
    ##################################################################

    # Création de l'objet Mail
    message = Mail(
        from_email='contact@brightness.fr',
        to_emails=email,
        subject=title,
        html_content=text)
    
    # Ajout des destinataires en BCC
    # for email in destinataires:
    message.add_bcc('contact@mikiane.com')
        
    # Tentative d'envoi de l'e-mail via SendGrid
    try:
        sg = SendGridAPIClient(SENDGRID_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)
        print("\n")
        print(str(e))



def extract_first_link(rss_url):
    ########################################################################################################################
    """
    Function that return the first feed from an RSS feed
    """
    ########################################################################################################################
    feed = feedparser.parse(rss_url)
    if len(feed.entries) > 0:
        return feed.entries[0].link
    else:
        return None


def extract_n_links(rss_url, n):
    ########################################################################################################################
    """
    Function that return the first n feed from an RSS feed
    """
    ########################################################################################################################
    feed = feedparser.parse(rss_url)
    links = []
    for i in range(min(n, len(feed.entries))):
        links.append(feed.entries[i].link)
    return links


def extract_title(url):
    ########################################################################################################################
    """
    Function that rextract a title from a web page basedon its url
    """
    ########################################################################################################################

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        return soup.title.string
    except AttributeError:
        return "No Title Found"





def execute(prompt, site, input_data, model="gpt-4"):
    ########################################################################################################################
    """
    Function execute a prompt with the OpenAI API / with some context (brain_id, ur or input_data) and return the result
    """
    ########################################################################################################################
    # extract news from url
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    context = ""
    if site:  # only proceed if site is not empty
        try:
            response = requests.get(site, headers=headers)
            response.raise_for_status()  # raise exception if invalid response
            
            soup = BeautifulSoup(response.content, "html.parser")
            # remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            context = soup.get_text()
        except (requests.RequestException, ValueError):
            print(f"Failed to get content from {site}.")

    prompt, context, input_data = truncate_strings(prompt, context, input_data, 6000)
    
    if model == "gpt-4":
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, context, input_data, 12000)
        
    if model == "gpt-4-1106-preview":
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, context, input_data, 200000)
  
        
    if model == "gpt-3.5-turbo-16k":
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, context, input_data, 24000)

        
    # Appel au LLM
    res = request_llm(prompt, context, input_data, model)
    return (res)


def create_image_with_text(text, input_file, output_file):
    # Ouvrir l'image existante
    img = Image.open(input_file)

    # Marge désirée
    margin = 30

    # Créez un objet de dessin
    draw = ImageDraw.Draw(img)

    # Déterminer la taille de la police à utiliser
    fontsize = 1  # commencer par une petite taille de police
    font = ImageFont.truetype("font/arial.ttf", fontsize)

    # Augmenter la taille de la police jusqu'à ce que le texte soit trop large
    while draw.textsize(text, font=font)[0] < img.width - 2*margin:
        fontsize += 1
        font = ImageFont.truetype("font/arial.ttf", fontsize)

    # Réduire la taille de la police d'un pas pour ne pas dépasser la largeur de l'image
    fontsize -= 1
    font = ImageFont.truetype("font/arial.ttf", fontsize)

    # Obtenir la largeur et la hauteur du texte
    textwidth, textheight = draw.textsize(text, font)

    # Calculer les coordonnées du centre
    x = (img.width - textwidth) // 2
    y = (img.height - textheight) // 2

    # Ajouter le texte avec un contour
    outline_amount = 3
    shadowcolor = "black"
    fillcolor = "white"

    for adj in range(outline_amount):
        # Déplacer un pixel...
        draw.text((x-adj, y), text, font=font, fill=shadowcolor)
        draw.text((x+adj, y), text, font=font, fill=shadowcolor)
        draw.text((x, y-adj), text, font=font, fill=shadowcolor)
        draw.text((x, y+adj), text, font=font, fill=shadowcolor)

    # Maintenant, dessinez le texte en blanc, mais en utilisant notre copie originale de l'image
    draw.text((x, y), text, font=font, fill=fillcolor)

    # Sauvegardez l'image
    img.save(output_file)






def convert_into_html(text, model="gpt-3.5-turbo-16k"):
    ########################################################################################################################
    """ 
    Function that uses GPT 3.5 to convert a text into an HTML page
    """
    ########################################################################################################################

    prompt = "Formater ce texte en HTML sans modifier le contenu et sans utiliser les balises doc type, head et body mais en ajoutant des titres et en les formatant : \n\n"
    return request_llm(prompt, text, "", model)



  
        


def generate_image(text, output_filename):
    ##############################################
    """
    The 'generate_image' function creates an image using DALL-E 3 based on provided text. 
    It loads environment variables, retrieves the OpenAI API key, initializes the OpenAI client, 
    sends a request to generate the image, and then downloads and saves the image to a specified file.
    """
    ###############################################

    # Charger les variables d'environnement à partir du fichier .env
    load_dotenv()

    # Récupérer la clé API à partir des variables d'environnement
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialiser le client OpenAI avec la clé API
    client = OpenAI(api_key=api_key)

    # Envoyer la requête pour générer l'image
    response = client.images.generate(
        model="dall-e-3",
        prompt=text,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # Récupérer l'URL de la première image
    image_url = response.data[0].url

    # Télécharger l'image
    image_response = requests.get(image_url)

    # Écrire l'image dans un fichier
    with open(output_filename, 'wb') as f:
        f.write(image_response.content)






def filter_urls(rss_urls):
    ########################################################################################################################
    """
    Function that allows to select the RSS feeds to keep
    """
    ########################################################################################################################
    # Affichez chaque url avec son index
    for i, url in enumerate(rss_urls):
        print(f"{i+1}. {url}")

    # Demandez à l'utilisateur de sélectionner les indices des urls à conserver
    selected_indices = input("Veuillez entrer les numéros des urls que vous souhaitez conserver, séparés par des virgules : ")

    # Convertissez les indices sélectionnés en une liste d'entiers
    selected_indices = list(map(int, selected_indices.split(',')))

    # Filtrer rss_urls pour ne conserver que les urls sélectionnées
    rss_urls = [rss_urls[i-1] for i in selected_indices]
    
    return rss_urls



def fetch_and_parse_urls(url):
    """
    Fetch and parse the content of web pages given a list of URLs.

    Args:
    url_list (list): A list of URLs to fetch and parse.

    Returns:
    list: A list where each element is the content of a web page. 
          Links in the text are formatted as 'text [link associated with the text]'.
    """

    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract and format text and links
        text = ''
        for element in soup.descendants:
            if element.name == 'a' and element.get('href') and element.text:
                # Add link in the specified format
                text += f"{element.text} [{element.get('href')}] "
            elif element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Add text elements (paragraphs and headers)
                text += element.get_text() + ' '

        res = text.strip()

    except requests.RequestException as e:
        # In case of request errors, append a descriptive message
        res = f"Error fetching {url}: {e}"

    return str(res)






import requests
import xml.etree.ElementTree as ET

def fetch_and_parse_rss_to_string(rss_url):
    """
    Fetch and parse the content of an RSS feed given its URL and return a string
    with each item's details separated by <br> tags.

    Args:
    rss_url (str): The URL of the RSS feed.

    Returns:
    str: A string representation of the RSS feed items, separated by <br> tags.
    """

    try:
        # Fetch the content from the RSS URL
        response = requests.get(rss_url)
        response.raise_for_status()

        # Parse the XML content
        root = ET.fromstring(response.content)

        # Initialize an empty string to store feed items
        feed_items_str = ""

        # Extract and format RSS feed items
        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else 'No title'
            link = item.find('link').text if item.find('link') is not None else 'No link'
            description = item.find('description').text if item.find('description') is not None else 'No description'

            # Append each item's details to the string, separated by <br>
            feed_items_str += f"Title: {title}<br>Link: {link}<br>Description: {description}<br><br>"

    except requests.RequestException as e:
        # In case of request errors, return a descriptive message
        return f"Error fetching RSS feed from {rss_url}: {e}"

    return feed_items_str
