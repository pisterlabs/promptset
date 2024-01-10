# -*- coding: utf-8 -*-
'''
Filename: generatechatcompletion.py
Author: Michel Levy Provencal
Description: This file defines two functions, generate_chat_completion and generate_chat, that use OpenAI's API to generate chat responses. It uses environmental variables for API keys and includes a default model of "gpt-4" if no model is specified in the function parameters.
'''

import openai  # Import the openai API package
import os  # Import os module for interacting with the operating system
from dotenv import load_dotenv  # Import dotenv module for loading .env files
import lib__anthropic
import lib__hfmodels
from huggingface_hub import InferenceClient


# Load the environment variables from the .env file
load_dotenv(".env")


# Set the OpenAI API key from the environment variables
openai.api_key = os.environ['OPEN_AI_KEY']



def extract_context(text, model):
    """
    Extraire un contexte de 'text' basé sur la limite spécifiée.

    Si la longueur de 'text' est inférieure à 'limit', renvoie le texte complet.
    Sinon, renvoie une combinaison des premiers et derniers caractères de 'text'
    avec ' [...] ' inséré au milieu pour indiquer la coupure.

    :param text: La chaîne de caractères à traiter.
    :param limit: La limite de longueur pour le contexte extrait.
    :return: La chaîne de caractères traitée.
    """
    token_nb = 2000
    
    if model == "claude-2":
        token_nb = 100000 
    if model == "gpt-4":
        token_nb = 8000
    if model == "gpt-4-1106-preview":
        token_nb = 128000
    if model == "gpt-3.5-turbo-16k": 
        token_nb = 16000
    if model == "hf":
        token_nb = 2000  
    if model == "mistral":
        token_nb = 2000      
    
    if token_nb > 2000:
        limit = (int(token_nb)*2) - 4000
    else:
        limit = int((int(token_nb)*2)/2)
    
    if len(text) < limit:
        return text
    else:
        half_limit_adjusted = limit // 2 - 4
        return text[:half_limit_adjusted] + ' [...] ' + text[-half_limit_adjusted:]


# Function to generate chat completions
def generate_chat_completion(consigne, texte, model="gpt-4", model_url=os.environ['MODEL_URL']):
    texte = extract_context(texte, model)
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    prompt = str(consigne + " : " + texte)  # Construct the prompt from the given consigne and texte

     
    if model == "claude-2":
        response = lib__anthropic.generate_chat_completion_anthropic(consigne, texte, model)
        for content in response:
            print(content)
            yield content
            
    else:
            
        if model == "hf":
            #prompt = str(consigne + "\n Le texte : ###" + texte + " ###\n")  # Construct the prompt from the given consigne and texte
            prompt = str(consigne + "\n" + texte)  # Construct the prompt from the given consigne and texte
            prompt = "<s>[INST]" + prompt + "[/INST]"
            print("Prompt : " + prompt + "\n")
            print("Model URL : " + model_url + "\n" + "HF TOKEN : " + os.environ['HF_API_TOKEN'] + "\n")
            
            client = InferenceClient(model_url, token=os.environ['HF_API_TOKEN'])
            response = client.text_generation(
                prompt,
                max_new_tokens=1024,
                stream=True
            )
            
            for result in response:
                yield result

        
        else:
            
            # Use OpenAI's Chat Completion API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': "Je suis un assistant parlant parfaitement le français et l'anglais capable de corriger, rédiger, paraphraser, traduire, résumer, développer des textes."},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
                stream=True
            )
            
            for message in completion:
            # Vérifiez ici la structure de 'chunk' et extrayez le contenu
            # La ligne suivante est un exemple et peut nécessiter des ajustements
            
                if message.choices[0].delta.content: 
                    text_chunk = message.choices[0].delta.content 
                    print(text_chunk, end="", flush="true")
                    yield text_chunk
                    
                


                        
                        
                        
                        
                        

# Function to generate chat
def generate_chat(consigne, texte, system="", model="gpt-4", model_url=os.environ['MODEL_URL']):
    prompt = str(consigne + " : " + texte)  # Construct the prompt from the given consigne and texte
    # Call the OpenAI API to create a chat

    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    texte = extract_context(texte, model)
    
    if model == "claude-2":
        response = lib__anthropic.generate_chat_completion_anthropic(consigne, texte, model)
        for content in response:
            print(content)
            yield content
    
    else:
        if model == "hf":
            prompt = str(consigne + "\n" + texte)  # Construct the prompt from the given consigne and texte
            #prompt = str(consigne + "\n Le texte : ###" + texte + " ###\n")  # Construct the prompt from the given consigne and texte
            prompt = "<s>[INST]" + prompt + "[/INST]"
            
            print("Prompt : " + prompt + "\n")
            print("Model URL : " + model_url + "\n" + "HF TOKEN : " + os.environ['HF_API_TOKEN'] + "\n")
            
            client = InferenceClient(model_url, token=os.environ['HF_API_TOKEN'])
            response = client.text_generation(
                prompt,
                max_new_tokens=1024,
                stream=True
            )
            
            for result in response:
                yield result


        else:   
            #Model = gpt-4-1106-preview 
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=True
            )

            for message in completion:
            # Vérifiez ici la structure de 'chunk' et extrayez le contenu
            # La ligne suivante est un exemple et peut nécessiter des ajustements
            
                if message.choices[0].delta.content: 
                    text_chunk = message.choices[0].delta.content 
                    print(text_chunk, end="", flush="true")
                    yield text_chunk
                    
