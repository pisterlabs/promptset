## This page is for functions related to generating training materials for the generated cocktails
# Once they have confirmed that they are happy with the cocktail, they will be able to generate training materials for the cocktail
# By sending the text of the recipe to GPT-3.5.

# Initial imports
import os
from dotenv import load_dotenv
load_dotenv()
import openai
import requests

# Load the openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

#Establish the function to submit the text of the recipe to the model and generate the training guide
def generate_training_guide(recipe_text):
    
    # Create the messages for the model
    messages = [
        {"role" : "system", "content" : f"Generate a detailed training guide for a cocktail recipe for a restaurant pre-shift staff education,\
          focusing on the history and specifics of the ingredients, the techniques used, and the flavor profile of the drink. The recipe is as\
          follows: {recipe_text}."
        },
    ]
    # Define the parameters for the API call
    params = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 750,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "temperature": 1,
        "top_p": 0.9,
        "n": 1,
    }
    
    # Call the OpenAI API and handle exceptions
    try:
        response = openai.ChatCompletion.create(**params)
    except (requests.exceptions.RequestException, openai.error.APIError):
        params["model"] = "gpt-3.5-turbo-0301"
        response = openai.ChatCompletion.create(**params)

    # Return the the response as training guide
    guide = response.choices[0].message.content

    return guide

