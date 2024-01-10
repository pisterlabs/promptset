from agent_descriptors import *
from agents_information import Agent
import openai 
import os

openai.api_key = "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"
STYLE = "Art Style: Generate an Pixel Art with the art style resembling the character sprites from the game Stardew Valley"


def create_agent_art(agent_object):
    agent_text_description = []

    agent_text_description.append(STYLE)

    agent_text_description.append("First Name: ")
    agent_text_description.append(agent_object.first_name)

    agent_text_description.append("Last Name: ")
    agent_text_description.append(agent_object.last_name)

    agent_text_description.append("Hair Color: ")
    agent_text_description.append(agent_object.hair_color)

    agent_text_description.append("Skin Tone: ")
    agent_text_description.append(agent_object.skin_tone)

    agent_text_description.append("Eye Color: ")
    agent_text_description.append(agent_object.eye_color)

    agent_text_description.append("Accessories: ")
    agent_text_description.append(agent_object.accesory)

    agent_text_description.append("Pronouns: ")
    agent_text_description.append(agent_object.gender)

    agent_text_description.append("Occupation: ")
    agent_text_description.append(agent_object.occupation)

    agent_sentence_description = " ".join(agent_text_description)

    
l
    response = openai.Image.create(prompt=agent_sentence_description, n=1, size='1024x1024')

    image_url = response[ 'data' ][0]['url']
    #print(image_url)
    return image_url
