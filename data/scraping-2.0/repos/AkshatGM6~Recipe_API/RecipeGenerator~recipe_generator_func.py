import os 
from .apikey import apikey
import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

# st.title('Recipe Generator')

def recipe_generator(preferredCharacterisitcs):
    # our prompt that will be sent to LLM
    recipe_template = PromptTemplate(

        input_variables=['recipe_characterisitcs'],
        template=  """
            Give me a recipe with following characteristics.
            
            Preferred Ingredients: {recipe_characterisitcs}

            If no recipe is found, return "Could not generate any recipe. Please change some constrains". If recipes are found then send them in following template - "Recipe Name : ", "Recipe Ingredients : ", "Recipe Instructions : " Follow this template only and send the response.
        """
    )

    # we are limiting maximum tokens to 180 to control cost
    llm = OpenAI(temperature=0.9, max_tokens=300)

    # single chain as of now. 
    # We can keep the history of our calls. Not used yet. 
    recipe_chain = LLMChain(llm = llm, prompt=recipe_template, verbose=True, memory=ConversationBufferMemory(max_len=400))
    
    response = recipe_chain.run(recipe_characterisitcs = preferredCharacterisitcs)

    # split response string into lines
    response_lines = response.split('\n')

    recipe_dict = {}

    current_key = ""
    for line in response_lines:
        if line.strip() == "":
            continue
        # check if line is a title (followed by ':')
        if ':' in line:
            current_key, value = line.split(':')
            current_key = current_key.strip()
            value = value.strip()
            recipe_dict[current_key] = value
        else:
            # if not a title, it is a continuation of a list
            recipe_dict[current_key] += "\n" + line.strip()
    print(recipe_dict)
    return recipe_dict

        # print(response) # returned as string 