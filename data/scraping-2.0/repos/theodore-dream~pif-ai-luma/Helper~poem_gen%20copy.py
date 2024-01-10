import datetime
import random
from decimal import Decimal
from time import sleep
import uuid
import tiktoken

import logging
import datetime
import os
import openai
import nltk
from modules import create_vars
from nltk.probability import FreqDist

from modules.logger import setup_logger

#start logger
logger = setup_logger("poem_gen")
logger.info("Logger is set up and running.")

nltk.download('wordnet')
from nltk.corpus import wordnet as wn

logging.basicConfig(level=logging.INFO)
openai.api_key = os.getenv("OPENAI_API_KEY")

def api_create_poem(steps_to_execute, creative_prompt, persona, lang_device, abstract_concept, randomness_factor):

    all_steps = {
        0: {"role": "system", "content": persona + " You write poems. Explicity state what step you are on and explain the changes made for each step before proceeding to the next step."},
        1: {"role": "user", "content": "Step 1: Produce three different versions of a poem inspired by the following: " + creative_prompt + ". Each poem can be three or four lines long. Each version should have a different structure - rhyme, free verse, sonnet, haiku, etc."},
        2: {"role": "user", "content": "Step 2: The chosen abstract concept is: " + abstract_concept + ". Next you evaluate the revisions and determine which most closely has a deep connection to then chosen concept, or could most elegantly be modified to fit the concept."},
        3: {"role": "user", "content": "Step 3: Create a new poem that is two to four lines long with the following parameters: Revise the selected poem to subtly weave in the chosen concept."},
        #4: {"role": "user", "content": "Step 4: Print five equals signs."},
        #5: {"role": "user", "content": "Step 5: Create a new poem that is two to four lines long with the following parameters: Introduce variation to reduce overall consistency in tone, language use, and sentence structure."},
        #4: {"role": "user", "content": "Step 4: Create a new poem that is two to four lines long with the following parameters: Revise the selected poem to achieve a poetic goal of expressing vivid imagery or evoking a specific emotion."},
        #5: {"role": "user", "content": "Step 5: Create a new poem that is two to four lines long with the following parameters: Consider how you could use this linguistic device: "  + lang_device + ". Revise the poem to incorporate the linguistic device"},
        
    }

    steps_for_api = [all_steps[step] for step in steps_to_execute]
    i = 0
    for i, step in enumerate(steps_for_api):
        logger.debug("Step %i: %s", i+1, step)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=steps_for_api,
        max_tokens=3600,
        n=1,
        stop=None,
        temperature=(1.2),
    )

    
    # print information about api call
    logging.debug(f"persona: {persona}")
    logging.debug(f"abstract_concept: {abstract_concept}")
    logging.debug(f"creative_prompt: {creative_prompt}")
    return response

def parse_response():
    # set a randomness factor between 0 and 1. Placeholder, will be logic for the buttons
    randomness_factor = 0.7
    creative_prompt = create_vars.gen_creative_prompt(create_vars.gen_random_words(randomness_factor), randomness_factor)
    abstract_concept = create_vars.get_abstract_concept()
    persona = create_vars.build_persona()
    lang_device = create_vars.get_lang_device()
    logger.debug(f"lang_device is: {lang_device}")
    logger.debug(f"abstract_concept is: {abstract_concept}")
    logger.debug(f"randomness factor is: {randomness_factor}")
    logger.debug(f"==========================")
    logger.debug(f"running pif_poetry_generator with prompt: {creative_prompt}")

    print("creative prompt: " + str(creative_prompt))


    # set the number of steps you want here
    #api_response = api_create_poem([0, 1, 2, 3],creative_prompt, persona, lang_device, abstract_concept, randomness_factor)
    #if api_response['choices'][0]['message']['role'] == "assistant":
    #    api_response_content = api_response['choices'][0]['message']['content'].strip()
    #else:
    #    api_response_syscontent = api_response['system'].strip()  # put into a var for later use 
    #print("-" * 30)

    #logger.info(f"Prompt tokens: {api_response['usage']['prompt_tokens']}")
    #logger.info(f"Completion tokens: {api_response['usage']['completion_tokens']}")
    #logger.info(f"Total tokens: {api_response['usage']['total_tokens']}")

    #logger.info(f"api_response_content: {api_response_content}")

    print("-" * 30)
    logger.debug("poem_gen completed successfully")
    #return api_response_content

if __name__ == "__main__":
    parse_response()

    # current issue is that there are 6 steps, 7 including the persona, and its too much complexity for the api to handle all of it
    # on the other hand the results are really good it seesm to only be going to step 3, maybe at this point I need to focus on
    # either I just want to output the final poem directly from the api but that could get dicey at different temperatures
    # alternatively I could use logic to modify the output from the api to get the final poem only. Will need to experiment on diff temps. 

    ## variables overview - goals
    ## build_persona - bad, needs more work / further testing, only seems to perhaps be effective with very few steps, 1-2 steps tops 
    ## get_random_words - happy with number of words because I modifed the api call to generate shorter sentence 
    ## get_abstract_concept - good, using a list and nltk to find synonyms
    ## delayed - poetic_goal ? - experimenting with this, seems like its stopping at step 3 and its step 4 now
    ## delayed - get_lang_device - seems good but needs more testing, might need to push this off for now, might be unnecesary, too much logic in a single prompt 
    ## delayed - ?incorporate the lyrics api into the poetry generator? prob save for a stage 2 

    ## other assorted ideas
    ## ====================
    ## seed the database with a script that pulls from nltk and compiles lists of words
    ## could use nltk to find synonyms for the words in the abstract concept list to seed that to the DB
    ## could find a list of meme related words somewhere, create categories, tags, individual columns or tables, etc.