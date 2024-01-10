import time
import random
from decimal import Decimal
from time import sleep

import os
import openai
#import nltk
from modules import create_vars
#from nltk.probability import FreqDist
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt

#from modules import logger
from modules.logger import setup_logger

#start logger
logger = setup_logger("poem_gen")
logger.debug("Logger is set up and running.")

# removed nltk to try to speed things up
#nltk.download('wordnet')
#from nltk.corpus import wordnet as wn

openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def poem_step_1(creative_prompt, persona, randomness_factor):
            completion = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": persona + " You write a poem. You can use up to 25 characters per line."},
                    {"role": "user", "content": "Produce a haiku inspired by the following words: " + creative_prompt + ""},
                    #{"role": "user", "content": "Explain why you created the poem the way you did."},
                ], 
                temperature=(randomness_factor * 2),
                max_tokens=500,
            )

            if completion['choices'][0]['message']['role'] == "assistant":
                step_1_poem = completion['choices'][0]['message']['content'].strip()
            else:
                step_1_syscontent = api_response['system'].strip()  # put into a var for later use 

            #logger.info(f"poem_step_1 Prompt tokens: {completion['usage']['prompt_tokens']}")
            #logger.info(f"poem_step_1 Completion tokens: {completion['usage']['completion_tokens']}")
            #logger.info(f"poem_step_1 Total tokens: {completion['usage']['total_tokens']}")
            return step_1_poem

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def poem_step_2(persona, randomness_factor, step_1_poem, abstract_concept):
            completion = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": persona + " You write a poem based on parameters provided as well as input text to build on."},
                    {"role": "user", "content": "Create a new poem based on the input text that is two to four lines long with the following parameters. The chosen abstract concept is: " + abstract_concept + ". Revise the input text to subtly weave in the chosen concept."},
                    {"role": "user", "content": "Input text: " + step_1_poem},
                ],
                temperature=(randomness_factor * 2),
                max_tokens=500,
            )

            if completion['choices'][0]['message']['role'] == "assistant":
                step_2_poem = completion['choices'][0]['message']['content'].strip()
            else:
                step_2_syscontent = completion['system'].strip()  # put into a var for later use 

            #logger.info(f"poem_step_2 Prompt tokens: {completion['usage']['prompt_tokens']}")
            #logger.info(f"poem_step_2 Completion tokens: {completion['usage']['completion_tokens']}")
            #logger.info(f"poem_step_2 Total tokens: {completion['usage']['total_tokens']}")
            return step_2_poem

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def poem_step_3(persona, randomness_factor, step_2_poem):
            completion = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": persona + "You generate poetry that is up to four lines long."},
                    {"role": "user", "content": "Create a new poem based on the input text that is up to four lines long with the following parameters. Introduce variation to reduce overall consistency in tone, language use, and sentence structure."},
                    {"role": "user", "content": "Input text: " + step_2_poem},
                ],
                temperature=(randomness_factor * 2),
                max_tokens=500,
            )

            if completion['choices'][0]['message']['role'] == "assistant":
                step_3_poem = completion['choices'][0]['message']['content'].strip()
            else:
                step_3_syscontent = api_response['system'].strip()  # put into a var for later use 

            #logger.info(f"poem_step_3 Prompt tokens: {completion['usage']['prompt_tokens']}")
            #logger.info(f"poem_step_3 Completion tokens: {completion['usage']['completion_tokens']}")
            #logger.info(f"poem_step_3 Total tokens: {completion['usage']['total_tokens']}")

            return step_3_poem

def api_poem_pipeline(creative_prompt, persona, randomness_factor, abstract_concept):
    logger.debug(f"creative_prompt: {creative_prompt}")
    step_1_poem = poem_step_1(creative_prompt, persona, randomness_factor)
    logger.info (f"step_1_poem:\n{step_1_poem}")
    #step_2_poem = poem_step_2(persona, randomness_factor, step_1_poem, abstract_concept)
    #logger.debug (f"step_2_poem:\n{step_2_poem}")
    #step_3_poem = poem_step_3(persona, randomness_factor, step_2_poem)
    #logger.info (f"step_3_poem:\n{step_3_poem}")
    return step_1_poem

def parse_response(entropy):
    # set a randomness factor between 0 and 1. Placeholder, will be logic for the buttons
    randomness_factor = entropy
    # this part of the code goes WAY too slow. Removing the use of nltk for initial generation of the creative_prompt words
    #creative_prompt = create_vars.gen_creative_prompt(create_vars.gen_random_words(randomness_factor), randomness_factor)
    creative_prompt = create_vars.gen_creative_prompt_api(entropy)
    abstract_concept = create_vars.get_abstract_concept()
    persona = create_vars.build_persona()
    lang_device = create_vars.get_lang_device()

    logger.debug(f"persona is: {persona}")
    logger.debug(f"lang_device is: {lang_device}")
    logger.debug(f"abstract_concept is: {abstract_concept}")
    logger.debug(f"randomness factor is: {randomness_factor}")

    logger.debug(f"==========================")
    logger.debug(f"creative_starting_prompt: {creative_prompt}")

    poem_result = api_poem_pipeline(creative_prompt, persona, randomness_factor, abstract_concept)
    logger.debug(f"poem result:\n{poem_result}")

    print("-" * 30)
    logger.debug("poem_gen completed successfully")
    return poem_result

#if __name__ == "__main__":
#    parse_response()


    # add tokens cost logging
    # remove the explanation for the poems its too much, useless tokens spend 
    # add proper retry logic again... I guess. Just add it to the whole thing. 

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

    ## Parking lot - not used
        #4: {"role": "user", "content": "Step 4: Create a new poem that is two to four lines long with the following parameters: Revise the selected poem to achieve a poetic goal of expressing vivid imagery or evoking a specific emotion."},
        #5: {"role": "user", "content": "Step 5: Create a new poem that is two to four lines long with the following parameters: Consider how you could use this linguistic device: "  + lang_device + ". Revise the poem to incorporate the linguistic device"},
        
