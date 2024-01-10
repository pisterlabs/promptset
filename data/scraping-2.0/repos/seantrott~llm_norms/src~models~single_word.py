"""
Script for analyzing single words or sentences on a dimension, e.g., iconicity, etc.

"""

import pandas as pd
import numpy as np
import openai
import backoff  # for exponential backoff
import os
from tqdm import tqdm



PROMPTS = {
    'iconicity': "On a scale from 1 (not at all iconic) to 7 (very iconic), how iconic is the word '{word}'?\n\nRating:",
    'cs_norms_perception': "To what extent do you experience '{word}' {sense}?\n\nRating:",
    'cs_norms_action': "To what extent do you experience '{word}' by performing an action with your {sense}?\n\nRating:",
    'concreteness': "On a scale from 1 (abstract and language-based) to 5 (concrete and experience-based), how concrete is the word '{word}'?\n\nRating:",
    'glasgow': ''
}

SYSTEM_PROMPTS = {
    'iconicity': "You are a helpful assistant. Your job is to rate how much each word sounds like what it means.",
    'concreteness': "You are a helpful assistant. Your job is to rate how much each word sounds like what it means.",
    'cs_norms_perception': "You are a helpful assistant. Your job is to rate the sensorimotor dimensions associated with different words.",
    'cs_norms_action': "You are a helpful assistant. Your job is to rate the sensorimotor dimensions associated with different words.",
    'glasgow': "You are a helpful assistant. Your job is to rate the semantic dimensions associated with different words."
    }


SENSE_MAPPINGS = {
    'Hearing': 'by hearing',
    'Touch': 'by feeling through touch',
    'Vision': 'by seeing',
    'Interoception': 'by sensations inside the body',
    'Olfaction': 'by smelling',
    'Taste': 'by tasting'
}

ACTION_MAPPINGS = {
    'Hand_arm': 'Head/Arm',
    'Foot_leg': 'Foot/Leg',
    'Torso': 'Torso',
    'Head': 'Head excluding mouth',
    'Mouth_throat': 'Mouth/throat'
}


GLASGOW_DEFINITIONS = {
    "Arousal": "Arousal is a measure of excitement versus calmness. A word is AROUSING if it makes you feel stimulated, excited, frenzied, jittery, or wide-awake. A word is UNAROUSING if it makes you feel relaxed, calm, sluggish, dull, or sleepy.\nPlease indicate on a scale from 1 to 9 how arousing you think each word is on a scale of VERY UNAROUSING to VERY AROUSING, with the midpoint representing moderate arousal.",
    "Valence": "Valence is a measure of value or worth. A word is POSITIVE if it represents something considered good, whereas a word is NEGATIVE if it represents something considered bad.\nPlease indicate on a scale from 1 to 9 the valence of each word on a scale of VERY NEGATIVE to VERY POSITIVE, with the midpoint representing NEUTRAL.",
    "Dominance": "Dominance is a measure of the degree of control you feel. A word can make you feel DOMINANT, influential, in control, important, or autonomous. Conversely, a word can make you feel CONTROLLED, influenced, cared-for, submissive, or guided.\nPlease indicate on a scale from 1 to 7 how each word makes you feel on a scale of YOU ARE VERY CONTROLLED to YOU ARE VERY DOMINANT, with the midpoint being neither controlled nor dominant.",
    "Concreteness": "Concreteness is a measure of how concrete or abstract something is. A word is CONCRETE if it represents something that exists in a definite physical form in the real world. In contrast, a word is ABSTRACT if it represents more of a concept or idea.\nPlease indicate on a scale from 1 to 7 how concrete you think each word is on a scale of VERY ABSTRACT to VERY CONCRETE, with the midpoint being neither especially abstract nor concrete.",
    "Imageability": "Imageability is a measure of how easy or difficult something is to imagine. A word is IMAGEABLE if it represents something that is very easy to imagine or picture. In contrast, a word is UNIMAGEABLE if it represents something that is very difficult to imagine or picture.\nPlease indicate on a scale from 1 to 7 how imageable you think each word is on a scale of VERY UNIMAGEABLE to VERY IMAGEABLE, with the midpoint being moderately imageable.",
    "Familiarity": "Familiarity is a measure of how familiar something is. A word is very FAMILIAR if you see/hear it often and it is easily recognizable. In contrast, a word is very UNFAMILIAR if you rarely see/hear it and it is relatively unrecognizable.\nPlease indicate on a scale from 1 to 7 how familiar you think each word is on a scale of VERY UNFAMILIAR to VERY FAMILIAR, with the midpoint representing moderate familiarity.",
    "AoA": "A word’s age of acquisition is the age at which that word was initially learned. Please estimate when in your life you think you first acquired or learned each word. That is, try to remember how old you were when you learned each word either in its spoken or written form (whichever came first). The scale is defined as a series of consecutive 2-year periods from the ages of 0 to 12 years, and a final period encompassing 13 years and older.",
    "Size": "Size is a measure of something’s dimensions, magnitude, or extent. A word represents something BIG if it refers to things or concepts that are large. A word represents something SMALL if it refers to things or concepts that are little.\nPlease indicate on a scale from 1 to 7 the semantic size of each word on a scale of VERY SMALL to VERY BIG, with the midpoint being neither small nor big.",
    "Gender": "A word’s gender is how strongly its meaning is associated with male or female behavior. A word can be considered MASCULINE if it is linked to male behavior. Alternatively, a word can be considered FEMININE if it is linked to female behavior.\nPlease indicate on a scale from 1 to 7 the gender associated with each word on a scale of VERY FEMININE to VERY MASCULINE, with the midpoint being neuter (neither feminine nor masculine)."
}

def openai_auth():
    """Try to authenticate with OpenAI."""
    ## Read in key
    with open('src/models/gpt_key', 'r') as f:
        lines = f.read().split("\n")
    org = lines[0]
    api_key = lines[1]
    openai.organization = org # org
    openai.api_key = api_key # api_key


def open_instructions(task = "iconicity"):
    filepath = "data/raw/{task}/instructions.txt".format(task = task)
    with open(filepath, "r") as f:
        instructions = f.read()
    return instructions


def construct_prompt(task, row, **args):
    prompt = PROMPTS[task]
    if task in ["iconicity", "concreteness"]:
        word = row['word']
        return prompt.format(word = word)
    elif task in ['cs_norms_perception', 'cs_norms_action']:
        word = row['word']
        sentence = row['sentence']
        prompt = row['sentence'] + "\n\n" + prompt
        ### TODO: Add sense
        return prompt.format(word = word, sense = args['sense'])
    elif task == "glasgow":
        word = row['word']
        prompt = args['dimension'] + "\n\n" + word + "\n\n" + "Please respond with a single number.\n\nAnswer: "
        return prompt


@backoff.on_exception(backoff.expo, openai.error.RateLimitError) ## Testing this one
def pred_tokens(prompt, system_content, n=10, model="gpt-4"):
    """Get response."""
    output = openai.ChatCompletion.create(
        model = model,
        temperature = 0,
        messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
      max_tokens=10,
      top_p=1
        )

    return output# output['choices'][0]['message']['content']


def main(task="iconicity", model='gpt-4'):
    """
    Run GPT-3 on stims.
    """
    # Authenticate
    openai_auth()

    # Assemble data path
    data_path = "data/raw/{task1}/{task2}.csv".format(task1 = task, task2 = task)
    print(data_path)

    # Print config for user
    print(f"Task: \t{task}")
    print(f"Model: \t{model}")

    
    # Read in stims
    df_stimuli = pd.read_csv(data_path)
    print(f"No. stimuli: \t{len(df_stimuli)}")
    print(df_stimuli.head(5))

    # Get instructions
    instructions = open_instructions(task = task)

     # Ensure directory exists
    dirpath = "data/processed/{task}".format(task = task)
    # Create dataframe
    os.makedirs(dirpath, exist_ok=True)
    filename = f"{dirpath}/{task}_{model}.csv".format(task = task)
    file_txt = f"{dirpath}/{task}_{model}.txt".format(task = task)


    answers = []
    col_name = "{m}_response".format(m = model)

    df_stimuli = df_stimuli[540 :len(df_stimuli)]

    for index, row in tqdm(df_stimuli.iterrows(), total=df_stimuli.shape[0]):
        
        
        ### Left off at 84 for cs_norms_perception
        if task == 'cs_norms_perception':
            
            for sense, q in SENSE_MAPPINGS.items():
                prompt = construct_prompt(task = task, row = row, sense = q)
                system_prompt = SYSTEM_PROMPTS[task]
    
                instructions_plus_prompt = instructions + "\n\n" + prompt

                response = pred_tokens(instructions_plus_prompt, system_content = system_prompt, n = 3)
                extracted_response = response['choices'][0]['message']['content']

                row[col_name] = extracted_response 

                answers.append(extracted_response)


                with open(file_txt, "a") as f:
                    f.write("{word},{response},{sense},{sentence}\n".format(word = row['word'],
                                                response = extracted_response, sense=sense, sentence=row['sentence']))

        elif task == 'cs_norms_action':
            
            for sense, q in ACTION_MAPPINGS.items():
                prompt = construct_prompt(task = task, row = row, sense = q)
                system_prompt = SYSTEM_PROMPTS[task]
    
                instructions_plus_prompt = instructions + "\n\n" + prompt

                response = pred_tokens(instructions_plus_prompt, system_content = system_prompt, n = 3)
                extracted_response = response['choices'][0]['message']['content']

                row[col_name] = extracted_response 

                answers.append(extracted_response)


                with open(file_txt, "a") as f:
                    f.write("{word},{response},{sense},{sentence}\n".format(word = row['word'],
                                                response = extracted_response, sense=sense, sentence=row['sentence']))

        elif task == 'glasgow':
            
            for dimension, q in GLASGOW_DEFINITIONS.items():


                system_prompt = SYSTEM_PROMPTS[task]
                
                # Create Glasgow prompt
                prompt = construct_prompt(task = task, row = row, dimension = q)

                # Combine with complete instructions
                instructions_plus_prompt = instructions + "\n\n" + prompt

                response = pred_tokens(instructions_plus_prompt, system_content = system_prompt, n = 3)
                extracted_response = response['choices'][0]['message']['content']


                to_write = "{word},{response},{dimension}\n".format(word = row['word'],
                                                response = extracted_response, dimension=dimension)
                with open(file_txt, "a") as f:
                    f.write(to_write)


        else:
            prompt = construct_prompt(task = task, row = row)
            system_prompt = SYSTEM_PROMPTS[task]
            
            instructions_plus_prompt = instructions + "\n\n" + prompt

            response = pred_tokens(instructions_plus_prompt, system_content = system_prompt, n = 3)
            extracted_response = response['choices'][0]['message']['content']

            row[col_name] = extracted_response 

            answers.append(extracted_response)


            with open(file_txt, "a") as f:
                f.write("{word},{response}\n".format(word = row['word'],
                                            response = extracted_response))


    # Create dataframe
    if task not in ['cs_norms_perception', 'cs_norms_action', 'glasgow']:
        col_name = "{m}_response".format(m = model)
        df_stimuli[col_name] = answers 

        # Ensure directory exists
        dirpath = "data/processed/{task}".format(task = task)
        os.makedirs(dirpath, exist_ok=True)

        # Save file
        filename = f"{dirpath}/{task}_{model}.csv".format(task = task)
        df_stimuli.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--task", type=str, dest="task",
                        default="glasgow")
    parser.add_argument("--m", type=str, dest="model",
                        default="gpt-4")
    
    args = vars(parser.parse_args())
    main(**args)