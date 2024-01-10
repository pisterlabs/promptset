import os
import openai
from dotenv import load_dotenv
import random

load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')

INSTRUCTIONS = ['Rephrase this', 'Reword this']

TEXT_CHUNKS = ['There are two regular factors: color and word.',
               'The color factor consists of four levels: "red", "green", "blue", "brown".The word factor also consists of the four levels: "red", "green", "blue", "brown".',
               'We counterbalanced the color factor with the word factor.',
               'The experiment sequences were generated subject to the constraint that no congruent trials were included.',
               'All experiment sequences contained at least 20 trials and were sampled uniformly from the space of all counterbalanced sequences.']


def gpt3_paraphrase(text, instr):
    response = openai.Edit.create(
        model='text-davinci-edit-001',
        input=text,
        instruction=instr,
    )
    return response.choices[0]['text']


chunk = random.choice(TEXT_CHUNKS)
instruction = random.choice(INSTRUCTIONS)

new_text_prompt = gpt3_paraphrase(chunk, instruction)

f = open('altered_text_prompts.txt', 'a')
f.write('\nInstruction:\t')
f.write(instruction)
f.write('\nOriginal:\t')
f.write(chunk)
f.write('\nAltered:\t')
f.write(new_text_prompt)
f.write('\n*********')
f.close()
