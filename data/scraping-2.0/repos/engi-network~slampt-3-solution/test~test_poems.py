import openai
import random
import os
import pronouncing
import json
import pytest
import string

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_random_poem_topics():
    return [
        random.choice(open("nouns.txt").read().split()),
        random.choice(open("nouns.txt").read().split()),
        random.choice(open("nouns.txt").read().split()),
        random.choice(open("nouns.txt").read().split()),
        random.choice(open("nouns.txt").read().split()),
    ]

def run_random_poem_completion(rhyming_pattern, completion_parameters):
    instructions_prompt = completion_parameters['prompt']
    topics = ', and'.join(', '.join(get_random_poem_topics()).rsplit(',', 1))
    prompt = instructions_prompt + "\n\nWrite a poem about " + topics + " in a " + rhyming_pattern + " rhyming pattern.\n\nPOEM:\n\n"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=completion_parameters['temperature'],
        top_p=completion_parameters['top_p'],
        frequency_penalty=completion_parameters['frequency_penalty'],
        presence_penalty=completion_parameters['presence_penalty'],
        stop=completion_parameters['stop']
    )
    print(response['choices'][0]['text'])
    return response['choices'][0]['text']


def test_aabb():
    # must consistently predict
    for _ in range(10):
        parameters_file = open('AABB-params.json')
        parameters = json.load(parameters_file)
        poem = run_random_poem_completion('AABB', parameters)
        cleaned_poem = poem.translate(str.maketrans('', '', string.punctuation)).strip()
        cleaned_poem_lines = cleaned_poem.splitlines()
        for previous_line, line in zip(cleaned_poem_lines[::2], cleaned_poem_lines[1::2]):
            line_rhyming_word = line.split()[-1]
            previous_line_rhyming_word = previous_line.split()[-1]
            assert previous_line_rhyming_word in pronouncing.rhymes(line_rhyming_word)

@pytest.mark.skip(reason="Implement ABAB check.")
def test_abab():
    print('TODO')

