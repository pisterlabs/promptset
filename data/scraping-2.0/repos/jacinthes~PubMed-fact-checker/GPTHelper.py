import openai
from time import time
import os
import logging
import streamlit as st

openai.api_key = st.secrets['openai_API_key']


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def gpt35_rephrase(fact):
    # Dynamically generate the prompt to rephrase the fact as a PubMed query using GPT3.5
    prompt = open_file('prompts/gpt35_rephrase.txt').replace('<<FACT>>', fact)
    try:
        response = openai.Completion.create(
          model='text-davinci-003',
          prompt=prompt,
          max_tokens=250,
          temperature=0
        )
        response = response['choices'][0]['text'].strip()
        filename = '%s_gpt3.txt' % time()

        # Create the logs folder if it does not exist
        if not os.path.exists('gpt3_rephrase_logs'):
            os.makedirs('gpt3_rephrase_logs')

        # Save the whole prompt and the response so that we can inspect it when necessary
        with open('gpt3_rephrase_logs/%s' % filename, 'w', encoding="utf-8") as outfile:
            outfile.write('PROMPT:\n\n' + prompt + '\n\n###############\n\nRESPONSE:\n\n' + response)

        return response

    except Exception as e:
        logging.error('Error communicating with OpenAI (rephrase): ', exc_info=e)


def gpt35_check_fact(evidence, fact):
    # Dynamically generate the prompt to check the fact against the given PubMed article conclusion/abstract
    prompt = open_file('prompts/gpt35_fact_check.txt').replace('<<EVIDENCE>>', evidence).replace('<<HYPOTHESIS>>', fact)
    try:
        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=prompt,
          max_tokens=3,  # Don't need more for Entails/Contradicts/Undetermined
          temperature=0
        )
        response = response['choices'][0]['text'].strip()
        response = response.replace('.', '')
        filename = '%s_gpt3.txt' % time()

        if not os.path.exists('gpt3_factchecking_logs'):
            os.makedirs('gpt3_factchecking_logs')

        with open('gpt3_factchecking_logs/%s' % filename, 'w', encoding='utf-8') as outfile:
            outfile.write('PROMPT:\n\n' + prompt + '\n\n###############\n\nRESPONSE:\n\n' + response)

        return response

    except Exception as e:
        logging.error('Error communicating with OpenAI (check_fact): ', exc_info=e)


def gpt35_turbo_rephrase(fact):
    # Dynamically generate the prompt to rephrase the fact as a PubMed query using GPT3.5 turbo - lower cost than 3.5
    prompt = open_file('prompts/gpt35_rephrase.txt').replace('<<FACT>>', fact)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'user',
                 'content': prompt}
              ]
        )
        response = response['choices'][0]['message']['content'].strip()
        filename = '%s_gpt3.txt' % time()

        if not os.path.exists('gpt35_rephrase_logs'):
            os.makedirs('gpt35_rephrase_logs')

        with open('gpt35_rephrase_logs/%s' % filename, 'w', encoding="utf-8") as outfile:
            outfile.write('PROMPT:\n\n' + prompt + '\n\n###############\n\nRESPONSE:\n\n' + response)

        return response

    except Exception as e:
        logging.error('Error communicating with OpenAI (gpt35_rephrase): ', exc_info=e)
