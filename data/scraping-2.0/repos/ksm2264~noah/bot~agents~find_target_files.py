# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:50:05 2023

@author: karl
"""

import openai

from bot.cli import get_file_contents

import logging
import sys

handler = logging.StreamHandler(sys.stderr)
logging.basicConfig(level=logging.INFO, handlers=[handler])

gpt_model = 'gpt-3.5-turbo'

def find_target_files(app_summary, file_list, feature_request):
    
    system_message = '''
    given an app summary, a feature request, and file contents for one file,
    you respond with yes or no to indicate whether this file needs modification to implement a feature.
    Respond with "yes" or "no" ONLY, no other text
    '''

    yes_token = 3919
    no_token = 8505

    max_prob = 10

    logit_bias = {
        yes_token:max_prob,
        no_token:max_prob,
    }

    target_file_names = []

    for file_name in file_list:

        logging.info(f'processing: {file_name}')

        user_message = f'''
        Do I need to modify this file to implement this feature?
        app_summary: {app_summary}, feature: {feature_request}, file contents: {get_file_contents(file_name)}
        '''
        # print(feature_request)
        messages = [
            {"role":"system",
            "content":system_message},
            {"role":"user",
            "content":user_message}
        ]

        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=messages,
            logit_bias = logit_bias
        )

        response_dict = response.to_dict()
        raw_text = response_dict['choices'][0]['message']['content']
        raw_text = raw_text.strip('.').lower()
        logging.info(f'result: {raw_text}')

        if 'yes' in raw_text:
            target_file_names.append(file_name)
        else:
            pass


    return target_file_names
    