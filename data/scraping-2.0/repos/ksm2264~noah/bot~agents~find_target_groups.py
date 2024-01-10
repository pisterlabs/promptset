# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:50:05 2023

@author: karl
"""

import openai
import json

from bot.agents.clean_json import json_cleaner

gpt_model = 'gpt-3.5-turbo'

def find_groups(app_summary, target_categories, feature_request):
    
    system_message = '''
    determine which categories of files should be changed to implement a feature request.
    format like this:
    '["category_1", "category_2"]'
    Please make sure you are only responding with that format, and that it is valid
    '''

    user_message = f'app_summary: {app_summary}, existing groups: {target_categories}, feature_request: {feature_request}'
    # print(feature_request)
    messages = [
        {"role":"system",
         "content":system_message},
        {"role":"user",
         "content":user_message}
    ]

    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages
    )

    response_dict = response.to_dict()
    raw_text = response_dict['choices'][0]['message']['content']

    try:
        categories = json.loads(raw_text)
    except:
        categories = json.loads(json_cleaner(raw_text))

    return categories
    