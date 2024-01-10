# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:04:54 2023

@author: karl
"""

import openai

from bot.cli import get_file_contents

gpt_model = "gpt-3.5-turbo"

def write_code(feature_description, files_to_change):

    system_message = '''
    You are a programming expert, respond with new files to implement a feature.
    In your response alternate file names and file content like this:
    @@@file_name1@@@file_content1@@@file_name2@@@file_content_2@@@ etc.
    Only respond with that format. Make sure to start and end with a @@@
    '''
    
    files_dict = {}
    
    for file in files_to_change:
        files_dict[file] = get_file_contents(file)

    user_message = f'''Please implement this feature, responding
    with the desired @@@ format as instructed. Please make sure to format the code correctly.
    The feature is: {feature_description}, the files to change are: {files_dict}
    '''
    # print(feature_description)
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

    split_txt = raw_text.split('@@@')

    if split_txt[0]=='':
        split_txt=split_txt[1:]

    if split_txt[-1]=='':
        split_txt = split_txt[:-1]

    changed_files = {}
    
    for i in range(0, len(split_txt), 2):
        key = split_txt[i]
        value = split_txt[i+1]
        changed_files[key] = value
            
    return changed_files
    