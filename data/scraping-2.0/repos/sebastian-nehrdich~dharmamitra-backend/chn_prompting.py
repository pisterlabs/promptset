
import pandas as pd
import time
import requests
import json
import chinese_converter as cc
import openai
from translator import KoTranslator
from prompting import OpenAiChatGPT3, OpenAiChatGPT4, OpenAiDavinci
from indexing import ChnIndex


OpenAiModel = OpenAiChatGPT3()
Index = ChnIndex()

dictionary_path = "data/dict.json"

Translator = KoTranslator()


def create_samples(query):
    query = " ".join(query)
    results = Index.search(query, 15)
    result_string = ""
    for result in results:
        result_string += "Buddhist Chinese " + result[0].replace(" ", "") + " means English: " + result[1] + "\n"
    return result_string

def create_dictionary(path):
    return_list = []
    json_dict = json.load(open(path, 'rb'))
    for key in json_dict:
        if len(key) >= 2:
            return_list.append([cc.to_simplified(key), json_dict[key][1]])
    # sort return_list with longest item on top
    return_list.sort(key=lambda x: len(x[0]))
    return return_list[::-1]


main_list = create_dictionary(dictionary_path)

def create_dictionary_entries(chn_string):
    result = ""
    known_items = []
    for item in main_list:
        if item[0] in chn_string:
            got_flag = False
            for known_item in known_items:
                if item[0] in known_item:
                    got_flag = True
            if not got_flag:
                result += item[0] + " means " + item[1] + "\n"
                known_items.append(item[0])
    return result


def construct_prompt(query, explanation_level):    
    query = cc.to_simplified(query)
    prompt = create_dictionary_entries(query)    
    prompt += " The following example sentences might help you: \n" + create_samples(query) + "\n"
    prompt += "The Korean translation might help you: " + Translator.translate(query) + "\n"
    prompt += "\nTranslate the following Classical Buddhist Chinese into English: " + query + "\n"
    print("PROMPT", prompt)
    if explanation_level == 0:
        prompt += "Output only the English translation.\n"
    elif explanation_level == 1:
        prompt += "Output only the English translation together with an explanation of the meaning of the Chinese words.\n"
    elif explanation_level == 2:
        prompt += "Output only the English translation together with an explanation of its structure and the meaning of the Chinese words.\n"    
    
    return prompt


def get_translation_chn(input_sentence, explanation_level=0):
    prompt = construct_prompt(input_sentence, explanation_level)
    response = OpenAiModel.get_translation(prompt)
    return response
