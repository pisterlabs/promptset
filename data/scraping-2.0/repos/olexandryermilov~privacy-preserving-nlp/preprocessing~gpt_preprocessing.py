import csv
import logging
import openai
import sys
import re
from tqdm import tqdm

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--token", help="Token from openai platform")
argParser.add_argument("-i", "--input", help="Input file")
argParser.add_argument("-o", "--output", help="Where to save your output")

args = argParser.parse_args()
openai_key = args.token
def get_unrepetitive_sentence(sentence):
    """
    Returns the smallest unrepetitive version of the given sentence.
    """
    words = sentence.split()
    num_words = len(words)
    pattern_length = 1
    while pattern_length <= num_words // 2:
        pattern = words[:pattern_length]
        i = pattern_length
        while i + pattern_length <= num_words and words[i:i+pattern_length] == pattern:
            i += pattern_length
        if i == num_words:
            return ' '.join(pattern)
        pattern_length += 1
    return sentence
def get_named_entities(text):
    openai.api_key = openai_key
    prompt = "Find all the locations, names and organizations in the following text. Write them separated by commas: \nText: Daniel worked in Google for five years before moving from America to France. Daniel is now working with Emma in Danone and living in Paris.\nAnswer: Daniel, Google, America, France, Emma, Danone, Paris.\nText:\nAnswer:"+text+"\nAnswer:"
    response = openai.Completion.create(
            model="text-curie-001",
            prompt=prompt,
            temperature=0,
            max_tokens=577,
            top_p=1.0,
            frequency_penalty=1.0,
            presence_penalty=1.0
    )
    #print(response)
    return response['choices'][0]['text'].strip()
def replace_named_entities_chat(text):
    openai.api_key = openai_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Change following named entities using different named entities of same type. "},
            {"role": "user", "content": "Africa, James Potter, Google, Poland, Lily Jameson, Danone"},
            {"role": "assistant", "content": "Asia, John Lennon, Microsoft, Germany, Anna Smith, Starbucks"},
            {"role": "user", "content": text},
        ]
    )
    return response['choices'][0]['message']['content']

replacement_map = {}

def replace_words(text, word_dict):
    # Split text by spaces and punctuation
    words = re.findall(r'\b\w+\b|[^\w\s]', text)

    # Replace words that are keys in the dictionary
    for i, word in enumerate(words):
        if word in word_dict:
            words[i] = get_unrepetitive_sentence(word_dict[word])

    return ' '.join(words)
def preprocess_text(text):
    named_entities = get_named_entities(text).replace('"','').split(",")
    #print("ne = "+','.join(named_entities))
    named_entities = [x for x in named_entities if len(x) >= 2]
    replacement_map = {}
    res = text
    replacements = replace_named_entities_chat(','.join(named_entities)).split(',')
    #print("re = "+ ','.join(replacements))
    for i in range(len(replacements)):
        replacement_map[named_entities[i].strip()] = replacements[i].strip()
        if ' ' in named_entities[i].strip():
           res = res.replace(named_entities[i].strip(), replacements[i].strip())
    res = replace_words(res, replacement_map).replace('\n',' ')
    return res
with open("conll_curie.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["text"])
    with open(args.input, "r") as f:
        lines1 = f.readlines()
        i = 0
        for row in tqdm(lines1):
            try:
                src = row
                writer.writerow((preprocess_text(src).replace('"',"'")))
            except:
                continue
