# Script for using chatgpt for summary and content analysis
import os
import pandas as pd
from datetime import datetime
import openai
import config
import requests
import json
from dateutil.parser import parse

openai.api_key = config.openai_key
""" Read in data
"""
# open and read '#'-separated ocred letters in pupikofer.txt
with open(os.path.join(os.getcwd(),'..','data','literature','pupikofer.txt'), 'r', encoding='utf-8') as file:
    text = file.read()
letters = text.split('#')
# drop empty first item in letters list
letters = letters[1:]
"""
# open and read 'final_register.csv' to get date and letter number
df = pd.read_csv(os.path.join(os.getcwd(),'..','data','register','final_register.csv'), sep=';')
# join 'date' and 'year' to get a single string
df['date'] = df['Datum'] + df['Jahr']
df = df[['Nummer','date','Name_voll']]
# covert date to iso format yyyy-mm-dd to match return of chatGPT
df['date_iso'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
df = df.drop(columns=['date'])
print(df)
"""
# create dictionary for logging
log = {}

""" Get summary and content analysis from chatGPT
"""

# Queries to chatGPT
summary_en = "The following text is a letter in German from Johan Pupikofer to Joseph von Laßberg. Summarize in English in 100 words: "
summary_de = "The following text is a letter in German from Johan Pupikofer to Joseph von Laßberg. Summarize in German in 100 words: "
date = "date the letter in a single string using the format yyyy-mm-dd: "
key_topics = "Extract a list of the letter's key topics in German: "
named_entities = "Extract a list of named entities mentioned in the letter: " 
objects = "Extract a list of objects mentioned in the letter: "
tei = "Encode the letter in TEI using the elements <opener>, <salute>, <p> and <closer>, where apropriate: "

# Loop through letters and get chatGPT responses
for letter in letters[1]:
    print(letter)
    try:
        content = summary_en + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_summary_en = completion.choices[0].message.content

        content = summary_de + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_summary_de = completion.choices[0].message.content

        content = date + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_date = completion.choices[0].message.content

        content = key_topics + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_key_topics = completion.choices[0].message.content

        content = named_entities + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_named_entities = completion.choices[0].message.content

        content = objects + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_objects = completion.choices[0].message.content

        content = tei + letter
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
        chat_response_tei = completion.choices[0].message.content

        print(chat_response_summary_en)
        print(chat_response_summary_de)
        print(chat_response_date)
        print(chat_response_key_topics)
        print(chat_response_named_entities)
        print(chat_response_objects)
        print(chat_response_tei)

        """ Get letter number from date and save return to corresponding xml file
        """
        correspondent = "Pupikofer"
        # filter df for letters send to particular correspondent
        df = df[df.Name_voll.str.contains(correspondent)]
        print(df)
        letter_number = df[df.date_iso == pd.Timestamp(chat_response_date)]
    
        # check if there are more than two letters in letter_number
        if len(letter_number) > 1:
            # add chat_response_summary, chat_response_date, chat_response_key_topics, chat_response_named_entities, chat_response_objects, chat_response_tei and letter to log
            log[correspondent + '-' + str(chat_response_date)] = {'chat_response_summary_en': chat_response_summary_en, 'chat_response_summary_de': chat_response_summary_de, 'chat_response_date': chat_response_date, 'chat_response_key_topics': chat_response_key_topics, 'chat_response_named_entities': chat_response_named_entities, 'chat_response_objects': chat_response_objects, 'chat_response_tei': chat_response_tei}
        else:
            letter_number = letter_number['Nummer'].values[0]
            print(letter_number)
            # construct filename
            filename = f'lassberg-letter-{letter_number}.xml'

            # open xml file with filename
            with open(os.path.join(os.getcwd(),'..','data','letters',filename), 'r', encoding='utf-8') as file:
                xml = file.read()
                xml = xml.replace('<p/>', chat_response_tei)
                xml = xml.replace('<note type="chatgpt-summary"/>', f'<note type="chatgpt-summary"/>\n<note type="chatgpt-summary" xml:lang="en">{chat_response_summary_en}</note>')
                xml = xml.replace('<note type="chatgpt-summary"/>', f'<note type="chatgpt-summary" xml:lang="de">{chat_response_summary_de}</note>')
                xml = xml.replace('<note type="chatgpt-keytopics"/>', f'<note type="chatgpt-keytopics">{chat_response_key_topics}</note>')
                xml = xml.replace('<note type="chatgpt-persons"/>', f'<note type="chatgpt-persons">{chat_response_named_entities}</note>')
                xml = xml.replace('<note type="chatgpt-objects"/>', f'<note type="chatgpt-objects">{chat_response_objects}</note>')
            with open(os.path.join(os.getcwd(),'..','data','letters',filename), 'w', encoding='utf-8') as new_file:    
                new_file.write(xml)
                # add chat_response_summary, chat_response_date, chat_response_key_topics, chat_response_named_entities, chat_response_objects, chat_response_tei and letter to log
                log[correspondent + '-' + str(chat_response_date)] = {'letter-number': letter_number, 'chat_response_summary_en': chat_response_summary_en, 'chat_response_summary_de': chat_response_summary_de,'chat_response_date': chat_response_date, 'chat_response_key_topics': chat_response_key_topics, 'chat_response_named_entities': chat_response_named_entities, 'chat_response_objects': chat_response_objects, 'chat_response_tei': chat_response_tei}
    except:
        # add chat_response_summary, chat_response_date, chat_response_key_topics, chat_response_named_entities, chat_response_objects, chat_response_tei and letter to log
        log[correspondent + '-' + str(chat_response_date)] = {'status': 'error', 'chat_response_summary_en': chat_response_summary_en, 'chat_response_summary_de': chat_response_summary_de, 'chat_response_date': chat_response_date, 'chat_response_key_topics': chat_response_key_topics, 'chat_response_named_entities': chat_response_named_entities, 'chat_response_objects': chat_response_objects, 'chat_response_tei': chat_response_tei}
    


# dump log to json file
with open(os.path.join(os.getcwd(),'..','data','logs','log.json'), 'w') as log_file:
    json.dump(log, log_file)



