#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
from PyPDF2 import PdfReader

import configparser
import hashlib
import json
import openai
import os
import random
import re
import requests
import time

config_parser = configparser.ConfigParser()
config_parser.read('config.ini')

llm = config_parser.get('General', 'LLM')
genai_url = config_parser.get('GenAI', 'URL')
openai.api_key = config_parser.get('OpenAI', 'APIKey')

alfresco_url = config_parser.get('Alfresco', 'URL')
alfresco_username = config_parser.get('Alfresco', 'Username')
alfresco_password = config_parser.get('Alfresco', 'Password')
alfresco_auth = requests.auth.HTTPBasicAuth(alfresco_username, alfresco_password)

# Ensure the script resumes from where it left off.
random.seed(1)

chatgpt_messages = [ {'role': 'system', 'content': 'You are a intelligent assistant.'} ]
def chatgpt_send_message(message, append_to_messages=True, remove_blank_lines=False):
    chatgpt_messages.append({'role': 'user', 'content': message},)
    chat = openai.ChatCompletion.create(
        model=config_parser.get('OpenAI', 'Model'),
        messages=chatgpt_messages
    )
    reply = chat.choices[0].message.content
    if append_to_messages:
        chatgpt_messages.append({"role": "assistant", "content": reply})
    else:
        chatgpt_messages.pop()
    if remove_blank_lines:
        reply = '\n'.join(line for line in reply.split('\n') if line.strip() != '')
    return reply

cache_directory = 'cache'

def ask_question(question, llm='genai', only_use_cache=False):
    start = time.time()
    digest = hashlib.md5(question.encode('utf-8')).hexdigest()
    cache_filename = ''.join(character for character in question if character.isalnum())[:32] + '-' + digest + '.json'
    cache_path = os.path.join(cache_directory, llm, cache_filename)
    if os.path.isfile(cache_path):
        with open(cache_path) as cache_file:
            response = json.load(cache_file)
    elif only_use_cache:
        raise Exception('Not cached and not sending to LLM')
    else:
        if llm == 'genai':
            params = {'text': question, 'rag': False}
            response = requests.get(genai_url, params).json()['result'].strip()
        elif llm == 'chatgpt':
            response = chatgpt_send_message(question)
        else:
            raise('Unsupported LLM: ' + llm)
        with open(cache_path, 'w') as cache_file:
            json.dump(response, cache_file)
    end = time.time()
    if not only_use_cache:
        print('Question answered in {:.0f} seconds'.format(end - start))
    return response

def update_canonical_category(new_category, old_categories):
    """Increment the number of times new_category has been suggested.
    
    If a similar category is already in use (e.g. different case or plural) then merge them together."""
    if new_category.strip() == '':
        return
    for old_category in old_categories:
        new_lower = new_category.lower()
        old_lower = old_category.lower()
        only_new_has_s = old_lower + 's' == new_lower
        only_old_has_s = old_lower == new_lower + 's'
        if new_lower == old_lower or only_new_has_s or only_old_has_s:
            new_capitals = sum(1 for character in new_category if character.lower() != character)
            old_capitals = sum(1 for character in old_category if character.lower() != character)
            if new_capitals > old_capitals:
                if only_old_has_s:
                    old_categories[new_category + 's'] = old_categories[old_category] + 1
                    del old_categories[old_category]
                else:
                    old_categories[new_category] = old_categories[old_category] + 1
                    del old_categories[old_category]
            else:
                if only_new_has_s:
                    old_categories[old_category + 's'] = old_categories[old_category] + 1
                    del old_categories[old_category]
                else:
                    old_categories[old_category] = old_categories[old_category] + 1
            return
    old_categories[new_category] = 1

def read_document(document_path):
    if document_path.endswith('.pdf'):
        pdf_reader = PdfReader(document_path)
        document = ''
        for page in pdf_reader.pages:
            document += page.extract_text()
            if len(document) > 1000:
                break
    else:
        try:
            with open(document_path) as document_file:
                document = document_file.read()
        except:
            with open(document_path, encoding = 'ISO-8859-1') as document_file:
                document = document_file.read()
    return document

document_paths = []
for path, _, documents in os.walk('corpus'):
    for document in documents:
        document_path = os.path.join(path, document)
        document_paths.append(document_path)

all_categories = Counter()
random.shuffle(document_paths)
# Just generate the category hierarchy from a sample of the documents.
for document_path in document_paths[:100]:
    document = read_document(document_path)
    question = 'Please give a list of thirty hashtags for the following document:\n\n' + document[:1000]
    try:
        response = ask_question(question, llm)
    except:
        continue
    categories = [re.split(r'[^a-zA-Z0-9]', line.strip())[0] for line in response.split('#')[1:-1]]
    for category in categories:
        update_canonical_category(category, all_categories)

print('Pre-processed all existing documents')

categories_list = []
for category in [cat_count[0] for cat_count in all_categories.most_common() if cat_count[1] > 2]:
    category_text = ' '.join(re.findall(r'([A-Z]*[a-z]+|[A-Z]{0,2}[0-9]+|[A-Z]{3,})', category))
    if len(category_text) < len(category):
        category_text = category
    categories_list.append(category_text)
mappings = {}
topic_count = Counter()
additional_text = ''
for i, category in enumerate(categories_list):
    question = f'Only answer using a single word. Give a one word answer to the following question - what type of thing is {category}?' + additional_text
    response = ask_question(question, llm)
    if ':' in response:
        response = response.split(':')[-1]
    if ' - ' in response:
        response = response.split(' - ')[-1]
    if '=' in response:
        response = response.split('=')[-1]
    response = response.strip().strip('.')
    mappings[category] = response
    topic_count[response] += 1

root_categories = []
hierarchy = {}
to_process = list(topic[0] for topic in topic_count.most_common() if topic[1] > 1)
processed = []
while len(to_process) > 0:
    topic = to_process.pop(0)
    if topic not in mappings.keys() or mappings[topic] == topic:
        hierarchy[topic] = [item[0] for item in mappings.items() if item[1] == topic and item[0] != topic]
        root_categories.append(topic)
        processed.append(topic)
    elif mappings[topic] in processed:
        hierarchy[topic] = [item[0] for item in mappings.items() if item[1] == topic]
        processed.append(topic)
    elif mappings[topic] not in mappings.keys() and mappings[topic] not in root_categories and mappings[topic] not in to_process:
        # Ignore the suggested mapping and just treat as another root category.
        hierarchy[topic] = [item[0] for item in mappings.items() if item[1] == topic]
        root_categories.append(topic)
        processed.append(topic)
    else:
        # Put back in queue
        print(f'Putting {topic} back in queue')
        to_process.append(topic)
        continue

all_categories = set()
for parent, children in hierarchy.items():
    all_categories.add(parent)
    for child in children:
        all_categories.add(child)

def create_categories_within(parent, created_category_ids, hierarchy):
    if parent not in hierarchy.keys():
        return
    for category in hierarchy[parent]:
        response = requests.post(alfresco_url + '/api/-default-/public/alfresco/versions/1/categories/' + created_category_ids[parent] + '/subcategories', json={'name': category}, auth=alfresco_auth).json()
        if 'error' in response and response['error']['statusCode'] == 409:
            response = requests.get(alfresco_url + '/api/-default-/public/alfresco/versions/1/categories/' + created_category_ids[parent] + '/subcategories', auth=alfresco_auth).json()
            created_category_ids[category] = [entry['entry']['id'] for entry in response['list']['entries'] if entry['entry']['name'] == category][0]
        else:
            category_id = response['entry']['id']
            print(f'Created category: {category} with id {category_id}')
            created_category_ids[category] = category_id
        create_categories_within(category, created_category_ids, hierarchy)

created_category_ids = {'/': '-root-'}
for category in root_categories:
    response = requests.post(alfresco_url + '/api/-default-/public/alfresco/versions/1/categories/-root-/subcategories', json={"name": category}, auth=alfresco_auth).json()
    if 'error' in response and response['error']['statusCode'] == 409:
        response = requests.get(alfresco_url + '/api/-default-/public/alfresco/versions/1/categories/-root-/subcategories', auth=alfresco_auth).json()
        created_category_ids[category] = [entry['entry']['id'] for entry in response['list']['entries'] if entry['entry']['name'] == category][0]
    else:
        category_id = response['entry']['id']
        print(f'Created root category: {category} with id {category_id}')
        created_category_ids[category] = category_id
    create_categories_within(category, created_category_ids, hierarchy)

    # Periodically cache the category id mappings.
    with open(os.path.join(cache_directory, 'category_ids.json'), 'w') as category_ids_file:
        json.dump(created_category_ids, category_ids_file)
