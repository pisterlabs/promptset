#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

site = 'test'

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

def find_matching_categories(categories, created_category_names):
    existing_category_map = {}
    for category in created_category_names:
        if re.sub(r'[^a-zA-Z0-9]', '', category).lower() not in existing_category_map:
            existing_category_map[re.sub(r'[^a-zA-Z0-9]', '', category).lower()] = category
    return_list = []
    for category in categories:
        if re.sub(r'[^a-zA-Z0-9]', '', category).lower() in existing_category_map.keys():
            return_list.append(existing_category_map[re.sub(r'[^a-zA-Z0-9]', '', category).lower()])
    return set(return_list)

with open(os.path.join(cache_directory, 'category_ids.json')) as category_ids_file:
    created_category_ids = json.load(category_ids_file)

response = requests.get(alfresco_url + f'/api/-default-/public/alfresco/versions/1/sites/{site}/containers', auth=alfresco_auth).json()
doc_lib_id = response['list']['entries'][0]['entry']['id']

folder_ids = [doc_lib_id]
folders = {doc_lib_id: site}
files = {}
while len(folder_ids) > 0:
    folder_id = folder_ids.pop()
    response = requests.get(alfresco_url + f'/api/-default-/public/alfresco/versions/1/nodes/{folder_id}/children', auth=alfresco_auth).json()
    for entry in response['list']['entries']:
        node = entry['entry']
        path = folders[folder_id] + '/' + node['name']
        if node['isFolder']:
            folder_ids.append(node['id'])
            folders[node['id']] = path
        else:
            files[path] = node['id']

document_paths = []
for path, _, documents in os.walk(site):
    for document in documents:
        document_path = os.path.join(path, document)
        document_paths.append(document_path)

# Just generate the category hierarchy from a sample of the documents.
for document_path in document_paths:
    if document_path not in files.keys():
        continue
    document = read_document(document_path)
    question = 'Please give a list of thirty hashtags for the following document:\n\n' + document[:1000]
    try:
        response = ask_question(question, llm)
    except:
        continue
    if len(response.split('#')[1:-1]) == 0:
        original_categories = [re.sub('^[^a-zA-Z]*', '', line) for line in response.split('\n') if re.sub('^[^a-zA-Z]*', '', line) != ''][1:-1]
    else:
        original_categories = [re.split(r'[^a-zA-Z0-9]', line.strip())[0] for line in response.split('#')[1:-1]]
    categories = find_matching_categories(original_categories, created_category_ids.keys())
    if len(categories) == 0:
        print(document_path, ' has no categories')
        hashtag_str = ', '.join(map(lambda category: '#' + re.sub(r'[^0-9a-zA-Z]', '', category), random.sample(created_category_ids.keys(), 20)))
        question = f'Which hashtags from [{hashtag_str}] would be suitable for the following document:\n\n' + document[:1000]
        try:
            response = ask_question(question, llm)
        except:
            continue
        categories = [re.split(r'[^a-zA-Z0-9]', line.strip())[0] for line in response.split('#')[1:-1]]
        categories = find_matching_categories(categories, created_category_ids.keys())
        print(f'Suggested categories: {categories}')
    category_links = [{'categoryId': created_category_ids[category]} for category in categories]
    requests.post(alfresco_url + '/api/-default-/public/alfresco/versions/1/nodes/' + files[document_path] + '/category-links', json=category_links, auth=alfresco_auth).json()
