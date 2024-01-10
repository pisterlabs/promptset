import json
import openai
import Levenshtein
import itertools
from time import sleep
import get_paragraph_labels_16k as pgc
import split_paragraph_16k as sp
# import get_complete_act_labels as gett
from fix_labels import sanitize_labels

# file = open("donnees-test.json", "r")
file = open("test-grande-echelle-raw.json", "r")

data = json.load(file)
file.close()

import os

# first we need to split the text in paragraphs


def split_text(text_to_split):
    splitted = sp.split_text(text_to_split)
    return splitted


def get_labels(text):
    labels = pgc.get_labels(text)
    return labels



one_shot_levenshtein_history = []
few_shot_levenshtein_history = []
one_shot_errors_history = {}
few_shot_errors_history = {}
one_shot_error_count = []
few_shot_error_count = []

iter = 0

already_done = []
labels_history = {}
with open('test-grande-echelle-text_result.json', 'r') as f:
    labels_history = json.load(f)
for i in labels_history.keys():
    already_done.append(i)
print('total already done : ', len(already_done))


for i, name in enumerate(data):
    if name in already_done:
        continue
    iter += 1
    print('==================================')
    print('Now testing : ', name)
    print('==================================')

    text = data[name]['texte']
    if 'divorce' in text:
        continue
    #reference = data[name]['questions']


    splitted = split_text(text)
    # print('splitted : ', splitted)
    labels = get_labels(splitted)
    #print('labels : ', labels)
    # extract labels into a list
    dic = {}
    for key in labels.keys():
        # print('key : ', key)
        if key == 'p4':
            if ('Nom-mari' in dic.keys()) and ('Prenom-mari' in dic.keys()) and ('Nom-mariee' in dic.keys()) and ('Prenom-mariee' in dic.keys()):
                continue
        for bkey in labels[key].keys():
            # print('bkey : ', bkey)
            dic[bkey] = labels[key][bkey]
    labels = dic
    if 'Pays-residence-pere-mari' not in labels.keys():
        labels['Pays-residence-pere-mari'] = ''
    if 'Pays-residence-pere-mariee' not in labels.keys():
        labels['Pays-residence-pere-mariee'] = ''
    # for key in reference.keys():

    #     # Patch for Boolean, temporary
    #     if isinstance(reference[key], bool):
    #         continue
    #     ################################

    #     if key not in labels.keys():
    #         labels[key] = ''

    labels = sanitize_labels(labels)
    #print(labels)
    #print('labels : ', labels['Jour-mariage'])

    labels_history[name] = labels
    # store labels_history in json file
    with open('test-grande-echelle-text_result.json', 'w') as outfile:
        json.dump(labels_history, outfile, indent=4)

