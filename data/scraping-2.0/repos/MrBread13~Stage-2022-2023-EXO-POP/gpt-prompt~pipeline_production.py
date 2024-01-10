import json
import openai
import Levenshtein
import itertools
from time import sleep
import get_paragraph_labels_16k as pgc
import split_paragraph_16k as sp
# import get_complete_act_labels as gett
from fix_labels import sanitize_labels

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



# one_shot_levenshtein_history = []
# few_shot_levenshtein_history = []
# one_shot_errors_history = {}
# few_shot_errors_history = {}
# one_shot_error_count = []
# few_shot_error_count = []

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

    text = data[name]#['texte']
    if 'divorce' in text:
        continue
    #reference = data[name]['questions']


    splitted = split_text(text)
    # print('splitted : ', splitted)
    labels = get_labels(splitted)
    #print('labels : ', labels)


    # if 'Pays-residence-pere-mari' not in labels['p2'].keys():
    #     labels['p2']['Pays-residence-pere-mari'] = ''
    # if 'Pays-residence-pere-mariee' not in labels['p3'].keys():
    #     labels['p3']['Pays-residence-pere-mariee'] = ''

    reference = {}
    with open('labels-reference.json', 'r') as f:
        reference = json.load(f)


    for paragraph in reference.keys():
        if paragraph not in labels.keys():
                labels[paragraph] = {}
        for label in reference[paragraph].keys():
            if label not in labels[paragraph].keys():
                labels[paragraph][label] = ""



    labels_keys_old = labels

    #extract labels into a list
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


    # for key in reference.keys():

    #     # Patch for Boolean, temporary
    #     if isinstance(reference[key], bool):
    #         continue
    #     ################################

    #     if key not in labels.keys():
    #         labels[key] = ''

    labels = sanitize_labels(labels)

    for paragraph in labels_keys_old.keys():
        for key in labels_keys_old[paragraph].keys():
            if key in labels.keys():
                labels_keys_old[paragraph][key] =  labels[key]

    labels = labels_keys_old
    #print(labels)
    #print('labels : ', labels['Jour-mariage'])

    labels_history[name] = {'labels': labels, 'text': splitted}
    # store labels_history in json file
    with open('test-grande-echelle-text_result.json', 'w') as outfile:
        json.dump(labels_history, outfile, indent=4)
