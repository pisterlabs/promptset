import requests
import os
from os import listdir
from os.path import isfile, join
import json
import argparse
from timeit import default_timer as timer
import time
from datetime import timedelta
from pprint import pprint
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager, Process
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import openai
import string
import re
import os.path
from os import path

# Read the list of phrasal verbs
with open("complete-pv/Complete-PV-list.txt") as f:
    lines = f.readlines()
phrasal_verbs = {}
verbs = set()
for line in lines:
    if re.search('.[A-Z].', line.strip()):
        if not re.search('.[A-Z][A-Z].', line.strip()):
            end = re.search('.[A-Z].', line.strip()).start()
            tmp_line = line[0:end]
            words = tmp_line.strip().split(" ")
    else:
        words = line.strip().split(" ")
    if len(words) > 1 and len(words) < 4:
        if words[0][0].isupper() and words[-1][-1] not in string.punctuation and words[-1][0] not in string.punctuation:
            lower_words = []
            for word in words:
                lower_words.append(word.lower())
            if lower_words[0] not in phrasal_verbs.keys():
                phrasal_verbs[lower_words[0]] = {" ".join(lower_words)}
            else:
                phrasal_verbs[lower_words[0]].add(" ".join(lower_words))
                
# This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.                
model = SentenceTransformer('all-MiniLM-L6-v2') 

#manager = Manager()
#IE_output = manager.list()

# if not specified, start and end denote the word id at the doc level
# "start_sent_level" denotes the start word id at the sentence level
def view_map_update(output):
    count = 0
    view_map = {}
    for view in output['views']:
        view_map[view['viewName']] = count
        count += 1
    return view_map

def sent_id_getter(token_id, SRL_output):
    i = -1
    for sEP in SRL_output['sentences']['sentenceEndPositions']:
        i += 1
        if token_id < sEP:
            return i
    #raise ValueError("Cannot find sent_id.")
    return i + 1    # NER tokenizer may differ from SRL tokenizer

def read_doc(fname):
    tag_list = []
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            tag_list.append(line.split(' - ')[0])
    return tag_list

def CP_getter(sentence):
    # Constituency Parsing
    headers = {'Content-type':'application/json'}
    CP_response = requests.post('http://127.0.0.1:6003/annotate', json={"text": sentence}, headers=headers)
    if CP_response.status_code != 200:
        print("CP_response:", CP_response.status_code)
    result = json.loads(CP_response.text)
    return result

def relation_preparer(SRL_output):
    new_output = {'corpusId': SRL_output['corpusId'], 
                  'id': SRL_output['id'], 
                  'sentences': SRL_output['sentences'],
                  'text': SRL_output['text'],
                  'tokens': SRL_output['tokens'],
                  'views': []
                 }
    for view in SRL_output['views']:
        my_view = {}
        if view['viewName'] == 'Event_extraction':
            my_view['viewName'] = view['viewName']
            my_view['viewData'] = [{'viewType': 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView',
                                    'viewName': 'event_extraction',
                                    'generator': 'cogcomp_kairos_event_ie_v1.0',
                                    'score': 1.0,
                                    'constituents': view['viewData'][0]['constituents'],
                                    'relations': view['viewData'][0]['relations'],
                                   }]
            
            new_output['views'].append(my_view)
    return new_output

def temporal_getter(SRL_output, onepass = 1):
    headers = {'Content-type':'application/json'}
    #if onepass:
    if True:
        temporal_service = 'http://localhost:6009/annotate'
    #else:
    #    temporal_service = 'http://dickens.seas.upenn.edu:4024/annotate'
    print("Calling service from " + temporal_service)
    temporal_response = requests.post(temporal_service, json=SRL_output, headers=headers)
    
    if temporal_response.status_code != 200:
        print("temporal_response:", temporal_response.status_code)
    try: 
        result = json.loads(temporal_response.text)
        return result
    except:
        return None
    
def subevent_getter(SRL_output):
    headers = {'Content-type':'application/json'}
    subevent_response = requests.post('http://localhost:6004/annotate', json=SRL_output, headers=headers)
    if subevent_response.status_code != 200:
        print("subevent_response:", subevent_response.status_code)
    try:
        result = json.loads(subevent_response.text)
        return result
    except:
        return None
    
def coref_getter(SRL_output):
    # Note: coref service is not provided in this repo
    headers = {'Content-type':'application/json'}
    coref_response = requests.post('http://localhost:8888/annotate', json=SRL_output, headers=headers)
    if coref_response.status_code != 200:
        print("coref_response:", coref_response.status_code)
    try:
        result = json.loads(coref_response.text)
        return result
    except:
        return None

def extract_head_noun(children):
    Clause_Level = read_doc('CP_Clause_Level.txt')
    Phrase_Level = read_doc('CP_Phrase_Level.txt')
    Word_Level = read_doc('CP_Word_Level.txt')
    num_c = len(children)
    child_index = -1
    for child in children:
        child_index += 1
        if child['nodeType'] in Word_Level:
            if child['nodeType'] in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
                next_index = child_index+1
                if next_index < num_c:
                    if children[next_index]['nodeType'] not in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
                        return child['word']
                    else:
                        while children[next_index]['nodeType'] in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']:
                            next_index += 1
                            if next_index >= num_c:
                                break
                        return children[next_index-1]['word']
                else:
                    return child['word']
        elif child['nodeType'] in Phrase_Level:
            if 'NP' in child['attributes']: 
                # we are not interested in the extraction of any nouns in the query, 
                # but only those that appear within the NP component,
                # e.g., NP -> NP + VP (VP -> POS + NP), you cannot let the function search within VP
                return extract_head_noun(child['children'])
        elif child['nodeType'] in Clause_Level:
            return extract_head_noun(child['children'])
        else:
            #print("extract_head_noun:", child['nodeType'], "is not in any list")
            #print("child:", child)
            pass

def similar(string1, string2):
    if string2 in string1 and len(string1) - len(string2) <= 2:
        #print("similar:", string1, string2)
        return True
    else:
        return False
    
def find(children, query):
    # return value is a dict or None
    for child in children:
        if child['word'] == query or similar(child['word'], query):
            return child
        else:
            if 'children' in child.keys():
                result = find(child['children'], query)
                if type(result) == dict:
                    return result
    return None
    
def head_word_extractor(CP_result, query):
    children = CP_result['hierplane_tree']['root']['children']
    target_child = find(children, query)
    try:
        if 'children' in target_child.keys(): # target_child can be None, so it might have no keys
            return extract_head_noun(target_child['children'])
        else:
            return target_child['word']
    except:
        #print("Did not find '", query, "' in Constituency Parsing result")
        return None

def entity_info_getter(query, sent_id, entities):
    if sent_id in entities:
        for entity in entities[sent_id]:
            if query in entity['mention']:
                return entity['label'], ' '.join(entity['mention']), entity['start'], entity['end']
    else:
        #print("NER module detected no entity in the {i}-th sentence".format(i=sent_id))
        return None

def event_extractor(text, text_id, NOM=True):
    if text == '':
        return {}
    headers = {'Content-type':'application/json'}
    SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate', json={"sentence": text}, headers=headers)
    if SRL_response.status_code != 200:
        print("SRL_response:", SRL_response.status_code)
    try:
        SRL_output = json.loads(SRL_response.text)
    except:
        return {}
    
    token_num = len(SRL_output['tokens'])
    if token_num not in SRL_output['sentences']['sentenceEndPositions']:
        SRL_output['sentences']['sentenceEndPositions'].append(token_num)
    print("SRL done")
    
    headers = {'Content-type':'application/json'}
    NER_response = requests.post('http://dickens.seas.upenn.edu:4022/ner/', json={"task": "kairos_ner","text" : text}, headers=headers)
    if NER_response.status_code != 200:
        print("NER_response:", NER_response.status_code)
    try:
        NER_output = json.loads(NER_response.text)
        NER_view_map = view_map_update(NER_output)
        print("NER done")
    except:
        print("NER result empty")
        assert 0 == 1
    entities = {}
    for mention in NER_output['views'][NER_view_map['NER_CONLL']]['viewData'][0]['constituents']:
        sent_id = sent_id_getter(mention['start'], SRL_output)
        # TODO: Check whether SRL tokenizer is the same as NER's
        entity = {'mention': NER_output['tokens'][mention['start']:mention['end']], \
                  'label': mention['label'], \
                  'start': mention['start'], \
                  'end': mention['end'], \
                  'sentence_id': sent_id, \
                 }
        if sent_id in entities.keys():
            entities[sent_id].append(entity)
        else:
            entities[sent_id] = [entity]
            
    '''Append NER results to SRL'''
    SRL_output['views'].append(NER_output['views'][NER_view_map['NER_CONLL']])
    SRL_view_map = view_map_update(SRL_output)
    #print(SRL_view_map)

    CP_output = []
    pEP = 0
    for sEP in SRL_output['sentences']['sentenceEndPositions']:
        this_sentence = " ".join(SRL_output['tokens'][pEP:sEP])
        pEP = sEP
        CP_output.append(CP_getter(this_sentence))
    if SRL_output['sentences']['sentenceEndPositions'][-1] < len(SRL_output['tokens']):
        this_sentence = " ".join(SRL_output['tokens'][SRL_output['sentences']['sentenceEndPositions'][-1]:])
        CP_output.append(CP_getter(this_sentence))
    print("CP done")
        
    Events = []
    argument_ids = []
    
    if NOM: 
        source = ['SRL_ONTONOTES', 'SRL_NOM']
    else:
        source = ['SRL_ONTONOTES']
    for viewName in source:
        for mention in SRL_output['views'][SRL_view_map[viewName]]['viewData'][0]['constituents']:
            sent_id = sent_id_getter(mention['start'], SRL_output)
            mention_id_docLevel = str(text_id) + '_' + str(sent_id) + '_' + str(mention['start'])
            if mention['label'] == 'Predicate':
                if sent_id == 0:
                    start = mention['start']
                    end = mention['end']
                else:
                    start = mention['start'] - SRL_output['sentences']['sentenceEndPositions'][sent_id-1] # event start position in the sentence = event start position in the document - offset
                    end = mention['end'] - SRL_output['sentences']['sentenceEndPositions'][sent_id-1]
                    
                event_id = str(text_id) + '_' + str(sent_id) + '_' + str(start)
                predicate = ''
                if mention['properties']['predicate'] in phrasal_verbs.keys() and mention['start'] < len(SRL_output['tokens']) - 2:
                    next_token = SRL_output['tokens'][mention['start'] + 1]
                    token_after_next = SRL_output['tokens'][mention['start'] + 2]
                    potential_pv_1 = " ".join([mention['properties']['predicate'], next_token, token_after_next])
                    #print(potential_pv_1)
                    potential_pv_2 = " ".join([mention['properties']['predicate'], next_token])
                    #print(potential_pv_2)
                    if potential_pv_2 in phrasal_verbs[mention['properties']['predicate']]:
                        predicate = potential_pv_2
                        print(predicate)
                    if potential_pv_1 in phrasal_verbs[mention['properties']['predicate']]:
                        predicate = potential_pv_1
                        print(predicate)
                    if predicate == '':
                        predicate = mention['properties']['predicate']
                else:
                    predicate = mention['properties']['predicate']
                
                
                try:
                    assert mention['start'] != None
                    assert mention['end'] != None
                    Events.append({'event_id': event_id, \
                                   'event_id_docLevel': mention_id_docLevel, \
                                   'start': mention['start'], \
                                   'end': mention['end'], \
                                   'start_sent_level': start, \
                                   'end_sent_level': end, \
                                   'properties': {'predicate': [mention['properties']['predicate']], \
                                                  'SenseNumber': '01', \
                                                  'sentence_id': sent_id
                                                 }, \
                                   'label': predicate
                                  })
                except:
                    print("mention with None start or end:", mention)
                    pass
                 
            else:
                start = mention['start'] # document level position
                end = mention['end']
                query = ' '.join(SRL_output['tokens'][start:end]).strip()
                ENTITY_INFO = entity_info_getter(query, sent_id, entities)
                if mention['label'] in Events[-1]['properties'].keys():
                    count = 1
                    for label in Events[-1]['properties'].keys():
                        if '_' in label and label.split('_')[0] == mention['label']:
                            count += 1
                    arg_label = mention['label'] + '_' + str(count)
                else:
                    arg_label = mention['label']
                if ENTITY_INFO:
                    # the argument found by SRL is directly an entity detected by NER
                    Events[-1]['properties'][arg_label] = {'entityType': ENTITY_INFO[0], \
                                                           'mention': ENTITY_INFO[1], \
                                                           'start': ENTITY_INFO[2], \
                                                           'end': ENTITY_INFO[3], \
                                                           'argument_id': str(text_id) + '_' + str(sent_id) + '_' + str(ENTITY_INFO[2]), \
                                                          }
                    argument_ids.append(str(text_id) + '_' + str(sent_id) + '_' + str(ENTITY_INFO[2]))
                else:
                    # the argument found by SRL might be a phrase / part of clause, hence head word extraction is needed
                    head_word = head_word_extractor(CP_output[sent_id], query)
                    if head_word:
                        ENTITY_INFO = entity_info_getter(head_word, sent_id, entities)
                        if ENTITY_INFO:
                            # if the head word is a substring in any entity mention detected by NER
                            Events[-1]['properties'][arg_label] = {'entityType': ENTITY_INFO[0], \
                                                                   'mention': ENTITY_INFO[1], \
                                                                   'start': ENTITY_INFO[2], \
                                                                   'end': ENTITY_INFO[3], \
                                                                   'argument_id': str(text_id) + '_' + str(sent_id) + '_' + str(ENTITY_INFO[2]), \
                                                                  }
                            argument_ids.append(str(text_id) + '_' + str(sent_id) + '_' + str(ENTITY_INFO[2]))
                        else:
                            Events[-1]['properties'][arg_label] = {'mention': head_word, 'entityType': 'NA', 'argument_id': mention_id_docLevel} # actually not exactly describing its position
                            argument_ids.append(mention_id_docLevel)
                    else:
                        Events[-1]['properties'][arg_label] = {'mention': query, 'entityType': 'NA', 'argument_id': mention_id_docLevel}
                        argument_ids.append(mention_id_docLevel)
    print("head word extraction done") 
    """
    Can directly go to the Events_final if ignoring event typing (line 441, before '''Append Event Typing Results to SRL''')
    
    
    #Events_with_arg = [event for event in Events if len(event['properties']) > 3]
    #Events_non_nom = [event for event in Events_with_arg if event['event_id_docLevel'] not in argument_ids]
    #print("Removal of nominal events that serve as arguments of other events")
    
    #for event in Events_non_nom:
    for event in Events:
        sent_id = int(event['event_id'].split('_')[1]) # 0-th: text_id    1-st: sent_id    2-nd: event_start_position_in_sentence
        if sent_id < len(SRL_output['sentences']['sentenceEndPositions']):
            sEP = SRL_output['sentences']['sentenceEndPositions'][sent_id] # sEP: sentence End Position
            if sent_id == 0:
                tokens = SRL_output['tokens'][0:sEP]
            else:
                pEP = SRL_output['sentences']['sentenceEndPositions'][sent_id-1] # pEP: previous sentence End Position
                tokens = SRL_output['tokens'][pEP:sEP]
        else:
            pEP = SRL_output['sentences']['sentenceEndPositions'][-1]
            tokens = SRL_output['tokens'][pEP:]
        
        event_sent = " ".join(tokens)
        if event_sent[-1] != '.':
            event_sent = event_sent + '.'
        
        headers = {'Content-type':'application/json'}
        #ET_response = requests.post('http://dickens.seas.upenn.edu:4036/annotate', json={"tokens": tokens, "target_token_position": [event['start_sent_level'], event['end_sent_level']]}, headers=headers)
        ET_response = requests.post('http://leguin.seas.upenn.edu:4023/annotate', json={"text": event_sent}, headers=headers)
        if ET_response.status_code != 200:
            print("ET_response:", ET_response.status_code)
        
        try:
            ET_output = json.loads(ET_response.text)
            for view in ET_output['views']:
                if view['viewName'] == 'Event_extraction':
                    for constituent in view['viewData'][0]['constituents']:
                        if constituent['start'] == event['start_sent_level']:
                            event['label'] = constituent['label']
        #try:
        #   event['label'] = ET_output['predicted_type']
        except:
            event['label'] = "NA"
            print("-------------------------------- Event Typing result: NA! --------------------------------")
            print("the sentence is: " + event_sent)
            print("the event is: " + event['properties']['predicate'][0])
    
    Events_non_reporting = [event for event in Events if event['label'] not in ['NA', 'Reporting', 'Statement'] and event['properties']['predicate'][0] not in ["be", "have", "can", "could", "may", "might", "must", "ought", "shall", "will", "would", "say", "nee", "need", "do", "happen", "occur"]]
    
    print("event typing done, removed 'be', Reporting, Statement, NA events")
    print("event num:", len(Events_non_reporting))
    #print(Events[0])
    
    # remove repeated events
    event_types = []
    Events_final = []
    for event in Events_non_reporting:
        if event['label'] not in event_types:
            Events_final.append(event)
            event_types.append(event['label'])
    print("num of events with different types:", len(Events_final))
    """
    Events_final = [event for event in Events if event['label'] not in ["be", "have", "can", "could", "may", "might", "must", "ought", "shall", "will", "would", "say", "nee", "need", "do", "happen", "occur"]]
    
    '''Append Event Typing Results to SRL'''
    Event_Extraction = {'viewName': 'Event_extraction', \
                        'viewData': [{'viewType': 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView', \
                                      'viewName': 'event_extraction', \
                                      'generator': 'Event_ONTONOTES+NOM_MAVEN_Entity_CONLL02+03', \
                                      'score': 1.0, \
                                      'constituents': Events_final, \
                                      'relations': []
                                     }]
                       }
    #pprint(Events_final)
    SRL_output['views'].append(Event_Extraction)
    print("event extraction done")
    #IE_output.append(SRL_output)
    print("------- The {i}-th piece of generated text processing complete! -------".format(i=text_id))
    return SRL_output

def call_nlpcloud_API(prompt, token):
    # Deprecated function
    headers = {"Authorization": "Token " + token}
    generation_response = requests.post('https://api.nlpcloud.io/v1/gpu/gpt-j/generation', \
                                        json={"text": prompt, \
                                              "min_length": 50, \
                                              "max_length": 256, \
                                              "temperature": 0.9, \
                                              "top_p": 0.8, \
                                             }, \
                                        headers=headers
                                       )
    if generation_response.status_code != 200:
        print("generation_response:", generation_response.status_code)
    return generation_response

def headline_generator(event, news):
    # TODO: test this function
    event = event.lower()
    if news:
        prompt = "The headline of the news about " + event + " was '"
    else:
        #prompt = "The title for 'How to make " + event + " possible' is '"
        return "How to make " + event + " possible"
    response = call_nlpcloud_API(prompt)
    len_hp = len(prompt)
    generated_text = json.loads(response.text)['generated_text'][len_hp:]
    end_of_headline = generated_text.find("'")
    if end_of_headline:
        return generated_text[0:end_of_headline]
    else:
        return event

def print_event(event_extraction_results, f_out, NA_event=True):
    # event_extraction_results: list
    for event in event_extraction_results:
        #To_print = "Event: '{mention}' ({label}, {event_id})\t".format(event_id=event['event_id_docLevel'], mention=event['properties']['predicate'][0], label=event['label'])
        To_print = "Event: '{mention}' ({event_id})\t".format(event_id=event['event_id_docLevel'], mention=event['label'])
        for key in event['properties'].keys():
            if key not in ["predicate", "sentence_id", "SenseNumber"]:
                To_print += "{arg}: '{mention}' ({entityType}, {argument_id})\t".format(arg=key, mention=event['properties'][key]['mention'], entityType=event['properties'][key]['entityType'], argument_id=event['properties'][key]['argument_id'])
                
        if NA_event: # printing info for events with type "NA"
            print(To_print, file = f_out)
        else:
            if event['label'] != 'NA':
                print(To_print, file = f_out)
                
def schema_induction(prompt, call_n, f_out, gt_input = False, gt_output = False, debugging = 1, temporal = True, print_events = True, subevent = True, coref = False):
    IE_output = []
    if gt_input:
        generated_text = gt_input
    else:
        if debugging:
            with open('parrot.pkl', 'rb') as f:
                generated_text = pickle.load(f)
                generated_text = generated_text[0:debugging]
        else:
            generated_text = []
            print("\tGenerating text")
            for i in range(call_n):
                response = call_nlpcloud_API(prompt)
                generated_text.append(json.loads(response.text)['generated_text'])
        
    if gt_output:
        return generated_text
    
    print("Schema Induction module is going to run IE for " + str(len(generated_text)) + " pieces of text.")
    text_ids = [i for i in range(len(generated_text))]
    with Pool(processes=2) as pool:
        IE_output = pool.starmap(event_extractor, zip(generated_text, text_ids))
    if print_events:
        for SRL_output in IE_output:
            if SRL_output == {}:
                continue
            print_event(SRL_output['views'][-1]['viewData'][0]['constituents'], f_out)
    
    if subevent:
        IE_output_subevent = []
        print("start working on subevent...")
        for SRL_output in IE_output:
            if SRL_output == {}:
                continue
            temp = relation_preparer(SRL_output)
            subevent_res = subevent_getter(temp)
            if subevent_res:
                IE_output_subevent.append(subevent_res)
        IE_output = []
        IE_output = IE_output_subevent
        
    if coref:
        IE_output_coref = []
        print("start working on coref...")
        for SRL_output in IE_output:
            if SRL_output == {}:
                continue
            temp = relation_preparer(SRL_output)
            coref_res = coref_getter(temp)
            if coref_res:
                IE_output_coref.append(coref_res)
        IE_output = []
        IE_output = IE_output_coref
    
    if temporal:
        IE_output_temporal = []    
        count = -1
        print("start working on temporal...")
        for SRL_output in IE_output:
            if SRL_output == {}:
                continue
            temp = relation_preparer(SRL_output)
            """
            count += 1
            dump_EE = True
            if dump_EE:
                with open("intermediate/temp" + str(count) + ".json", 'w') as f:
                    json.dump(temp, f)
            """
            print("schema induction -- num of events:", len(temp['views'][-1]['viewData'][0]['constituents']))
            temporal_res = temporal_getter(temp)
            if temporal_res:
                IE_output_temporal.append(temporal_res)
        return IE_output_temporal
    else:
        return IE_output
    
def print_stats(IE_output, topic, f_out):
    event_types_total = {}
    #event_mentions_total = {}
    event_types_detail = {}
    event_args = {}
    for SRL_output in IE_output:
        if SRL_output == {}:
            continue
        event_types = {}
        #event_mentions = {}
        
        for event in SRL_output['views'][-1]['viewData'][0]['constituents']:
            if event['label'] != "NA": # not reporting those events w/o types
                #event_mentions[event['properties']['predicate'][0]] = 1
                event_types[event['label']] = 1
                if event['label'] not in event_types_detail.keys():
                    event_types_detail[event['label']] = set()
                event_types_detail[event['label']].add(event['event_id_docLevel'])
                if event['label'] not in event_args.keys():
                    event_args[event['label']] = {}
                    for arg in event['properties'].keys():
                        arg_no_index = arg.split('_')[0]
                        if "ARG" in arg:
                            if event['properties'][arg]['entityType'] != 'NA':
                                event_args[event['label']][arg_no_index] = {event['properties'][arg]['entityType']: 1}
                else:
                    for arg in event['properties'].keys():
                        arg_no_index = arg.split('_')[0]
                        if "ARG" in arg:
                            if event['properties'][arg]['entityType'] != 'NA':
                                if arg_no_index in event_args[event['label']].keys():
                                    if event['properties'][arg]['entityType'] in event_args[event['label']][arg_no_index].keys():
                                        event_args[event['label']][arg_no_index][event['properties'][arg]['entityType']] += 1
                                    else:
                                        event_args[event['label']][arg_no_index][event['properties'][arg]['entityType']] = 1
                                else:
                                    event_args[event['label']][arg_no_index] = {event['properties'][arg]['entityType']: 1}
                            
        for event_type in event_types.keys():
            if event_type in event_types_total.keys():
                event_types_total[event_type] += 1
            else:
                event_types_total[event_type] = 1
        #for mention in event_mentions.keys():
        #    if mention in event_mentions_total.keys():
        #        event_mentions_total[mention] += 1
        #    else:
        #        event_mentions_total[mention] = 1

    #print('top 20 event mentions:')            
    #pprint(sorted(event_mentions_total.items(), key=lambda x: x[1], reverse=True)[:20])
    #print('\ntop 30 events:\n', file = f_out)   
    print('\ntop events:\n', file = f_out)   
    #pprint(sorted(event_types_total.items(), key=lambda x: x[1], reverse=True)[:20])
    #for et, count in sorted(event_types_total.items(), key=lambda x: x[1], reverse=True)[:30]:
    for et, count in sorted(event_types_total.items(), key=lambda x: x[1], reverse=True): # Oct 17 2022
        print("'" + et + "'", "appears in", str(count), "docs, mentions:", event_types_detail[et], end = '', file = f_out)
        print(", arguments:", event_args[et], file = f_out)
        #print("\n'" + et + "'", "appears in", str(count), "docs, mentions:", end=' ')
        #for mention in event_types_detail[et]:
        #    print("'" + mention + "':" + str(event_mentions_total[mention]), end=', ')
    
    temporal_relation = {}
    subevent_relation = {}
    coref_relation = {}
    text_id = -1
    for SRL_output in IE_output:
        if SRL_output == {}:
            continue
        text_id += 1
        for relation in SRL_output['views'][-1]['viewData'][0]['relations']:
            rel = relation['relationName']
            src = int(relation['srcConstituent']) # coref result: '1'; temporal / subevent result: 1
            tgt = int(relation['targetConstituent'])
            source = SRL_output['views'][-1]['viewData'][0]['constituents'][src]['label']
            target = SRL_output['views'][-1]['viewData'][0]['constituents'][tgt]['label']
            #logits = relation['logits']
            #print(rel, source, target, logits)
            if source == target:
                continue
            if rel in ['before', 'after']:
                if rel == 'before':
                    pair = (source, target)
                else:
                    pair = (target, source)
                if pair in temporal_relation.keys():
                    temporal_relation[pair].add(text_id)
                else:
                    temporal_relation[pair] = {text_id}
            if rel in ['SuperSub', 'SubSuper']:
                if rel == 'SuperSub':
                    pair = (source, target)
                else:
                    pair = (target, source)
                if pair in subevent_relation.keys():
                    subevent_relation[pair].add(text_id)
                else:
                    subevent_relation[pair] = {text_id}
            if rel == "coref":
                pair = (source, target)
                if pair in coref_relation.keys():
                    coref_relation[pair].add(text_id)
                else:
                    coref_relation[pair] = {text_id}
                        
    #print("\ntop 30 temporal relations:\n", file = f_out)
    #for et, count in sorted(temporal_relation.items(), key=lambda x: len(x[1]), reverse=True)[:30]:
    print("\ntop temporal relations:\n", file = f_out)
    for et, count in sorted(temporal_relation.items(), key=lambda x: len(x[1]), reverse=True): # Oct 17 2022
        print("'" + str(et) + "'", "appears in", str(len(count)), "docs:", count, file = f_out)
        
    #print("\ntop 30 subevent relations:\n", file = f_out)
    #for et, count in sorted(subevent_relation.items(), key=lambda x: len(x[1]), reverse=True)[:30]:
    print("\ntop subevent relations:\n", file = f_out)
    for et, count in sorted(subevent_relation.items(), key=lambda x: len(x[1]), reverse=True): # Oct 17 2022
        print("'" + str(et) + "'", "appears in", str(len(count)), "docs:", count, file = f_out)
        
    #print("\ntop 30 coref relations:\n", file = f_out)
    #for et, count in sorted(coref_relation.items(), key=lambda x: len(x[1]), reverse=True)[:30]:
    #print("\ntop coref relations:\n", file = f_out)
    #for et, count in sorted(coref_relation.items(), key=lambda x: len(x[1]), reverse=True): # Oct 17 2022
    #    print("'" + str(et) + "'", "appears in", str(len(count)), "docs:", count, file = f_out)
    """ 
    G=nx.Graph()
    for pair in temporal_relation_total.keys():
        count = temporal_relation_total[pair]
        if count >= 3:
            G.add_edge(pair[0], pair[1])
            nx.set_edge_attributes(G, {pair: {"weight": count}})
    pos = nx.spring_layout(G)
    plt.figure(3,figsize=(12,12)) 
    nx.draw(G, pos, with_labels = True)
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig('png/' + topic + '.png') 
    """
    
def search_for_events(IE_output, event_type = "", event_mention = ""):
    for SRL_output in IE_output:
        if SRL_output == {}:
            continue
        for event in SRL_output['views'][-1]['viewData'][0]['constituents']:
            if event['label'] == event_type or event['properties']['predicate'][0] == event_mention:
                To_print = "Event: '{mention}' ({label}, {event_id})\t".format(event_id=event['event_id_docLevel'], mention=event['properties']['predicate'][0], label=event['label'])
                for key in event['properties'].keys():
                    if key not in ["predicate", "sentence_id"]:
                        To_print += "{arg}: '{mention}' ({entityType}, {argument_id})\t".format(arg=key, mention=event['properties'][key]['mention'], entityType=event['properties'][key]['entityType'], argument_id=event['properties'][key]['argument_id'])
                print(To_print)
                
def save_generated_text(generated_text, topic):
    time_str = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    with open('generated_text/' + topic + '_' + time_str + '.pkl', 'wb') as f:
        pickle.dump(generated_text, f)
        
def save_IE_output(IE_output, topic):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    with open('IE_output/' + topic + '_' + time_str + '.pkl', 'wb') as f:
        pickle.dump(IE_output, f)

'''        
def similarity(topic, text):
    encoded_input = tokenizer(text, return_tensors="pt", max_length=256)
    output = model(**encoded_input)
    if topic == text:
        return 1
    else:
        return 0
    
def filter_gt(generated_text, topic):
    ranking = {}
    text_id = 0
    for text in generated_text:
        ranking[text_id] = similarity(topic, text)
        text_id += 1
    ranked_list = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    new_gt = []
    count = -1
    for rank in ranked_list:
        count += 1
        if count < len(ranked_list) / 2:
            new_gt.append(generated_text[rank[0]])
    return new_gt
'''

def filter_gt_sbert(generated_text, topic):
    # https://www.sbert.net/docs/usage/semantic_textual_similarity.html
    num = len(generated_text)
    topic_ = [topic] * num
    embeddings1 = model.encode(generated_text, convert_to_tensor=True)
    embeddings2 = model.encode(topic_, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    ranking = []
    for i in range(num):
        ranking.append({'index': i, 'score': cosine_scores[i][i]})
    ranking = sorted(ranking, key=lambda x: x['score'], reverse=True)
    new_gt = []
    count = -1
    for rank in ranking:
        count += 1
        if count < num / 2:
            new_gt.append(generated_text[rank['index']])
    return new_gt
    
if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    #parser.add_argument("--event", default='Boston Marathon bombing', type=str, required=True,
    #                    help="choose your event of interest for schema induction")
    parser.add_argument("--call_n", default=4, type=int, required=False,
                        help="number of pieces of generated text per headline")
    parser.add_argument("--headline_n", default=10, type=int, required=False, 
                        help="number of headlines to be generated")
    parser.add_argument("--debugging", default=0, type=int, required=False,
                        help="debugging mode: True or False")
    args = parser.parse_args()
    
    #scenarios = ['Bombing Attacks', 'Pandemic Outbreak', 'Civil Unrest', 'International Conflict', 'Disaster and Rescue', 'Terrorism Attacks', 'Election', 'Sports Games', 'Kidnapping', 'Business Change', 'Mass Shooting']
    scenarios = []
    dir_name = "/shared/kairos/Data/LDC2020E25_KAIROS_Schema_Learning_Corpus_Phase_1_Complex_Event_Annotation_V4/docs/ce_profile"
    onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == ".txt"]

    for f in onlyfiles:
        scenarios.append(" ".join(f.split("_")[2:-1]))
    
    #with open("generated_text/2021-12-18.pkl", 'rb') as f:
    with open("generated_text/2022-01-06.pkl", 'rb') as f:
    #with open("generated_text/2022-06-10.pkl", 'rb') as f:
        text = pickle.load(f)
        
    if args.debugging:
        topic = "Aviation-accident"
        f_out = open('output/' + topic + '.txt', 'w')
        IE_output = schema_induction('', args.call_n, f_out, gt_input = False, gt_output = False, debugging = args.debugging)
        print("printing stats...")
        #print_stats(IE_output, topic = topic, f_out = f_out)
        f_out.close()
        
    else:
        for topic in scenarios:
            #if path.exists('output_Typing_OnePass/' + topic + '.txt'):
            #if path.exists('GPT2_output/' + topic + '.txt'):
            if path.exists('output_all/' + topic + '.txt'):
                continue
                
            #f_out = open('output_Typing_OnePass/' + topic + '.txt', 'w')
            #f_out = open('GPT2_output/' + topic + '.txt', 'w')
            f_out = open('output_all/' + topic + '.txt', 'w')
            #gt_input = False
            induce = False
            gt_input = text[topic]
            #gt_input = ["They had to account for all the money that had gone missing. The police were acting on a tip from an informer and caught the gang redhanded."]
            
            if gt_input:
                IE_output = schema_induction('', args.call_n, f_out, gt_input, False, args.debugging, True, True, True, False)
                save_IE_output(IE_output, topic)
                try:
                    print("printing stats...")
                    print_stats(IE_output, topic = topic, f_out = f_out)
                except:
                    pass
                
            else:
                print("Generating headline for '{event}'".format(event=topic))
                
                ''' # Manually selecting appropiate headlines
                while True:
                    headline = headline_generator(topic, news = True)
                    x = input("The generated headline for '" + topic + "' is: '" + headline + "'. Enter A (Accept) or R (Reject):")
                    if x == 'A':
                        break
                    elif x == 'R':
                        print("Alright, let's try again")
                    else:
                        print("Enter A (Accept) or R (Reject):")
                '''
                
                generated_text = []
                # generate 10 headlines for news & how-to 
                for i in range(args.headline_n):
                    headline = headline_generator(topic, news = True)
                    print("News-like headline:", headline)
                    # generate call_n pieces of text for each headline
                    generated_text.extend(schema_induction(headline, args.call_n, f_out, gt_input = False, gt_output = True, debugging = args.debugging))
                    
                headline = headline_generator(topic, news = False)
                print("HowTo-like headline:", headline)
                generated_text.extend(schema_induction(headline, args.headline_n * args.call_n, f_out, gt_input = False, gt_output = True, debugging = args.debugging))
                save_generated_text(generated_text, topic)
                
                if induce:
                    IE_output = schema_induction('', args.call_n, f_out, gt_input = filter_gt_sbert(generated_text, topic), gt_output = False, debugging = args.debugging)
                    save_IE_output(IE_output, topic)
                    print("printing stats...")
                    #print_stats(IE_output, topic = topic, f_out = f_out)
            f_out.close()
    end = timer()
    print(timedelta(seconds=end-start))
    
    """
    #This version does not work
    start = timer()
    with open('parrot.pkl', 'rb') as f:
        generated_text = pickle.load(f)
    #print(f'starting computations on {cpu_count()} cores')

    #debug_text = ['The first passengers rescued from a helicopter that ditched in the North Sea have arrived at hospital.', 'The Sea King helicopter, which had been on a search and rescue mission, came down off the coast of the Orkney Islands.']
    text_ids = [i for i in range(len(generated_text))]
    processes = [Process(target=event_extractor, args=(generated_text, text_ids)) for x in range(len(generated_text))]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    #with Pool() as pool:
    #    IE_output = pool.starmap(event_extractor, zip(generated_text, text_ids))

    for SRL_output in IE_output:
        print_events(SRL_output['views'][-1]['viewData'][0]['constituents'])

    end = timer()
    print(f'elapsed time: {end - start}')
    """
    
    
    """
    # Let's try this version... And it works!
    start = timer()
    print(f'starting computations on {cpu_count()} cores')

    #debug_text = ['The first passengers rescued from a helicopter that ditched in the North Sea have arrived at hospital.', 'The Sea King helicopter, which had been on a search and rescue mission, came down off the coast of the Orkney Islands.']
    with open('parrot.pkl', 'rb') as f:
        generated_text = pickle.load(f)
    #generated_text = generated_text[0:3]
    text_ids = [i for i in range(len(generated_text))]

    with Pool(processes=3) as pool:
        IE_output = pool.starmap(event_extractor, zip(generated_text, text_ids))

    for SRL_output in IE_output:
        print_events(SRL_output['views'][-1]['viewData'][0]['constituents'])

    end = timer()
    print(f'elapsed time: {end - start}')
    """
    