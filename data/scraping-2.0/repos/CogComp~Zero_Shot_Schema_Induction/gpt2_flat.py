import os
import openai
#openai.api_key = "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"
#openai.api_key = "sk-tP9LtUEWkDAn9AuhdZuohNGjZnMjWEX2b7NBzPeP"
openai.api_key = "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"

import requests
import json

def SRL(text):
    headers = {'Content-type':'application/json'}
    SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate', json={"sentence": text}, headers=headers)
    if SRL_response.status_code != 200:
        print("SRL_response:", SRL_response.status_code)
    try:
        SRL_output = json.loads(SRL_response.text)
        predicates = []
        
        for view in SRL_output['views']:
            if view['viewName'] in ['SRL_ONTONOTES', 'SRL_NOM_ALL']:
                for constituent in view['viewData'][0]['constituents']:
                    if constituent['label'] == 'Predicate':
                        predicate = {}
                        predicate['predicate'] = constituent['properties']['predicate']
                        predicate['SenseNumber'] = constituent['properties']['SenseNumber']
                        predicate['sense'] = constituent['properties']['sense']
                        predicate['viewName'] = view['viewName']
                        predicates.append(predicate)
                    else:
                        predicates[-1][constituent['label']] = ' '.join(SRL_output['tokens'][constituent['start']:constituent['end']])
        return predicates
    except:
        return []

import json
import requests

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def call_gpt2(prompt, event, n, temperature, max_length, presence_penalty, headline = None, subtopic = 0):
    if not prompt:
        if subtopic:
            prompt="Subtopics of " + event + " are:\n\n1."
        else:
            if headline:
                prompt="Write a news story titled \"" + headline + "\""
                print("--- Generating text for '" + headline + "' ...")
            else:
                prompt="Write a news headline about " + event + ", \""
                print("--- Generating headlines for '" + event + "' ...")
    print("--- prompt:", prompt)
    data = query(
        {
            "inputs": prompt, 
            "parameters": {"max_length": max_length,
                           "num_return_sequences": n,
                          },
        }
    )
    return_text = []
    for gt in data:
        try:
            return_text.append(gt['generated_text'].replace(prompt, ''))
        except:
            continue
    return return_text, None

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class schema:
    def __init__(self, name, hierarchy_num, generator, headline_num = 1, news_per_headline = 15, HowTo_num = 15):
        self.name = name
        self.hierarchy_num = hierarchy_num
        self.generator = generator # call_openai_api
        self.temperature = 0.9
        self.stop = None
        self.presence_penalty = 0.1
        self.headline_num = headline_num
        self.news_per_headline = news_per_headline
        self.HowTo_num = HowTo_num
        self.hierarchy = {name: {'subtopics': [], 'text': {}}}
        
    def subtopic_gen(self, topic):
        texts, response = self.generator(None, topic, 1, self.temperature, 64, self.presence_penalty, headline = None, subtopic = 1)
        print("printing within subtopic_gen():", texts[0])
        predicates = SRL(texts[0].replace('\n', ' '))
        subtopics = set()
        for predicate in predicates:
            if len(subtopics) <= 4 and predicate['predicate'] not in stop_words:
                if 'ARG1' in predicate.keys():
                    subtopics.add(predicate['predicate'] + ' ' + predicate['ARG1'] + " (" + topic + ")")
                else:
                    subtopics.add(predicate['predicate'] + " (" + topic + ")")
        return subtopics
            
    def text_gen_helper(self, event, mode):
        # mode 1: direct generation for steps
        # mode 2: news-style text generation
        # mode 3: how-to article generation
        if mode == 1:
            prompt = "Write essential steps for " + event + ":\n\n1."
            texts, response = self.generator(prompt, event, 1, self.temperature, 256, self.presence_penalty)
            return texts
        if mode == 2:
            news = []
            headlines, response = self.generator(None, event, self.headline_num, self.temperature, 64, self.presence_penalty)
            for headline in headlines:
                end = headline.find("\"")
                headline = headline[:end]
                texts, response = self.generator(None, event, self.news_per_headline, self.temperature, 256, self.presence_penalty, headline)
                for text in texts:
                    news.append(headline + ' ' + text)
            return news
        if mode == 3:
            prompt = "How to make " + event
            texts, response = self.generator(prompt, event, self.HowTo_num, self.temperature, 256, self.presence_penalty)
            return texts
        
    def text_gen(self, event):
        return {'steps': self.text_gen_helper(event, 1),
                'news': self.text_gen_helper(event, 2),
                'HowTo': self.text_gen_helper(event, 3)}
    
    def learning_corpus_gen(self):
        if self.hierarchy_num >= 1:
            self.hierarchy[self.name]['text'] = self.text_gen(self.name)
        if self.hierarchy_num >= 2:
            subtopics = self.subtopic_gen(self.name)
            for subtopic in subtopics:
                print("%%% subtopic of", self.name, ":", subtopic)
                st_dict = {'subtopics': []}
                st_dict['text'] = self.text_gen(subtopic)
                self.hierarchy[self.name]['subtopics'].append({subtopic: st_dict})
                if self.hierarchy_num == 3:
                    subsubtopics = self.subtopic_gen(subtopic)
                    for subsubtopic in subsubtopics:
                        sub_st_dict = {'subtopics': []}
                        sub_st_dict['text'] = self.text_gen(subsubtopic)
                        self.hierarchy[self.name]['subtopics'][-1][subtopic]['subtopics'].append({subsubtopic: sub_st_dict})
    
    def print_hierarchy(self):
        for i in self.hierarchy.keys():
            print(i)
            for subtopic in self.hierarchy[i]['subtopics']:
                for j in subtopic.keys():
                    print(j)
                    for subsubtopic in subtopic[j]['subtopics']:
                        for k in subsubtopic.keys():
                            print(k)
            
from os import listdir
from os.path import isfile, join
dir_name = "/shared/kairos/Data/LDC2020E25_KAIROS_Schema_Learning_Corpus_Phase_1_Complex_Event_Annotation_V4/docs/ce_profile"
onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == ".txt"]

scenarios = ['Bombing Attacks', 'Pandemic Outbreak', 'Civil Unrest', 'International Conflict', 'Disaster and Rescue', 'Terrorism Attacks', 'Election', 'Sports Games', 'Kidnapping', 'Business Change', 'Mass Shooting']
for f in onlyfiles:
    scenarios.append(" ".join(f.split("_")[2:-1]))
print(len(scenarios))
model = 'gpt2'
hier = 1
generated_text = {}
import pickle
'''
with open("generated_text/2022-06-10.pkl", 'wb') as f:
    for scenario in scenarios:
        s = schema(scenario, hier, call_gpt2)
        s.learning_corpus_gen()
        generated_text[scenario] = s.hierarchy[scenario]['text']['news']
        generated_text[scenario] += s.hierarchy[scenario]['text']['HowTo']
    pickle.dump(generated_text, f)
'''    
with open('generated_text/2022-06-10.pkl', 'rb') as f:
    gt = pickle.load(f)
    print(len(gt))
f_11 = open('generated_text/2022-06-11.pkl', 'wb')
    
print(set(scenarios).difference(set(list(gt.keys()))))    
for scenario in scenarios:
    if scenario in gt.keys():
        continue
    else:
        print(scenario)
        s = schema(scenario, hier, call_gpt2)
        s.learning_corpus_gen()
        generated_text[scenario] = s.hierarchy[scenario]['text']['news']
        generated_text[scenario] += s.hierarchy[scenario]['text']['HowTo']
pickle.dump(generated_text, f_11)
f_11.close()
