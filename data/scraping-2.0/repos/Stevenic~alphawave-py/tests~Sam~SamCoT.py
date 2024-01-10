import os
import traceback
import requests
import json
import nyt
import ipinfo
import random
import socket
import time
import numpy as np
import faiss
import pickle
import hashlib
from datetime import datetime
import openai
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.ChoiceResponseValidator import ChoiceResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from alphawave.alphawaveTypes import PromptCompletionOptions

# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_city_state():
   api_key = os.getenv("IPINFO")
   handler = ipinfo.getHandler(api_key)
   response = handler.getDetails()
   city, state = response.city, response.region
   return city, state

host = '127.0.0.1'
port = 5004

def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

class SamInnerVoice():
    def __init__(self, model):
        self.client = OSClient(api_key=None)
        self.openAIClient = OpenAIClient(apiKey=openai_api_key)

        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.max_tokens = 4000
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.model = model
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.docEs = None
        docHash_loaded = False
        try:
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'rb') as f:
                data = pickle.load(f)
                self.docHash = data['docHash']
                self.metaData = data['metaData']
            docHash_loaded = True
        except Exception as e:
            # no docHash, so faiss is useless, reinitialize both
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'wb') as f:
                data = {}
                data['docHash'] = self.docHash
                data['metaData'] = self.metaData
                pickle.dump(data, f)


    
    def logInput(self, input):
        with open('SamInputLog.txt', 'a') as log:
            log.write(input.strip()+'\n')

    def search_titles(self, query, news_details):
        titles = []; articles = []
        for key in news_details.keys():
            for item in news_details[key]:
                titles.append(item['title'])
                articles.append(item)
        title_embeddings = self.embedder.encode(titles)
        query_embedding = self.embedder.encode(query)
        # Find closest title by cosine similarity
        cos_sims = spatial.distance.cdist([query_embedding], title_embeddings, "cosine")[0]
        most_similar = cos_sims.argmin()
        print(f'similar title {titles[most_similar]}')
        return articles[most_similar]
    
    def sentiment_analysis(self, profile_text):
       if self.docEs is not None:
          return None
       try:
          with open('SamInputLog.txt', 'r') as log:
             inputLog = log.read()

          lines = inputLog.split('\n')
          prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=150)
        
          analysis_prompt_text = f"""Analyze the input from doc below for it's emotional tone, and respond with a few of the prominent emotions present. Note that the later lines are more recent, and therefore more indicitave of current state. Select emotions from the following list: 'Anger', 'Happy', 'Joyful', 'Sad', 'Fearful', 'Surprised', 'Disgusted', 'Anticipatory', 'Excited', 'Relieved', 'Frustrated', 'Guilty', 'Feeling Gratitude',  'Nostalgic', 'Melancholy' 'Resentful', 'Curious', 'Confused', 'Hungry' 'Bored', 'Satisfied', 'Lonely', 'Belonging','Peace', 'Despair'."""

          analysis_prompt = Prompt([
             SystemMessage(profile_text),
             SystemMessage(analysis_prompt_text),
             UserMessage("Doc's input: {{$input}}\n"),
             AssistantMessage(' ')
          ])
          analysis = ut.run_wave (self.client, {"input": lines}, analysis_prompt, prompt_options,
                                  self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                  logRepairs=False, validator=DefaultResponseValidator())

          if type(analysis) is dict and 'status' in analysis.keys() and analysis['status'] == 'success':
             es = analysis['message']['content']
             prompt_text = f"""Given who you are:\n{profile_text}\nand your analysis of doc's emotional state\n{es}\nWhat would you say to him? If so, pick only the one or two most salient emotions. Remember he has not seen the analysis, so you need to explicitly include the names of any emotions you want to discuss. You have only about 100 words.\n"""
             prompt = Prompt([
                UserMessage(prompt_text),
                AssistantMessage(' ')
             ])
             analysis = ut.run_wave (self.client, {"input": lines}, prompt, prompt_options,
                                     self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                     logRepairs=False, validator=DefaultResponseValidator())
             if type(analysis) is dict and 'status' in analysis.keys() and analysis['status'] == 'success':
                response = analysis['message']['content']
             else: response = None
             self.docEs = response # remember analysis so we only do it at start of session
             return response
       except Exception as e:
          traceback.print_exc()
          print(f' idle loop exception {str(e)}')
       return None

    def recall(self, tags, query, retrieval_count=2, retrieval_threshold=.8):
        if tags =='cancel':
            return []
        if tags is None or len(tags) == 0:
            query_tags = ''
        else:
            query_tags = tags.lower().split(',')
        query_embed = self.embedder.encode(query)

        # gather docs matching tag filter
        candidate_ids = []
        vectors = []
        # gather all docs with matching tags
        for id in self.metaData.keys():
            # gather all potential docs
            if len(query_tags) == 0 or any(tag in query_tags for tag in self.metaData[id]['tags']):
                candidate_ids.append(id)
                vectors.append(self.metaData[id]['embed'])

        # add all matching docs to index:
        index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
        index.add_with_ids(np.array(vectors), np.array(candidate_ids))
        distances, ids = index.search(query_embed.reshape(1,-1), min(10, len(candidate_ids)))
        print("Distances:", distances)
        print("Id:",ids)
        texts = []
        timestamps = [self.metaData[i]['timestamp'] for i in ids[0]]
        # Compute score combining distance and recency
        scores = []
        for dist, id, ts in zip(distances[0], ids[0], timestamps):
            age = (datetime.now() - ts).days
            score = dist + age * 0.1 # Weighted
            scores.append((id, score))
        # Sort by combined score
        results = sorted(scores, key=lambda x: x[1])
        print(results)
        
        texts = []
        for idx in range(min(len(results), retrieval_count)):
            if (len(texts) < retrieval_count):
                texts.append(self.docHash[results[idx][0]])
        return texts
    
    def remember(self, tags, text):
        if tags =='cancel':
            return
        tags = tags.lower().split(',')
        embed = self.embedder.encode(text)
        id = generate_faiss_id(text)
        if id in self.docHash:
            print('duplicate, skipping')
        #self.semanticIDMap.add_with_ids(embed.reshape(1,-1), np.array([id]))
        self.docHash[id] = text
        self.metaData[id] = {"tags": tags, "embed": embed, "timestamp":datetime.now()}
        # and save - write docHash first, we can always recover from that.
        with open('SamDocHash.pkl', 'wb') as f:
            data = {}
            data['docHash'] = self.docHash
            data['metaData'] = self.metaData
            pickle.dump(data, f)

    def get_all_tags(self):
        tags = []
        for id in self.metaData.keys():
            if 'tags' in self.metaData[id]:
                tags = list(set(tags) | set(self.metaData[id]['tags']))
        return tags
                        

    def action_selection(self, input, response, profile, news_details):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        #print(news_details.keys())
        self.articles = []
        for key in news_details.keys():
            for item in news_details[key]:
                self.articles.append({"title": item['title'],
                                      #"abstract": item['abstract']
                                      })
        print(f'SamCoT profile {profile}')
        prompt_text = f"""Given the following user-provided recent exchange, your task is to determine if profile\n{profile}\n should act at this time. Only select an action if explicity called for by the user in the recent exchange, or use action <ask> if you need more information. Your priorities are as follows: 
1. Monitor doc's mental and emotional state. This can be inferred from analyzing his input, or using <ask>
2. Respond to doc's input.
3. Continue to explore your interest in the arts, sciences, and philosophy.

The usual default action should be to choose 'none'
The following New York Times articles are available:
{self.articles}

Respond using the following TOML format:
[RESPONSE]
action="<action name>"
value= "<action argument>"
[STOP]


The actions available are:
action_name\naction_argument\tdescription
ask\t<question>\t# ask doc a question.
article\t<article title>\t# retrieve a NYTimes article.
gpt4\t<question>\t# ask gpt4 a question
web\t<search query string>\t# perform a web search, using the <search query string> as the subject of the search.
wiki\t<search query string>\t# search the local wikipedia database.
none\t'none'\t# no action is needed.
"""
        action_validation_schema={
            "action": {
                "type":"string",
                "required": True,
                "meta": "<action to perform>"
            },
            "value": {
                "type":"string",
                "required": True,
                "meta": "argument for action"
            }
        }


        #print(f'action_selection {input}\n{response}')
        prompt = Prompt([
            SystemMessage(prompt_text),
            UserMessage('Exchange: {{$input}}'),
            AssistantMessage(' ')
        ])
        prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=50)

        summary_prompt_text = f"""Summarize the information in the following text with respect to its title {{$title}}. Do not include meta information such as description of the content, instead, summarize the actual information contained."""

        summary_prompt = Prompt([
            SystemMessage(summary_prompt_text),
            UserMessage('{{$input}}'),
            AssistantMessage(' ')
        ])
        summary_prompt_options = PromptCompletionOptions(completion_type='chat', model='gpt-3.5-turbo', max_tokens=240)

        text = 'doc: '+input+'\nSam: '+response
        analysis = ut.run_wave (self.client, {"input": text}, prompt, prompt_options,
                              self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                              logRepairs=False, validator=TOMLResponseValidator(action_validation_schema))
      
        print(f'SamCoT analysis {analysis}')
        if type(analysis) == dict and 'status' in analysis and analysis['status'] == 'success':
            content = analysis['message']['content']
            print(f'SamCoT content {content}')
            if type(content) == dict and 'action' in content.keys() and content['action']=='article':
                title = content['value']
                print(f'Sam wants to read article {title}')
                article = self.search_titles(title, news_details)
                url = article['url']
                print(f' requesting url from server {title} {url}')
                response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}')
                data = response.json()
                #print(f'summarizing {data["result"]}')
                
                summary = ut.run_wave(self.openAIClient, {"input":data['result']}, summary_prompt, summary_prompt_options,
                                      self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                                      logRepairs=False, validator=DefaultResponseValidator())
                                      
                print(f'retrieved article summary:\n{summary}')
                if type(summary) is dict and 'status' in summary.keys() and summary['status']=='success':
                    return {"result":  summary['message']['content']}
                else:
                    return {"result": ''}

            if type(content) == dict and 'action' in content.keys() and content['action']=='web':
                return {"web":content['value']}
            if type(content) == dict and 'action' in content.keys() and content['action']=='wiki':
                return {"wiki":content['value']}
            if type(content) == dict and 'action' in content.keys() and content['action']=='ask':
                return {"ask":content['value']}
            if type(content) == dict and 'action' in content.keys() and content['action']=='gpt4':
                return {"gpt4":content['value']}
            

    def wakeup_routine(self):
        self.nytimes = nyt.NYTimes()
        self.news, self.details = self.nytimes.headlines()
        city, state = get_city_state()
        print(f"My city and state is: {city}, {state}")
        local_time = time.localtime()
        year = local_time.tm_year
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
        month_num = local_time.tm_mon
        month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
        month_day = local_time.tm_mday
        hour = local_time.tm_hour
        if hour < 12:
            return 'Good morning Doc!'
        if hour < 17:
            return 'Good afternoon Doc!'
        else:
            return 'Hi Doc.'
        # check news for anything interesting.
        # check todos
        # etc
        pass



    def idle(self, profile_text):
       global es
       es = self.sentiment_analysis(profile_text)
       return es

    

if __name__ == '__main__':
    sam = SamInnerVoice(model='alpaca')
    #print(sam.news)
    #print(sam.action_selection("Hi Sam. We're going to run an experiment ?",  'I would like to explore ', sam.details))
    #print(generate_faiss_id('a text string'))
    #sam.remember('language models','this is a sentence about large language models')
    #sam.remember('doc', 'this is a sentence about doc')
    #print(sam.recall('doc','something about doc', 2))
    #print(sam.recall('language models','something about doc', 2))
    #print(sam.recall('','something about doc', 2))
