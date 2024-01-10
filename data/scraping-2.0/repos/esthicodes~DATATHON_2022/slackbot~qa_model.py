#!/usr/bin/env python
# coding: utf-8
"""
Created on Sun May 15 02:23:38 2022

@author: The Learning Machine Team
"""

import os
import re
import time
import json
import copy
import openai
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from subprocess import Popen, PIPE, STDOUT

from haystack.utils import launch_es, print_documents
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline

class DataLoader:
    def __init__(self, repo_root_dir):
        self.repo_root_dir = repo_root_dir
        self.dataset_dir = os.path.join(repo_root_dir, 'dataset')
        self.covid_json_path = os.path.join(self.dataset_dir, 'COVID-19 FAQs | Allianz Global Assistance.json')
        self.travel_json_path = os.path.join(self.dataset_dir, 'Travel Insurance FAQs | Allianz Global Assistance.json')
        
        self.stop_words = set(stopwords.words('german'))
        self.stemmer = WordNetLemmatizer()
        
        
    def _preprocess(self, sent):
        # Remove all the special characters
        sent = re.sub(r'\W', ' ', sent)
        # remove all single characters
        sent = re.sub(r'\s+[a-zA-Z]\s+', ' ', sent)
        # Remove single characters from the start
        sent = re.sub(r'\^[a-zA-Z]\s+', ' ', sent) 
        # Substituting multiple spaces with single space
        sent = re.sub(r'\s+', ' ', sent, flags=re.I)
        # Removing prefixed 'b'
        sent = re.sub(r'^b\s+', '', sent)
        # Converting to Lowercase
        sent = sent.lower()
    
        text_tokens = word_tokenize(sent)
        sent = [self.stemmer.lemmatize(word)  for word in text_tokens if not word in self.stop_words]
        sent = (" ").join(sent)
    
        return sent
    
    
    def _load_qa_jsons(self):
        covid_data = pd.read_json(self.covid_json_path)
        travel_data = pd.read_json(self.travel_json_path)
        
        covid_df = pd.json_normalize(covid_data['FaqDocuments'])
        travel_df = pd.json_normalize(travel_data['FaqDocuments'])
        
        self.df = pd.concat([covid_df, travel_df], axis=0)
        
        # self.df["Question"] = self.df['Question'].apply(self._preprocess)
        # self.df["Answer"] = self.df['Answer'].apply(self._preprocess)
        
        questions = self.df["Question"].values.tolist()
        answers = self.df["Answer"].values.tolist()
        
        df_ = copy.deepcopy(self.df)
        df_ = df_.rename(columns={"Answer":"text", "Question":"metadata"})
        df_.to_json (os.path.join(self.dataset_dir, 'qa_exported_jl.json'), orient='records', lines=True)

        return questions, answers
    
    
    def make_document(self):
        questions, answers = self._load_qa_jsons()
        docs = []
        for i in range(len(questions)):
            entry = {"meta" : {"question":questions[i]}, "content" : answers[i]}
            docs.append(entry)
        return docs
          

class HayStacker:
    def __init__(self, docs):
        self.docs = docs
        
        self._start_elastic_search()
        self.document_store = self._build_doc_store()
        
    
    @staticmethod
    def _start_elastic_search():
        launch_es()


    @staticmethod
    def _build_doc_store():
        return ElasticsearchDocumentStore()
    
    
    def bulit_retriver(self, retriver_name="DPR"):
        if retriver_name == "DPR":
            self.retriever = DensePassageRetriever(
                document_store=self.document_store,
                query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                use_gpu=True,
                embed_title=True)
        return self.retriever


    def build_generator(self):
        self.generator = RAGenerator(
            model_name_or_path="facebook/rag-token-nq",
            use_gpu=False,
            top_k=1,
            max_length=200,
            min_length=2,
            embed_title=True,
            num_beams=2)
        return self.generator


    def compute_docs_embeddings(self):
        self.document_store.write_documents(self.docs)
        self.document_store.update_embeddings(self.retriever)


    def build_pipeline(self):
        self.pipe = DocumentSearchPipeline(self.retriever)
        return self.pipe

class GPT3:
    def __init__(self, json_line_path):
        self.json_line_path = json_line_path
        
        opanai_api_key_path = "opanai_api_key.txt"
        openai.api_key_path = opanai_api_key_path
        
        self.opfilename = self._upload_data()
    
    
    def completion_api(self, info, query):
        TAG = "use the above information to answer the question"
        response = openai.Completion.create(
          engine="text-davinci-002",
          prompt=f"information:\n{info}\n\n\n{TAG}: {query}",
          temperature=0.7,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0)

        answer = str(response['choices'][0].to_dict()['text'])[2:]
        
        return answer

    
    def _upload_data(self, first_time=True):
        if first_time:
            opfile = openai.File.create(file=open("dataset/covid_exported_jl.json"), purpose='answers')
            opfilename = opfile.to_dict()['id']
            print("openfilename", opfilename)
        else:
            opfilename = 'file-UqMyZRjqTGYJ4OIO6iNRxWEc'
        return opfilename
        
    
    def answer_api(self, query):
        resp = openai.Answer.create(
            search_model="ada", 
            model="text-davinci-002", 
            question=query, 
            file=self.opfilename, 
            examples_context="In 2017, U.S. life expectancy was 78.6 years.", 
            examples=[["What is human life expectancy in the United States?", "78 years."]], 
            max_rerank=10,
            max_tokens=10,
            stop=["\n", "<|endoftext|>"]
        )

        return resp['selected_documents'][-1]['text']

class Responder:
    def __init__(self, repo_root_dir):
        self.repo_root_dir = repo_root_dir
        self.dataset_dir = os.path.join(repo_root_dir, 'dataset')
        self.json_line_path = os.path.join(self.dataset_dir, 'qa_exported_jl.json')
        
        self.data_loader = DataLoader(repo_root_dir)
        self.docs = self.data_loader.make_document()
        
        self.haystacker = HayStacker(self.docs)
        self.retriever = self.haystacker.bulit_retriver()
        self.generator = self.haystacker.build_generator()
        self.haystacker.compute_docs_embeddings()
        self.pipe = self.haystacker.build_pipeline()
        
        self.gpt3 = GPT3(self.json_line_path)
        
    
    def call_hs(self, query):
        res = self.pipe.run(query=query, params={"Retriever": {"top_k": 2}})
        print_documents(res, max_text_len=512)
        return res
    
    
    def post_to_gpt3(self, info, query):
        comp_ans = self.gpt3.completion_api(info, query)
        answ_ans = self.gpt3.answer_api(query)
        return comp_ans, answ_ans
        

def qa_pipeline(responder, query, info=None, repo_root_dir="", name='hs'):
    if name == 'hs':
        ans = responder.call_hs(query)
        return ans, None
    
    if name == 'gpt3':
        comp_ans, answ_ans = responder.post_to_gpt3(info, query)
        return comp_ans, answ_ans


def setup_responder():
    repo_root_dir = ""
    info = "COVID-19 is a known and evolving epidemic that is impacting travel worldwide, with continued spread and impacts expected.  Our travel protection plans do not generally cover losses directly or indirectly related to known, foreseeable, or expected events, epidemics, government prohibitions, warnings, or travel advisories, or fear of travel. However, we are pleased to announce the introduction of our Epidemic Coverage Endorsement to certain plans purchased on or after March 6, 2021.  This endorsement adds certain new covered reasons related to epidemics (including COVID-19) to some of our most popular insurance plans.  Please see the below FAQ section on “Epidemic Coverage Endorsement” for more information.  Note, the Epidemic Coverage Endorsement may not be available for all plans or in all jurisdictions.  To see if your plan includes this endorsement, please look for “Epidemic Coverage Endorsement” on your Declarations of Coverage or Letter of Confirmation. Additionally, in response to the ongoing public health and travel crisis, we are temporarily extending certain claims accommodations as follows*: 1. For plans that do not include the Epidemic Coverage Endorsement, we are temporarily accommodating claims for the following:  Emergency medical care for an insured who becomes ill with COVID-19 while on their trip (if your plan includes the Emergency Medical Care benefit) Trip cancellation and trip interruption if an insured, or that insured’s traveling companion or family member, becomes ill with COVID-19 either before or during the insured’s trip (if your plan includes Trip Cancellation or Trip Interruption benefits, as applicable)  2. If an insured or their traveling companion become ill with COVID-19 while on their trip, that insured will not be subject to the Trip Interruption benefit’s five-day maximum limit for additional accommodation and transportation expenses (however, the maximum daily limit for such expenses and the maximum Trip Interruption benefit limit still apply). These temporary accommodations are strictly applicable to COVID-19 and are only available to customers whose plan includes the applicable benefit.  These accommodations apply to plans currently in effect but may not apply to plans purchased in the future, so please refer to our Coverage Alert for the most up to date information before purchasing."
    query = "I am worried about COVID-19 impacting a trip I have scheduled or plan to schedule. Should I buy an Allianz travel protection plan to cover me in case COVID-19 impacts my trip"
    
    responder = Responder(repo_root_dir)
    
    return responder

def call_hs(responder, query):
    ### HS
    ans, _ = qa_pipeline(responder, query, name='hs')
    return "hs: " + str(ans)

def call_gpt3(responder, query):
    ### GPT3
    comp_ans, answ_ans = qa_pipeline(responder, query, "", name='gpt3')
    return answ_ans
