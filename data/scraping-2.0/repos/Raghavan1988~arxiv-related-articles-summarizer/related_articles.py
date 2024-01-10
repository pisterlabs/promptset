#wget https://arxiv-r-1228.s3.us-west-1.amazonaws.com/arxiv-metadata-oai-snapshot.json
#wget https://arxiv-r-1228.s3.us-west-1.amazonaws.com/annoy_index.ann
## Download the .ann and .json files from Amazon S3 

import json 
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import time
import arxiv
from annoy import AnnoyIndex
import operator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import ArxivLoader


path = "arxiv-metadata-oai-snapshot.json"

data = []
def preprocess(path):
    start = time.time()
    print("Begin Loading of Json file:"+path)
    with open(path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    end = time.time()
    print("Time taken to load json file: ",end-start)

    start = time.time()
    sents = []
    for i in range(len(data)):
        sents.append(data[i]['title']+'[SEP]'+data[i]['abstract'])
    end = time.time()
    print("Time taken to create sentences: ",end-start)  

    return sents

def generate_annoy(fn):
    n_trees = 256           #Number of trees used for Annoy. More trees => better recall, worse run-time
    embedding_size = 768    #Size of embeddings
    top_k_hits = 5        #Output k hits

    annoy_index = AnnoyIndex(embedding_size, 'angular')


    annoy_index.load(fn)
    return annoy_index

def search(query,annoy_index,model):
    query_embedding = model.encode(query,convert_to_numpy=True)
    top_k_hits = 5
    hits = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
    return hits

    
def print_results(hits,sents):
    print ("entering results")
    response = ""
    for i in range(len(hits[0])):
        response += "<br>"
        response += "<b><a href=https://arxiv.org/abs/" + data[hits[0][i]]['id'] +">" + sents[hits[0][i]].split('[SEP]')[0] + "</a>"
        response += "</b><br>"
        response += "Abstract:" + sents[hits[0][i]].split('[SEP]')[1]
        response += "Authors:" +  data[hits[0][i]]['authors']
        response += "<br>"
     
        print ("result " + str(i))
        print ("Title:" + sents[hits[0][i]].split('[SEP]')[0])
        print ("Authors:" +  data[hits[0][i]]['authors'] )
        print ("ID:" + data[hits[0][i]]['id'])
    return response

model = SentenceTransformer('sentence-transformers/allenai-specter', device='cuda')
sents = preprocess(path)
an = generate_annoy("annoy_index.ann")
open_ai_key = "Enter your KEY HERE"

llm = OpenAI(openai_api_key=open_ai_key, model_name= "gpt-3.5-turbo-16k")


from flask import Flask, request

app = Flask(__name__)
@app.route('/search', methods=['GET'])
def search_endpoint():
    args = request.args
    query = args.get('q')
    tokens = query.split('/')
    id = tokens[-1] ## https://arxiv.org/abs/2310.03717

    docs = ArxivLoader(query=id, load_max_docs=2).load()
    print(docs[0].metadata['Title'])
    title = docs[0].metadata['Title']
    abstract = docs[0].metadata['Summary']
    page_content = docs[0].page_content[:40000]
    
    article = title +  '[SEP]' + abstract  + '[SEP]' + page_content

    print("Some related papers:")
    related = search(article,an,model)
    html_response = print_results(search(article,an,model), sents)
    template = """ Take a deep breath. You are a researcher. Your task is to read the  RESEARCH ARTICLE and generate 3 KEY POINTS of it in your own words and generate AN IDEA OF FUTURE EXTENSION based on the RELATED ARTICLES. Generate one actionable idea for extending the RESEARCH.
    RESEARCH ARTICLE: {article}
    RELATED ARTICLES: {related}
    INSTRUCTIONS
    1. Read the TITLE, ABSTRACT and the CONTENT of the RESEARCH ARTICLE.
    2. Generate 3 KEY POINTS of the RESEARCH ARTICLE in your own words. Each Key Point should be a bullet point of 10 WORDS are less.
    3. Read the RELATED ARTICLES
    4. Generate an IDEA OF FUTURE EXTENSION of the RESEARCH ARTICLE based on the RELATED ARTICLES.
    5. The IDEA OF FUTURE EXTENSION should be ONE sentence.
    6. Generate one actionable idea for extending the RESEARCH with Light Bulb emoji.
    7. STRICTLY generate the response in json format using the TYPESCRIPT SCHEMA below. Insert a line break after each bullet point.
    SCHEMA
    response:
        KEYPOINT1: String,
        KEYPOINT2: String,
        KEYPOINT3: String,
        FUTURE_EXTENSION_IDEA: String,
        ACTIONABLE_IDEA: String"""
    prompt = PromptTemplate(template=template, input_variables=["article", "related"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = ""
    try:
        output = llm_chain.run({'article': article, 'related':related})
        print (output)
    except Exception as e:
        print (e)

    html_content = "<html> <head> <title> Arxiv Summary for  " + title + " </title> </head> <body> <h1> "+ title + " </h1>  <h2> Summary </h2> <p>"
    jsonD = json.loads(output)
    html_content += "<br> 1." + jsonD['KEYPOINT1']
    html_content += "<br> 2." + jsonD['KEYPOINT2']
    html_content += "<br> 3. " + jsonD['KEYPOINT3']
    html_content += "<br> <b>FUTURE EXTENSION IDEA:</b> " + jsonD['FUTURE_EXTENSION_IDEA']
    html_content += "<br> <b>ACTIONABLE IDEA: </b>" + jsonD['ACTIONABLE_IDEA']

    html_content += "</p> Related Articles: <br>"


    html_content += html_response
    html_content += "</html>"
    return html_content




   
