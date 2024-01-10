### code for hackathon
from langchain.document_loaders import ArxivLoader
import replicate
import sys
import time

from flask import Flask, request, render_template
from flaskext.markdown import Markdown

"""
Terminal application to identify collaborators in a field of research.
Identifying the leading experts and potential collaborators in a particular domain or domains
"""
import json 
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import time
from annoy import AnnoyIndex
import operator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)
Markdown(app)



path1 = "arxiv_12_28_2022.json"
path2 = "since_dec22.json"

data_dict = {}

def preprocess(path):
    start = time.time()
    data = []
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

    data_dict[path] = data

    return sents

def generate_annoy(fn):
    n_trees = 256           #Number of trees used for Annoy. More trees => better recall, worse run-time
    embedding_size = 768    #Size of embeddings
    top_k_hits = 5        #Output k hits

    annoy_index = AnnoyIndex(embedding_size, 'angular')


    annoy_index.load(fn)
    return annoy_index

def search(query,annoy_index,model, topK):
    query_embedding = model.encode(query,convert_to_numpy=True)
    tokens = query.strip().split()
    

    hit_dict = {}
    hits = annoy_index.get_nns_by_vector(query_embedding, topK, include_distances=True)
    for hit in hits[0]:
        if hit not in hit_dict:
            hit_dict[hit] = 0
        hit_dict[hit] += 1

    for i in range(2,len(tokens)-1):

        gram = ' '.join(tokens[:i])
        if (len(gram) < 3):
            continue
        hits = annoy_index.get_nns_by_vector(query_embedding, topK, include_distances=True)
        for hit in hits[0]:
            if hit not in hit_dict:
                hit_dict[hit] = 0
            hit_dict[hit] += 1

    sorted_hits = sorted(hit_dict.items(), key=lambda item: item[1], reverse=True)    
    hits = []
    for (x,y) in sorted_hits:
        hits.append(x)
    return [hits, hits]

    
def collect_results(hits,sents, path, prefix=""):
    #print ("entering results")
    response = ""
    count = 0
    for i in range(len(hits[0])):
        #print ("result " + prefix + str(i))
        response += "## **result " + prefix + str(i) + "** \n "
        response += "## **id:" + data_dict[path][hits[0][i]]['id'] + "** \n "
        response += "### ***Title:" + sents[hits[0][i]].split('[SEP]')[0] + "*** \n "
        response += "\n\n"
        response += "Abstract: " + sents[hits[0][i]].split('[SEP]')[1] + "\n "
        response += "Authors:" +  data_dict[path][hits[0][i]]['authors'] + "\n\n\n "
        #print(sents[hits[0][i]].split('[SEP]')[0])
        #print(sents[hits[0][i]].split('[SEP]')[1])
        #print ("----------------------------------")
    return response

sents_from_path2 = preprocess(path2)

    ## indices from Annoy
an2 = generate_annoy("annoy_index_since_dec22.ann")

model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')



@app.route('/')
def my_form():
    return render_template('my-form.html')


@app.route('/', methods=['POST'])
def my_form_post():
  text = request.form['text']
  field = request.form['newfield']
  if field.strip() == "":
      field = "random new field"
  URL = text
  if (URL.endswith(".pdf")):
    URL = URL[:-4]
  URL = URL.replace("https://arxiv.org/abs/", "")
    
  docs = ArxivLoader(query=URL, load_max_docs=2).load()
  length= len(docs[0].page_content)
  page_content = docs[0].page_content[0:length]

  output = replicate.run(
     "01-ai/yi-34b-chat:914692bbe8a8e2b91a4e44203e70d170c9c5ccc1359b283c84b0ec8d47819a46",
     input={
        "top_k": 50,
        "top_p": 0.8,
        "prompt": "You are RESEARCH ASSISTANT. Read the research article below and generate 4 key points from it. Give me an Idea of how to apply this to the field of " + field + "  Bolden the Idea \n\n" + docs[0].metadata["Title"] + page_content,
        "temperature": 0.3,
        "max_new_tokens": 1024,
        "prompt_template": "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "repetition_penalty": 1.2
        }
        )
  

  print_stuff = docs[0].metadata['Title'] + "\n\n"
  for item in output:
     print_stuff += item
  print(print_stuff)

  print_stuff += "\n\n # ***Summary of related work in 2023*** \n\n"



  response2 = collect_results(search(docs[0].metadata['Title'],an2,model,10),sents_from_path2, path2)


  output_response2 = replicate.run(
     "01-ai/yi-34b-chat:914692bbe8a8e2b91a4e44203e70d170c9c5ccc1359b283c84b0ec8d47819a46",
     input={
        "top_k": 50,
        "top_p": 0.8,
        "prompt": "You are RESEARCH ASSISTANT. Read the research articles below and generate a summary. \n\n" +response2,
        "temperature": 0.3,
        "max_new_tokens": 1024,
        "prompt_template": "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "repetition_penalty": 1.2
        }
        )
  for item in output_response2:
      print_stuff += item
  return render_template('output.html', mkd=print_stuff)

    





