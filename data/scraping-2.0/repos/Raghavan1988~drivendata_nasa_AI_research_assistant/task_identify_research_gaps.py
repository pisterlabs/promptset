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
from getpass import getpass

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
    for i in range(len(hits[0])):
        #print ("result " + prefix + str(i))
        response += "result " + prefix + str(i) + "\n"
        response += "id:" + data_dict[path][hits[0][i]]['id'] + "\n"
        response += "Title:" + sents[hits[0][i]].split('[SEP]')[0] + "\n"
        response += "Abstract: " + sents[hits[0][i]].split('[SEP]')[1] + "\n"
        response += "Authors:" +  data_dict[path][hits[0][i]]['authors'] + "\n"
        response += "Published: " + data_dict[path][hits[0][i]]['versions'][0]['created'] + "\n"
        #print(sents[hits[0][i]].split('[SEP]')[0])
        #print(sents[hits[0][i]].split('[SEP]')[1])
        #print ("----------------------------------")
    return response



##### Start code
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')
OPENAI_API_KEY = getpass("Please enter your OPEN AI API KEY to continue:")
## load the OPENAI LLM model
open_ai_key = OPENAI_API_KEY

sents_from_path1 = preprocess(path1)
sents_from_path2 = preprocess(path2)

## indices from Annoy
an = generate_annoy("annoy_index.ann")
an2 = generate_annoy("annoy_index_since_dec22.ann")


llm = OpenAI(openai_api_key=open_ai_key, model_name= "gpt-3.5-turbo-16k")

while True:
    print ("===============================")

    query = input("Describe the field (or subfield) as a query:")
    print ("===============================")

    print (query)
    response1 = collect_results(search(query,an,model,30),sents_from_path1, path1)
    response2 = collect_results(search(query,an2,model,10),sents_from_path2, path2)

    response = response1 + "\n" + response2
    if (len(response) > 80000):
        response = response[:80000]
    print (len(response))


    template = """ Take a deep breath. 
    You are a researcher in the FIELD of {query}. You are tasked with identifying research gaps and opportunities. User entered the FIELD. The TOP ARTICLES from the search engine are {response}. Analyze the results and follow the instructions below to identify research gaps in the field of {query}.
    TASK DESCRIPTION 
    In this task, your goal is to recognize areas within a specified FIELD where further research is needed. 
    By analyzing existing literature and TOP ARTICLES , data, and perhaps the trajectories of ongoing research, the AI can help point out topics or questions that haven't been adequately explored or resolved. This is essential for advancing knowledge and finding new directions that could lead to significant discoveries or developments
    For example, consider the domain of renewable energy. one might analyze a vast number of research papers, patents, and ongoing projects in this domain. It could then identify that while there's abundant research on solar and wind energy, there might be a relative lack of research on harnessing tidal energy or integrating different types of renewable energy systems for more consistent power generation.
    By identifying these research gaps, you will assisint in pinpointing areas where researchers can focus their efforts to potentially make novel contributions. Moreover, it can help funding agencies and institutions to direct resources and support towards these identified areas of opportunity, fostering advancements that might otherwise be overlooked.
   
    INSTRUCTIONS:
    1. Analyze each article returned by the search engine.
    2. Identify the research gaps and opportunities for each article
    3. Summarize the research gaps and opportunities in the field of {query} across ARTICLES. The research gap should BE GROUNDED on the articles returned by the search engine.
    3. Identify an orthogonal field and generate keyword to get articles from that field that can solve the research gaps.
    4. Identify a related field and generate keyword to get articles from that field that can solve the research gaps
    5. Generate a JSON Response that STRICTLY follow the TYPESCRIPT SCHEMA of class Response below
    6. Output MUST BE a JSON  with the following fields in the SCHEMA.
    SCHEMA:
    class Response:
        orthogonal_field: String
        keyword_orthogonal_field: String
        related_field: String
        keyword_related_field: String
        research_gaps: String
        opportunies: String
    
    """
    prompt = PromptTemplate(template=template, input_variables=["query", "response"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = ""

    try:
        output = llm_chain.run({'query': query, 'response':response})
        print (output)
        input()
    except Exception as e:
        print (e)
        continue

    llm_response = json.loads(output)
    response1 = collect_results(search(llm_response["keyword_orthogonal_field"],an,model,30),sents_from_path1, path1)
    response2 = collect_results(search(llm_response["keyword_orthogonal_field"],an2,model,10),sents_from_path2, path2)

    response = response1 + "\n" + response2
    if (len(response) > 60000):
        response = response[:60000]

    orthogonal_field = llm_response["keyword_orthogonal_field"]
    research_gap = llm_response["research_gaps"]
    opportunies = llm_response["opportunities"]

    ## cross pollination

    template = """ Take a deep breath.  
    You are a researcher in the FIELD1 of {query}. You are tasked with exploring ideas from another FIELD2 of {orthogonal_field} and identify actionable ideas and fix the research gaps. 
    RESEARCH GAP: You are given the problem of {research_gap} in the FIELD1.
    OPPORTUNITIES : You are told the potential opportunities are {opportunies}
    Search Engine returned top articles from FIELD2 {response}.

    Task description:
    You are expected to identify actionable ideas from FIELD2 that can fix the research gap of FIELD1. Ideas should be grounded on the articles returned by the search engine.
    Example 1: The application of concepts from physics to biology led to the development of magnetic resonance imaging (MRI) which is now a fundamental tool in medical diagnostics.
    Example 2: Concepts from the field of biology (e.g., evolutionary algorithms) have been applied to computational and engineering tasks to optimize solutions in a manner analogous to natural selection.
    Follow the instructions below to identify actionable ideas from FIELD2  and fix the research gaps of FIELD1 

    INSTRUCTIONS:
    1. Read every article of FIELD2 from Search engine response and identify actionable ideas that can fix the RESEARCH GAP of FIELD1
    2. Summarize the actionable ideas in the field of FIELD2 that can fix the RESEARCH GAP of FIELD1
    3. Generate a JSON Response that STRICTLY follow the TYPESCRIPT SCHEMA of class Response below
    4. Output MUST BE a JSON ARRAY  with the following fields in the SCHEMA.
    5. Actionable idea should be a sentence or a paragraph that can fix the research gap of FIELD1
    5. Reason should be a sentence or a paragraph that explains why the actionable idea can fix the research gap of FIELD1 using the ideas of FIELD2
    6. GENERATE upto 10 ACTIONABLE IDEAS and atleast 3 ACTIONABLE IDEAS
    SCHEMA:
    Class Response:
        actionable_ideas: String
        reason: String

    """
    prompt = PromptTemplate(template=template, input_variables=["query", "orthogonal_field", "research_gap", "opportunies", "response"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = ""

    try:
        output = llm_chain.run({'query': query, 'orthogonal_field':orthogonal_field, 'research_gap':research_gap, 'opportunies':opportunies, 'response':response})
        print (output)
    except Exception as e:
        print (e)
        continue


