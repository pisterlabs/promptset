import warnings
warnings.filterwarnings("ignore")

import io
import requests
import json
import yaml
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chains import APIChain
# ----------------------------------------------------------------------------------------------------------------------
def get_chain_chat(LLM):
    chain = load_qa_chain(LLM)
    return chain
# ----------------------------------------------------------------------------------------------------------------------
def get_chain_summary(LLM,question_prompt,chain_type="refine"):
    chain = load_summarize_chain(LLM, question_prompt=question_prompt,chain_type=chain_type,verbose=True)
    return chain
# ----------------------------------------------------------------------------------------------------------------------
def get_chain_Neo4j(LLM,filename_config_neo4j):
    model = LLM
    with open(filename_config_neo4j, 'r') as config_file:
        config = yaml.safe_load(config_file)
        url = f"bolt://{config['database']['host']}:{config['database']['port']}"
        graph = Neo4jGraph(url=url,username=config['database']['user'],password=config['database']['password'])
    chain = GraphCypherQAChain.from_llm(model,graph=graph,verbose=True,return_intermediate_steps=True)
    return chain
# ----------------------------------------------------------------------------------------------------------------------
def yaml_to_json(text_yaml):
    io_buf = io.StringIO()
    io_buf.write(text_yaml)
    io_buf.seek(0)
    res_json = yaml.load(io_buf, Loader=yaml.Loader)
    return res_json
# ----------------------------------------------------------------------------------------------------------------------
def get_api_spec(api_spec,format='json'):

    if api_spec.find('http')>=0:
        api_spec = requests.get(api_spec, verify=False).text
        if format == 'json':
            api_spec = yaml_to_json(api_spec)
    elif api_spec[-5:].find('.json')==0:
        with open(api_spec, 'r') as f:
            api_spec = json.dumps(json.load(f))
            if format == 'json':
                api_spec = yaml_to_json(api_spec)
    elif api_spec[-5:].find('.yaml')==0:
        with open(api_spec, 'r') as f:
            api_spec = yaml.safe_load(f)
    else:
        api_spec = requests.get(api_spec, verify=False).text
        if format == 'json':
            api_spec = yaml_to_json(api_spec)

    return api_spec
# ----------------------------------------------------------------------------------------------------------------------
def get_chain_API(LLM,api_spec):
    api_spec = get_api_spec(api_spec, format='txt')
    chain = APIChain.from_llm_and_api_docs(LLM, api_docs=api_spec, verbose=True,limit_to_domains=['https://www.example.com'])

    return chain
# ----------------------------------------------------------------------------------------------------------------------
def wrap_chain(chain):
    class A(object):
        def __init__(self,chain):
            self.chain = chain
        def run_query(self,query):
            return self.chain(query),[]

    return A(chain)
# ----------------------------------------------------------------------------------------------------------------------