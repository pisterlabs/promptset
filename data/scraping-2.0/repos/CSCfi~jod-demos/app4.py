import flask
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Optional

import os

import openai
from dotenv import load_dotenv
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, ServiceContext, LLMPredictor, PromptHelper
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from gpt_index import LangchainEmbedding

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

import langchain

from yamlconfig import read_config

# ----------------------------------------------------------------------

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)
Bootstrap(app)

cfg = read_config()
debug = cfg['debug']

load_dotenv()

used_embedding = "huggingface" # or "openai"

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# OpenAI models
llm_deployment_name = "jod-text-davinci-003-1"
emb_deployment_name = "text-embedding-ada-002"

# HuggingFace embedding model
# default is all-mpnet-base-v2
#model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = 'TurkuNLP/sbert-cased-finnish-paraphrase'

# Create LLM via Azure OpenAI Service
llm = AzureOpenAI(deployment_name=llm_deployment_name)
llm_predictor = LLMPredictor(llm=llm)
if used_embedding == "openai":
    embedding_model = LangchainEmbedding(OpenAIEmbeddings(model=emb_deployment_name))
elif used_embedding == "huggingface":
    embedding_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))
else:
    assert 0

# Define prompt helper
max_input_size = 3000
num_output = 256
chunk_size_limit = 1000 # token window size per document
max_chunk_overlap = 20 # overlap for each token fragment
prompt_helper = PromptHelper(max_input_size=max_input_size,
                             num_output=num_output,
                             max_chunk_overlap=max_chunk_overlap,
                             chunk_size_limit=chunk_size_limit)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                               embed_model=embedding_model,
                                               prompt_helper=prompt_helper)

index_file = "/home/cloud-user/jod16/notebooks/index-fi-2-huggingface.json"
index = GPTSimpleVectorIndex.load_from_disk(index_file, service_context=service_context)

def queryindex(q):
    response = index.query(q, similarity_top_k=3)
    print('Query:', q)
    print('Source nodes:')
    print(response.source_nodes)
    return str(response)

tools = [
    Tool(
        name="JOD-bot",
        func=queryindex, 
        #func=lambda q: str(index.query(q, similarity_top_k=3)),
        description="Tämä palvelu vastaa kysymyksiin AMK- ja yliopistokoulutuksista ja ammateista. Syötteiden tulee olla suomenkielisiä lauseita.",
        return_direct=True
    ),
]
def reset():
    global memory
    global agent_chain
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)
reset()
    
print('All done')

# ----------------------------------------------------------------------

def get_results(query):
    res = agent_chain.run(input=query)

    mem = []
    for i in range(0, len(memory.chat_memory.messages), 2):
        mem.append((memory.chat_memory.messages[i].content,
                    memory.chat_memory.messages[i+1].content))
        
    return res, mem

# ----------------------------------------------------------------------

class MyForm(FlaskForm):
    name = StringField('Kysy JOD-botilta:',
                       render_kw={'placeholder':
                                  'Kirjoita tähän mitä haluat kysyä JOD-botilta'},
                       validators=[DataRequired()])
    chat_button = SubmitField(label="Lähetä")
    restart_button = SubmitField(label="Aloita alusta")

# ----------------------------------------------------------------------

@app.route("/",methods=["GET"])
def parse_get():
    global p
    txt=flask.request.args.get("text")
    if txt:
        results, memorylist = get_results(txt)
    else:
        results, memorylist = "", []
    
    form = MyForm(meta={'csrf': False})
    return flask.Response(flask.render_template(cfg['app4_html_template'],
                                                results=results,
                                                memory=memorylist,
                                                form=form, debug=debug),
                          mimetype="text/html; charset=utf-8")

@app.route("/",methods=["POST"])
def parse_post():
    form = MyForm(meta={'csrf': False})
    txt = form.name.data
    if not txt:
        return """Error occurred""", 400

    if form.restart_button.data:
        reset()
        results, memorylist = "", []
    else:
        results, memorylist = get_results(txt)

    form.name.data = ""

    return flask.Response(flask.render_template(cfg['app4_html_template'],
                                                results=results,
                                                memory=memorylist,
                                                form=form, debug=debug),
                          mimetype="text/html; charset=utf-8")

# ----------------------------------------------------------------------
