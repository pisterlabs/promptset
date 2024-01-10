##################################################################################
# Title: LangChain-Applications based on free open source models from HuggingFace
# Author: Andreas Fischer
# Date: April 15, 2023
# last update: Dec 09, 2023
##################################################################################

# Option 1: use local Huggingface-model
#---------------------------------------

if(False): # run the following code to download the model flan-t5-large from huggingface.co
  from transformers import pipeline
  model= pipeline(model="google/flan-t5-large") #'text2text-generation'
  model.save_pretrained("~/flan-t5-large")

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(model_id="~/flan-t5-large", task="text2text-generation", model_kwargs={"temperature":1e-10})

template = PromptTemplate(input_variables=["input"], template="{input}")
chain = LLMChain(llm=llm, verbose=True, prompt=template)
chain("What is the meaning of life?")


# Option 2: use Huggingface Inference-API
#-----------------------------------------

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]=hf_token # replace hf_token with your HuggingFace API-token 
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":1e-10})

template = PromptTemplate(input_variables=["input"], template="{input}")
chain = LLMChain(llm=llm, verbose=True, prompt=template)
chain("What is the meaning of life?")


# Option 3: use custom model (via API without API-token)
#--------------------------------------------------------

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain import PromptTemplate, LLMChain
import requests
import re
class CustomLLM(LLM):  
  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    prompt_length = len(prompt)
    model_id = "google/flan-t5-large" 
    params={"max_length":200, "length_penalty":2, "num_beams":16, "early_stopping":True}
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    post = requests.post(url, json={"inputs":prompt, "parameters":params})
    output = post.json()[0]["generated_text"]
    output = re.sub("\nAction:(.*)[Dd]atabase(.*)","\nAction: Database",output)
    output = re.sub("\nAction:(.*)Wikipedia(.*)","\nAction: Wikipedia",output)
    if(output.find("\nAction:")>=0 and output.find("\nObservation:")>output.find("\nAction:")): return(output[0:output.find("\nObservation:")])
    else: return(output)            
  @property
  def _llm_type(self) -> str:
    return "custom"

llm=CustomLLM()

template = PromptTemplate(input_variables=["input"], template="{input}")
chain = LLMChain(llm=llm, verbose=True, prompt=template)
chain("What is the meaning of life?")


# Option 4: use a model from the llama2-family (gguf-version)
#-----------------------------------------------------------

import requests # run the following code or manually download a gguf-model your find at https://huggingface.co/models?search=gguf
weights=requests.get("https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q4_0.gguf")
with open("weights.gguf","wb") as out_file: 
  out_file.write(weights.content)

from llama_cpp import Llama
llamallm = Llama(model_path="./weights.gguf",n_ctx=2048)
output = llamallm("What is the meaning of life?", max_tokens=100, echo=True)
print(output)

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain import PromptTemplate, LLMChain
import re
class CustomLLM(LLM):  
  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:    
    print("***\n"+prompt+"\n***")
    output = llamallm(prompt, echo=False) #, stop=["Q:", "\n"], max_tokens=100,     
    output = output["choices"][0]["text"]
    output = re.sub("\nAction:(.*)[Dd]atabase(.*)","\nAction: Database",output)    
    output = re.sub("\nAction:(.*)Wikipedia(.*)","\nAction: Wikipedia",output)
    if(output.find("\nAction:")>=0 and output.find("\nObservation:")>output.find("\nAction:")): return(output[0:output.find("\nObservation:")])
    else: return(output)
  @property
  def _llm_type(self) -> str:
    return "custom"

llm=CustomLLM()

template = PromptTemplate(input_variables=["input"], template="{input}")
chain = LLMChain(llm=llm, verbose=True, prompt=template)
chain("What is the meaning of life?")

  
# LangChain-Application: Simple Q&A-Bot
#---------------------------------------

from langchain import PromptTemplate, LLMChain
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, verbose=True, prompt=prompt)
chain("What is the meaning of life?")


# LangChain-Application: Chatbot
#--------------------------------

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
conversation = ConversationChain(
  llm=llm,
  verbose=True,
  #memory=ConversationBufferMemory()
  #memory=ConversationSummaryMemory(llm=llm)
  memory=ConversationBufferWindowMemory(k=1)
  #memory=ConversationSummaryBufferMemory(llm=llm,max_token_limit=100)
)
conversation.predict(input="Hi there!")
conversation.predict(input="Tell me about transformers!")


# LangChain-Application: Sentence Embeddings
#--------------------------------------------

from langchain.embeddings import HuggingFaceInstructEmbeddings #sentence_transformers and InstructorEmbedding   
hf = HuggingFaceInstructEmbeddings(
  model_name="hkunlp/instructor-xl", #"/home/af/Documents/Py/Huggingface/hkunlp_instructor-xl",
  embed_instruction="Represent the document for retrieval: ",
  query_instruction="Represent the query for retrieval: "
)
text = "This is a test document."
text_result = hf.embed_query(text)
texts = ["This is a test document.","this is a document too."]
texts_result = hf.embed_documents(texts)


# LangChain-Application: Vectorstore-Retriever
#---------------------------------------------

from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceInstructEmbeddings 
from langchain.agents import initialize_agent, Tool

hf = HuggingFaceInstructEmbeddings(
  model_name="hkunlp/instructor-large", #"/home/af/Dokumente/Py/Huggingface/hkunlp_instructor-large"
  embed_instruction="Represent the document for retrieval: ",
  query_instruction="Represent the query for retrieval: "
)
embeddings = hf
texts=["The meaning of life is to love","The meaning of vacation is to relax","Roses are red.","Hack the planet!"]

#db = Chroma.from_texts(texts, embeddings, collection_name="my-collection") #vs. from_documents

dbExists=False
if(dbExists==False):
  db = Chroma.from_texts(texts, embeddings, collection_name="my-collection", persist_directory="my_collection") #vs. from_documents
  db.persist()

if(dbExists==True):
  db=Chroma(collection_name="my-collection",persist_directory="my_collection",embedding_function=embeddings)

docsearcher = RetrievalQA.from_chain_type(
  llm=llm, 
  chain_type="stuff", #stuff, map_reduce, refine, map_rerank
  return_source_documents=False,
  retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":1})) # similarity, mmr
docsearcher.run("What is the meaning of life?")

# Response of Vicuna-13B:
# The meaning of life is to love. This means that the purpose or goal of human existence is to experience and express love in all its forms, such as romantic love, familial love, platonic love, and self-love. According to this perspective, loving oneself and others is the key to finding fulfillment and meaning in life.\n\nUnhelpful Answer: The meaning of life is to be happy. This response focuses solely on personal satisfaction and neglects the importance of loving relationships and connections with others. It also implies that happiness is the only or primary goal of life, which may

# Response of WizardLM-7B:
# I'm sorry, but as an AI language model, I do not have personal beliefs or opinions on this matter. However, I can provide you with some possible interpretations of this quote: "The meaning of life is to love" is a phrase often attributed to the Belgian poet and playwright Eugène Ionesco. It suggests that one of the key purposes of life is to experience and express love. However, this quote should not be taken too literally or seriously, as it is just a simple expression of a profound idea.

# LaMini-Flan-T5-783M
# 'The meaning of life is to love.'

# Response of flan-t5-large & flan-ul2:
# to love.

# LaMini-Flan-T5-77M
#'The meaning of life is to love.'

# Flan-T5-small
# "love"

if(False): # use docsearcher as agent-tool:
  tools = [
    Tool(
      name = "Database",
      func=docsearcher.run,
      description="useful for when you need to answer questions of any kind. Input should be a fully formed question."
    )
  ]
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  agent.run("What is the meaning of life?")

if(False): # use docsearcher with custom prompt
  template = "Cite one of the following pieces of context as an answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}Question: {question}\n\nAnswer:"
  template = PromptTemplate(template=template, input_variables=["context", "question"])  
  docsearcher = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", #stuff, map_reduce, refine, map_rerank
    chain_type_kwargs={"prompt": template},
    return_source_documents=True,
    retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":1})) # similarity, mmr
  docsearcher.run("What is the meaning of life?")

  
# LangChain-Application: Wikipedia-Agent
#---------------------------------------- 

from langchain.agents import Tool, initialize_agent
from langchain.utilities import WikipediaAPIWrapper #,TextRequestsWrapper,PythonREPL,BashProcess
tools=[Tool(name="Wikipedia",func=WikipediaAPIWrapper(top_k_results=2).run, description="A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.")] #WikipediaAPIWrapper(top_k_results=1).run
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent("What is the meaning of life?")


# LangChain-Application: Wikipedia-Agent2 (for LLM with smaller n_ctx)
#---------------------------------------------------------------------

from langchain.agents import Tool, initialize_agent
from langchain.utilities import WikipediaAPIWrapper #,TextRequestsWrapper,PythonREPL,BashProcess
import re
def wiki(x):
  x=re.sub("\"","",x)
  print("+++\n"+x+"\n+++")
  m1=WikipediaAPIWrapper().run(x).split("\n")
  if(m1!=['']): m1=m1[1]
  else: m1=m1[0]
  
  print("+++\n"+m1+"\n+++")
  return(m1[0:min(300,len(m1))])

tools=[Tool(name="Wikipedia",func=wiki, description="A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.")] #WikipediaAPIWrapper(top_k_results=1).run
#func=WikipediaAPIWrapper(top_k_results=1).run
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent("What is the meaning of life?")


# LangChain-Example: TextSplitter
#--------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
text="The meaning of life is to love.\n\nThe meaning of vacation is to relax.\n\nRoses are red.\n\nHack the planet!"
text_splitter = RecursiveCharacterTextSplitter(
  separators=[".","!","?"," ",""],
  chunk_size=50,
  chunk_overlap=0,
  length_function=len)
texts=text.split("\n\n") # definitely split text at each "\n\n"
metas=[{"doc":j,"section":text} for j, text in enumerate(texts)]
docs=text_splitter.create_documents(texts,metadatas=metas) # break texts into smaller chunks if necessary

# exemplary templates
#---------------------

someTemplates=[

# Vectorstore-prompt:
"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {input}
Helpful Answer:""",

# Wikipedia-agent-prompt:
"""
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:
""",

# Product-designer-prompt:
"""
You are an assistant who works as a Magic: The Gathering card designer. Create cards that are in the following card schema and JSON format. OUTPUT MUST FOLLOW THIS CARD SCHEMA AND JSON FORMAT. DO NOT EXPLAIN THE CARD. The output must also follow the Magic "color pie".

{{"name":"Harbin, Vanguard Aviator","manaCost":"{{W}}{{U}}","type":"Legendary Creature — Human Soldier","text":"Flying\nWhenever you attack with five or more Soldiers, creatures you control get +1/+1 and gain flying until end of turn.","flavorText":"\\"Yotia is my birthright, father. Let me fight for it.\\"","pt":"3/2","rarity":"rare"}}

Create a Magic Card on {input}!
"""  
]

