# import chromadb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from langchain.output_parsers import OutputFixingParser
import json
import os
import langchain
import openai
import sys
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import ast


from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import HumanMessage


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate
import logging
from langchain.schema.messages import HumanMessage, SystemMessage

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Dict, TypedDict

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from chromadb.utils import embedding_functions
import chromadb 

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


os.environ["OPENAI_API_KEY"] = ""
# openai.api_key = os.environ.get("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = os.environ.get("OPENAI_API_KEY"),model_name="text-embedding-ada-002")

persist_directory = '/home/arman/allsides/chroma/'
db_client = chromadb.PersistentClient(path=persist_directory)

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.4)
llm_comprehension = ChatOpenAI(model_name='gpt-4',temperature=0.4)


try:
  reddit_collection = db_client.get_collection("reddit_v2")
except:
  print("No collection found")

langchain_chroma_reddit = Chroma(
  client=db_client,
  collection_name=reddit_collection.name,
  embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))

try:
  congress_collection = db_client.get_collection("congress")
except:
  print("No collection found")



langchain_chroma_congress = Chroma(
  client=db_client,
  collection_name=congress_collection.name,
  embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))


print("Here are your available collections: ", db_client.list_collections())
# # - After providing the argument diagram, provide a list of topics that are commonly used in arguments made about {topic}, using both of the contexts provided and general knowledge. For each concept or topic cite whether it is from the Reddit context (author and/or forum), Congressional Hearing (Title and Speaker name) or general knowledge.


class RedditPromptTemplate(StringPromptTemplate):

  def format(self, **kwargs) -> str:
    context_query = """
    "Reddit Arguments: "{reddit_context} \n
    "Congress Argument: " {congress_context} \n    

    # If there is reddit_context cite each argument as (Source: Reddit) if its empty do note cite Reddit as a Source. \n
    # If there is congress_context cite each argument as (Source: Congress). If its empty do not cite Congress as a Source. \n 
    # For all the arguments given, organize them such that there is a core argument followed by supporting arguments. A core argument is centered around a concept or topic. A supporting argument is an argument that supports the core argument. \n
    # return the output as a python dictionary object, where the key is the core argument and the value is a list of supporting arguments. Return only the python dictionary object and nothing else. \n
    # Example output: \n
    # {{"The United States should not be involved in the war in Afghanistan" : ["The United States should not be involved in the war in Afghanistan because it is a waste of money (Source: Reddit)", "The United States should not be involved in the war in Afghanistan because it is a waste of lives (Source: Congress)"]}} \n
    
    # Perform these steps without incorprating any bias into the decision.\n
    # {use_general} \n
    # If using General Knowledge cite the knowledge source from which it came from 'GMOs are not harmful to the environment. (Source: Nature.org)'\n
    # If there is not enough information on the topic provided return the dictionary object "{{"Not enough data":["Please try again"]}}"\n 
   """
    
    return context_query.format(**kwargs)



def retrieve_reddit(topic,diversity):

  res = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=langchain_chroma_reddit.as_retriever(search_type="mmr",search_kwargs={'k':40,'fetch_k':40,'lambda_mult':diversity}))
  arguments = res.run("Find arguments made towards " + topic + " or concepts relating to " + topic + ". An argument is a statement that contains a claim supported by at least one premise. They are also authorative declarative statements.")# Write at the end of the output (Source: Reddit_
  print(arguments)
  return arguments 


def retrieve_congress(topic,diversity):
  res = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=langchain_chroma_congress.as_retriever(search_type="mmr",search_kwargs={'k':40,'fetch_k':40,'lambda_mult':diversity}))
  arguments = res.run("Find arguments made towards " + topic + " or concepts relating to " + topic + ". An argument is a statement that contains a claim supported by at least one premise. They are also authorative declarative statements.")
 
  print(arguments)
  return arguments


def complete(topic, reddit_context, congress_context,use_general):
  if reddit_context == None:
    reddit_context = ""
  if congress_context == None:
    congress_context = ""

  prompt_template = RedditPromptTemplate(input_variables=["reddit_context", "congress_context", "topic","use_general"])#,partial_variables={"format_instructions":format_instructions})

  if use_general:
    response=  llm_comprehension(messages=[SystemMessage(content=prompt_template.format(reddit_context=reddit_context,
                                     congress_context=congress_context,
                                     topic=topic,use_general="Use any source of information to perform the task. Perform the task yourself to the best of your ability."))        ]).content
    
  else:
    response=  llm_comprehension(messages=[SystemMessage(content=prompt_template.format(reddit_context=reddit_context,
                                      congress_context=congress_context,
                                      topic=topic,use_general=""))        ]).content
  
      
  return response




def query(topic,use_reddit,use_congress,diversity,use_general):
  start_time = time.time()
  if use_reddit:
    reddit_context = retrieve_reddit(topic,diversity)
    reddit_time = time.time() - start_time
  else:
    reddit_context = None
    reddit_time = 0
  if use_congress:
    congress_start = time.time()
    congress_context = retrieve_congress(topic,diversity)
    congress_time = time.time() - congress_start
  else:
    congress_context = None
    congress_time = 0

  print("Time to retrieve reddit context: ", reddit_time)
  print("Time to retrieve congress context: ", congress_time)
  args = complete(topic, reddit_context, congress_context,use_general)
  try:
    args = ast.literal_eval(str(args))
  except:
     args = args.replace("'", "\"")
     args = ast.literal_eval(str(args))
     print("errored_but resolved")


  return args
