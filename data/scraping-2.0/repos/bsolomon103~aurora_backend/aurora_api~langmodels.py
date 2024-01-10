#from .tools import tools
import time
from langchain.chat_models import ChatOpenAI
from .prompts import aurora_prompt
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain 
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, VertexAI 
from langchain.chat_models import ChatOpenAI
import pickle
from langchain.schema import retriever
import os
import json
from django.conf import settings
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import time
import re
from .celeryfuncs import cache_chat_message
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
import wikipedia
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain.tools.render import format_tool_to_openai_function
import requests
from pydantic import BaseModel, Field
import datetime
from langchain.memory.chat_message_histories import RedisChatMessageHistory
import redis 
from langchain.schema.runnable.history import RunnableWithMessageHistory


class OpenMeteoInput(BaseModel):
  latitude: float = Field(..., description="Latitude of the location to get weather for")
  longitude: float = Field(..., description="Longitude of the location to get weather for")
  

@tool
def get_current_temparature(latitude:float, longitude:float) -> str:
  """Fetch current weather for given coordinates"""
  BASE_URL = "https://api.open-meteo.com/v1/forecast"

  params = {
      'latitude': latitude,
      'longitude': longitude,
      'hourly': 'temperature_2m',
      'forecast_days': 1
  }

  response = requests.get(BASE_URL, params=params)

  if response.status_code == 200:
    results = response.json()
  else:
    raise Exception(f"Api Request failed with status code {response.status_code}")

  current_utc_time = datetime.datetime.utcnow()
  time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
  temperature_list = results['hourly']['temperature_2m']

  closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
  current_temperature = temperature_list[closest_time_index]
  return f'The current temperature is {current_temperature}°C'


@tool
def search_wikipedia(query: str) -> str:
  """Run wikipedia search and get page summaries"""
  page_titles = wikipedia.search(query)
  summaries = []
  for page_title in page_titles[:3]:
    try:
      wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
      summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
    except(
        self.wiki_client.exceptions.PageError,
        self.wiki_client.exception.DisambiguationError,
    ):
      pass
  if not summaries:
    return "No viable results"
  return "\n\n".join(summaries)



 
functions = [format_tool_to_openai_function(tool) for tool in [search_wikipedia, get_current_temparature]]
tools = [search_wikipedia, get_current_temparature]




def chain(session, msg):
    print('here')
    model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=200).bind(functions=functions)
    prompt = aurora_prompt()
    output = OpenAIFunctionsAgentOutputParser()
    chat_history = session['messages']
    start = time.time()
    store = FAISS.load_local(os.path.join(settings.BASE_DIR, 'media', 'training_file', os.path.basename(session['customer_name'])), OpenAIEmbeddings())
    retriever = store.as_retriever()
    end = time.time()
    print(f'Time taken to build chain: {start - end}')
    mapp = RunnablePassthrough.assign(
            context = lambda x : retriever.get_relevant_documents(x['input']),
            input =  lambda x : x['input'],
            agent_scratchpad = lambda x : format_to_openai_functions(x["intermediate_steps"])
    )
 
    chain = mapp | prompt | model | output
    agent_executor = AgentExecutor.from_agent_and_tools(agent=chain, tools=tools, verbose=False)
    response = agent_executor.invoke({"input": session["messages"]})

    current_messages = session['messages']
    current_messages = current_messages + f"assistant: {response['output']}\n"
    session['messages'] = current_messages
    session.save()
    
    if response['output'].__contains__('assistant:'):
        response = response['output'].split(':')[1]
    else:
        response = response['output']
  
    x = postprocessor(response)
    if x != None:
        response = response.replace(f"[{x}]", '😁 ')    
    cache_chat_message(session['session_key'], msg, response)
    return response


def postprocessor(text):
    # Find everything within square brackets
    matches = re.findall(r'\[([^]]+)\]', text)
    return matches[0] if len(matches) > 0 else None





    
    
