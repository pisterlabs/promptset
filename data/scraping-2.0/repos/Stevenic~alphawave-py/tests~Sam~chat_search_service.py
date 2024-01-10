import json
import sys
import os
import time
import re
import traceback
import readline as rl
from fastapi import FastAPI
from alphawave.MemoryFork import MemoryFork
from alphawave.OSClient import OSClient
from alphawave.OpenAIClient import OpenAIClient
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
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
from alphawave.alphawaveTypes import PromptCompletionOptions
import alphawave_pyexts.llmsearch.google_search_concurrent as gs
import alphawave_pyexts.utilityV2 as ut
import warnings
from itertools import zip_longest
import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import wordfreq as wf
from bs4 import BeautifulSoup
#from unstructured.partition.html import partition_html
history = {}


#client = OSClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'))#, logRequests=True)
model = 'gpt-3.5-turbo'
memory = VolatileMemory()
functions = None
tokenizer = GPT3Tokenizer()

app = FastAPI()


@app.get("/search/")
async def search(query: str, max_chars: int = 1200):
  global client, memory, functions, tokenizer
  response_text = ''
  storeInteraction = True
  try:
    query_phrase, keywords = ut.get_search_phrase_and_keywords(client, query, model, memory, functions, tokenizer)
    google_result= \
      gs.search_google(query, gs.QUICK_SEARCH, query_phrase, keywords, client, model, memory, functions, tokenizer, max_chars)

    print(f'google_result:\n{google_result}')
    if type(google_result) is list:
      text = ''
      for item in google_result:
        text += item['text'].strip()+'\n'
      
      print(f'text from google_search\n{text}\n')
      prompt = Prompt([
        SystemMessage(f'summarize the following text, removing duplications, with respect to {query}'),
        UserMessage('Text:\n{{$input}}'),
        AssistantMessage(' ')
        ])
      prompt_options = PromptCompletionOptions(completion_type='chat', model=model, max_tokens=int(max_chars/4))
      summary = ut.run_wave (client, {"input": text}, prompt, prompt_options,
                               memory, functions, tokenizer, max_repair_attempts=1,
                               validator=DefaultResponseValidator())
      if type(summary) is dict and 'status' in summary.keys() and summary['status'] == 'success':
        return {"result":summary['message']['content']}
      else: return {'result': 'failure'}
  except Exception as e:
    traceback.print_exc()
    return {"result":str(e)}

@app.get("/retrieve/")
async def retrieve(title: str, url: str, max_chars: int = 1000):
  global client, memory, functions, tokenizer
  response_text = ''
  try:
    query_phrase, keywords = ut.get_search_phrase_and_keywords(client, title, model, memory, functions, tokenizer)
    keyword_weights = gs.compute_keyword_weights(keywords)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      options = Options()
      options.page_load_strategy = 'eager'
      options.add_argument("--headless")
      result = ''
      with webdriver.Chrome(options=options) as dr:
        print(f'*****setting page load timeout {5} {url}')
        dr.set_page_load_timeout(5)
        try:
          dr.get(url)
          response = dr.page_source
          soup = BeautifulSoup(response, "html.parser")
          all_text = soup.get_text()
          # Remove script and style tags
          all_text = all_text.split('<script>')[0]
          all_text = all_text.split('<style>')[0]
          
          # Remove script and style content
          all_text = all_text.split('</script>')[0]
          all_text = all_text.split('</style>')[0]

          final_text = re.sub(r'(\n){2,}', '\n', all_text)
        except selenium.common.exceptions.TimeoutException as e:
          print(str(e))
          return {"result":'timeout'}
        except Exception as e:
          traceback.print_exc()
          return {"result":str(e)}
        
    return {"result":final_text}
  except Exception as e:
    traceback.print_exc()
    return {"result":str(e)}
  
