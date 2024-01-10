from langchain.retrievers import AmazonKendraRetriever

from langchain.chains import ConversationalRetrievalChain
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.prompts import PromptTemplate
import sys
import json
import os
from langchain.llms.bedrock import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains import RetrievalQA


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5

def build_chain():
  region = os.environ["AWS_REGION"]
  kendra_index_id = os.environ["KENDRA_INDEX_ID_BR"]
  AWS_PROFILE = os.environ["AWS_PROFILE"]
  
  class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        
        input_str = json.dumps({"prompt": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        
        response_json = json.loads(output.get("body").read().decode())

        
        return response_json['completions']



  

  model_kwargs_anthropic = {"max_tokens_to_sample":2048,
                               "temperature":1e-10,"top_k":250,
                               "top_p":1}  

  llm_BR = Bedrock(credentials_profile_name=AWS_PROFILE, model_id="anthropic.claude-v1",model_kwargs=model_kwargs_anthropic)
      
  
  retriever = AmazonKendraRetriever(index_id=kendra_index_id)


  prompt_template = """
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and writes lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know. now below is the context:
  {context}
  ###Instruction:###
   Based on the above documents (context), provide a detailed and well written answer for, 
   #####
   {question} 
   ####Answer "don't know" if not present in the above document. Answer:##
   
  """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  condense_qa_template = """
  Given the following conversation and a follow up question, rephrase the follow up question 
  to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""
  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

  
  qa = ConversationalRetrievalChain.from_llm(llm=llm_BR, retriever=retriever, 
                                            condense_question_prompt=standalone_question_prompt, 
                                             return_source_documents=True,
                                             combine_docs_chain_kwargs={"prompt":PROMPT})

  return qa

def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt, "chat_history": history})


if __name__ == "__main__":
  chat_history = []
  qa = build_chain()
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
  print(">", end=" ", flush=True)
  for query in sys.stdin:
    if (query.strip().lower().startswith("new search:")):
      query = query.strip().lower().replace("new search:","")
      chat_history = []
    elif (len(chat_history) == MAX_HISTORY_LENGTH):
      chat_history.pop(0)
    result = run_chain(qa, query, chat_history)
    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
