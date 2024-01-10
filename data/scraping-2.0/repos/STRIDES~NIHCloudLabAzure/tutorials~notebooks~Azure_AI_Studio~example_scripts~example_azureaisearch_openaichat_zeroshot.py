from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.retrievers import AzureCognitiveSearchRetriever
import sys
import json
import os


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

MAX_HISTORY_LENGTH = 1

def build_chain():
    
    os.getenv("AZURE_OPENAI_API_KEY")
    os.getenv("AZURE_OPENAI_ENDPOINT")
    os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
    os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
    os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    
    llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    #max_tokens = 3000
)

    retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=2)
    
  
    prompt_template = """
      Instructions:
      I will provide you question and scientific documents you will answer my question with information from documents in English, and you will create a cumulative summary that should be concise and should accurately. 
      You should not include any personal opinions or interpretations in your summary, but rather focus on objectively presenting the information from the papers. 
      Your summary should be written in your own words and ensure that your summary is clear, and concise.

      {question} Answer "don't know" if not present in the documents. 
      {context}
      Solution:"""

    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"],
    )

    condense_qa_template = """
    Chat History:
    {chat_history}
    Here is a new question for you: {question}
    Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT}
        )
    return qa

def run_chain(chain, prompt: str, history=[]):
    print(prompt)
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
        dict_meta=json.loads(d.metadata['metadata'])
        print(dict_meta['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
