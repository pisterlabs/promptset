from langchain.llms import OpenAIChat
from dotenv import load_dotenv
from constants import *
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
import os
import nltk
import time
import openai

MAX_RETRIES = 3
INITIAL_WAIT_TIME = 2  # seconds

ROOT_PATH = os.path.abspath('')
ENV = os.path.join(ROOT_PATH, '.env')
load_dotenv(ENV)

api_key=os.getenv('openai_api_key')

llm = OpenAIChat(model_name='gpt-3.5-turbo', openai_api_key=api_key )

def answer(txt_dir_path, uuid, question):
  try:

    loader = DirectoryLoader(txt_dir_path, glob='**/*.txt')
    docs = loader.load()
    char_text_splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap=0)
    doc_text = char_text_splitter.split_documents(docs)
    openai_embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if not os.path.exists(f'{pwd}/file_upload/vectors'):
      os.mkdir(f'{pwd}/file_upload/vectors')

    if not os.path.exists(f'{pwd}/file_upload/vectors/{uuid}'):
      os.mkdir(f'{pwd}/file_upload/vectors/{uuid}')


    vStore = Chroma.from_documents(doc_text, openai_embeddings, persist_directory=f'{pwd}/file_upload/vectors/{uuid}')
    model = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=vStore)
    answer = model.run(question)
    
    return answer
  
  except Exception as ex:
    print({"message":"exception in answer","status":"failed","reason":ex})
    return "failed"


def generate_response(prompt):
    retry_count = 0
    wait_time = INITIAL_WAIT_TIME
    while retry_count < MAX_RETRIES:
        try:
            response = openai.Completion.create(engine="text-davinci-003",prompt=prompt, max_tokens=1024,n=1,stop=None,temperature=0.7)
            return response.choices[0].text.strip()
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            retry_count += 1
            wait_time *= 2  # exponential backoff
    raise Exception("Maximum retries exceeded. Could not generate response.")


def chatpdf(text_path,question=""):
  """
    Returns the response from ChatGPT API based on the user input.
    Parameters:
        text_path (str): text file path.
        question (str): user input text
    Returns:
        answer (list): title and the answer
        questions (list) : list of questions

  """
  try:
    with open(text_path,"r") as txt_file:
      raw_text = txt_file.read()

    text_splitter = CharacterTextSplitter(separator="\n",  chunk_size= 1000, chunk_overlap=0, length_function = len )
    texts = text_splitter.split_text(raw_text)

    openai_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docsearch = FAISS.from_texts(texts, openai_embeddings)
    chain = load_qa_chain(OpenAI(openai_api_key=api_key), chain_type="stuff")
    
    if not question:
      query = 'what were the four questions or prompts that can be asked from existing content for better insights'
      docs = docsearch.similarity_search(query)
      result = chain.run(input_documents=docs, question=query)
      result_list  = result.split('\n')
      result_list = [x for x in result_list if x]
      print(result_list)

      value = result_list[0] 
      final = []
      final_dict = {}
      newval = value + "I need this question as title"
      final_dict.update({ "title": generate_response(newval) } )
      docs = docsearch.similarity_search(value)
      final_dict.update({"answer" :  chain.run(input_documents = docs, question = value)} )
      final.append(final_dict)

    else:
      result_list = []
      docs = docsearch.similarity_search(question)
      result = chain.run(input_documents=docs, question=question)
      final = result

    return { "answer": final , "questions": result_list[1:4] }

  except Exception as ex:
    return {"message":"exception in answer", "status":"failed", "reason":ex}