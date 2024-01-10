
import pinecone
import openai
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import fitz
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")
model = OpenAIEmbeddings(deployment="text-embedding-ada-002", openai_api_key=openai.api_key)
pinecone.init(api_key=os.getenv("PINE_API_KEY"), environment="gcp-starter")
index = pinecone.Index("chat-bot")


def find_match(input):
    input_em= model.embed_query(input)
    #input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=5, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text'] + "\n" + result['matches'][2]['metadata']['text']


'''def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.And if there is no conversation log please don't change the question and leave it as it is.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']'''
from openai import OpenAI
def query_refiner(conversation, query):
 client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

 response= client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.And if there is no conversation log please don't change the question and leave it as it is.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}],

)
 return response.choices[0].message.content


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

def pdf_text_convertor(file_path):
   arr=[]
   doc = fitz.open(file_path)
   text = ""
   for page in doc:
     text+=page.get_text()
   arr.append(text)
   return text
def pinecone_uploader(file_path):
   try:
    with open(file_path, 'r') as file:
        contents = file.read()
   except FileNotFoundError:
    print(f"File not found: {file_path}")
   text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1500,
    chunk_overlap  = 200,
    length_function = len,
)
   texts = text_splitter.split_text(contents)
   new_texts=[]
   for i in range(len(texts)):
      text= text_splitter.create_documents([texts[i]])
      new_texts.extend(text)
   return new_texts
