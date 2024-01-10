import os

import telebot

BOT_TOKEN = os.environ.get('BOT_TOKEN')

print(BOT_TOKEN)
bot = telebot.TeleBot(BOT_TOKEN)

import os

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
import os

from langchain.document_loaders import PyMuPDFLoader, AmazonTextractPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# python -m pip install chromadb=0.3.29

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import datetime as dt
import pytz




load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']

# Data Loader
pdf_path = "./pdfs/NIPS-2017-attention-is-all-you-need-Paper.pdf"

# documents = []
# for iter_pdf in list_pdfs:
#     loader = PyMuPDFLoader(f'/home/pi/Documents/academIA/docs/{iter_pdf}')
#     documents.extend(loader.load())
#
# # Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
# texts = text_splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="/home/pi/Documents/academIA/data",
            embedding_function=OpenAIEmbeddings())
print(db._collection.count())
# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8})
retriever = db.as_retriever(search_type="mmr")

memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

# Prompts
prompt_template = """Usa los siguientes elementos de contexto para responder la pregunta al final. 
Si la respuesta no se encuentra en el contexto, responde amablemente y con un mensaje upbeat diciendo que tu contexto
se restringe exclusívamente al contenido de Electricidad y Magnetismo.
Por supuesto, si te saluda o realiza preguntas amigables puedes responderle sin acudir al contexto. 

{context}

Recuerda siempre las posibles preguntas anteriores {chat_history}

Pregunta: {question}
"""

# Prompts
prompt_template_no_hist = """Usa los siguientes elementos de contexto para responder la pregunta al final. 
Si la respuesta no se encuentra en el contexto, responde amablemente y con un mensaje upbeat diciendo que tu contexto
se restringe exclusívamente al contenido de Electricidad y Magnetismo.
Por supuesto, si te saluda o realiza preguntas amigables puedes responderle sin acudir al contexto. 

{context}

Pregunta: {question}
"""


#prompt = PromptTemplate(
#    template=prompt_template,
#    input_variables=["context", "question"]
#)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt = PromptTemplate(input_variables=['chat_history', "question", "context"], template=prompt_template)
# prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template_no_hist)


chain_type_kwargs = {"prompt": prompt}

# Generate
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
#                                 retriever=retriever,
#                                 memory=memory,
#                                 chain_type_kwargs=chain_type_kwargs
#                                 )

qa = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                           retriever=retriever,
                                           memory=memory,
                                           combine_docs_chain_kwargs={"prompt": prompt}
                                           )

# qa = load_qa_chain(llm=llm, chain_type="stuff",
#                                           # retriever=retriever,
#                                           memory=memory,
#                                           prompt=prompt
#                                           )


print('Preguntas!')


# MongoDB
mongo_user = os.environ.get('MONGO_USER')
mongo_pass = os.environ.get('MONGO_USER')
mongo_host = f'mongodb+srv://{mongo_user}:{mongo_pass}@atheneia.dqt9y1t.mongodb.net/?retryWrites=true&w=majority'
mongo_port = 27017

from pymongo import MongoClient
mongo_client = MongoClient(mongo_host, mongo_port)
db = mongo_client.Atheneia.Conversations

# Obten un objeto de zona horaria para Madrid/Amsterdam
zona_horaria = pytz.timezone('Europe/Madrid')


chat_history = []

@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    user_id = message.from_user.id
    ts_mensaje = dt.datetime.now(tz=zona_horaria)


# Prompt You are a friendly chatbot assistant that responds in a conversational
    # manner to users questions. Keep the answers short, unless specifically
    # asked by the user to elaborate on something.
    print(message.text)
    llm_response = qa({'question': message.text, 'chat_history':chat_history}) # question -> ConversationalRetrievalChain, query
    ts_respuesta = dt.datetime.now(tz=zona_horaria)
    ts_diferencia = ts_respuesta - ts_mensaje
    ts_diferencia = ts_diferencia.seconds
    answer = llm_response['answer'] # answer -> ConversationalRetrievalChain, result
    chat_history.append((message.text, answer))


    # bot.reply_to(message, answer)
    bot.send_message(chat_id=user_id, text=answer)

    info_conversation = {'user_id': user_id,
                         'question': message.text,
                         'answer': answer,
                         'ts_respuesta': ts_respuesta,
                         'ts_mensaje': ts_mensaje,
                         'tiempo_respuesta': ts_diferencia,
                         'timestamp': dt.datetime.now(tz=zona_horaria)
                         }
    db.insert_one(info_conversation)
    # bot.reply_to(message, user_id)

bot.infinity_polling()