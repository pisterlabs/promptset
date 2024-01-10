# @title Установка пакетов
from IPython.display import clear_output

# pip install --upgrade tiktoken
# pip install langchain openai
# pip install faiss-cpu

import getpass
import openai
import os
def get_key_ОpenAI():
  openai.api_key = getpass.getpass(prompt='Введите секретный ключ для сервиса chatGPT: ')
  os.environ["OPENAI_API_KEY"] = openai.api_key

get_key_ОpenAI()

#@title Импорт библиотек и Сервисные функции
from IPython.display import clear_output
from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import tiktoken
import re


# ----------------------------------
MODEL_TURBO_16K = "gpt-3.5-turbo-16k"
MODEL_TURBO_0613 = "gpt-3.5-turbo-0613"
MODEL_GPT4 = "gpt-4-0613"
MODEL_GPT4_1106 = "gpt-4-1106-preview"
# ----------------------------------

# размер чанка, оптимально для Опыта работы - 1024, Обо мне - 2048
chunk_size_we = 1024
chunk_size_am = 2048

class WorkerОpenAI():
  def __init__(self, \
               system_promt = " ", \
               system_promt_assistant = " ", \
               mod = MODEL_TURBO_16K, \
               content_topics = None, \
               save_project = '/content/'):
    self.model = mod
    self.save_project  = save_project

    if content_topics:
      self.content_topics = self.load_txt_file(content_topics)

    # системные настройки
    self.system_promt = self.load_document_text(system_promt)
    self.speaker_system_promt = self.load_document_text(system_promt_assistant)


  def load_document_text(self, url: str) -> str:
      # функция для загрузки документа по ссылке из гугл док
      match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
      if match_ is None:
          raise ValueError('Invalid Google Docs URL')
      doc_id = match_.group(1)
      response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
      response.raise_for_status()
      text = response.text
      return text


  def load_txt_file(self, file_path):
      with open(file_path, 'r') as file_:
          text = file_.read()
      return text

  # пример подсчета токенов
  def num_tokens_from_messages(self, messages):
      """Returns the number of tokens used by a list of messages."""
      try:
          encoding = tiktoken.encoding_for_model(self.model)
      except KeyError:
          encoding = tiktoken.get_encoding("cl100k_base")
      # if self.model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      if self.model in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-4-0613"]:  # note: future models may deviate from this
          num_tokens = 0
          for message in messages:
              num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
              for key, value in message.items():
                  num_tokens += len(encoding.encode(value))
                  if key == "name":  # if there's a name, the role is omitted
                      num_tokens += -1  # role is always required and always 1 token
          num_tokens += 2  # every reply is primed with <im_start>assistant
          return num_tokens
      else:
          raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {self.model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


# Для пункта "Опыт работы"
  def create_embedding_one_file_we(self, doc_txt_dir="/content/", \
                                   file_ = "Имя файла.txt" , \
                                   faiss_db_dir ="/content/"):

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
      """Returns the number of tokens in a text string."""
      encoding = tiktoken.get_encoding(encoding_name)
      num_tokens = len(encoding.encode(string))
      return num_tokens

    # Для Копирайтера
    self.splitter = RecursiveCharacterTextSplitter(['\n\n', '\n', ' '], chunk_size=chunk_size_we, chunk_overlap=0)
    chunkID = 0
    count_tokens = 0
    self.file_name = file_[:-3]
    print("Загружается файл: ", file_)
    # проходимся по всем данным
    source_chunks = []
    # разбиваем на несколько частей с помощью метода split_text
    with open(os.path.join(doc_txt_dir, file_), "r") as f:
      for chunk in self.splitter.split_text(f.read()):
          chunkID += 1
          source_chunks.append(Document(page_content=chunk, \
                              metadata={'source': file_,
                                        'chunkID': chunkID}))

    # Создание индексов документа и СОХРАНЕНИЕ
    # Если документ не пуст, то создать и сохранить базу индексов эмбеддингов отрезков документа
    if len(source_chunks) > 0:
        self.db = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
        count_token = num_tokens_from_string(' '.join([x.page_content for x in source_chunks]), "cl100k_base")
        count_tokens += count_token
        print('Количество токенов в документе :', count_token)
        # print('ЦЕНА запроса:', 0.0004 * (count_token / 1000), ' $')

        self.db.save_local(os.path.join(faiss_db_dir, f"db_initial_we_{self.file_name[:35]}"))

    print('\nЦЕНА запроса создания базы индексов для пункта "Опыт работы":', 0.0004 * (count_tokens / 1000), ' $')

# Для пункта "Обо мне"
  def create_embedding_one_file_am(self, doc_txt_dir="/content/", \
                                   file_ = "Имя файла.txt" , \
                                   faiss_db_dir ="/content/"):

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
      """Returns the number of tokens in a text string."""
      encoding = tiktoken.get_encoding(encoding_name)
      num_tokens = len(encoding.encode(string))
      return num_tokens

    # Для Копирайтера
    self.splitter = RecursiveCharacterTextSplitter(['\n\n', '\n', ' '], chunk_size=chunk_size_am, chunk_overlap=0)
    chunkID = 0
    count_tokens = 0
    self.file_name = file_[:-3]
    print("Загружается файл: ", file_)
    # проходимся по всем данным
    source_chunks = []
    # разбиваем на несколько частей с помощью метода split_text
    with open(os.path.join(doc_txt_dir, file_), "r") as f:
      for chunk in self.splitter.split_text(f.read()):
          chunkID += 1
          source_chunks.append(Document(page_content=chunk, \
                              metadata={'source': file_,
                                        'chunkID': chunkID}))

    # Создание индексов документа и СОХРАНЕНИЕ
    # Если документ не пуст, то создать и сохранить базу индексов эмбеддингов отрезков документа
    if len(source_chunks) > 0:
        self.db = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
        count_token = num_tokens_from_string(' '.join([x.page_content for x in source_chunks]), "cl100k_base")
        count_tokens += count_token
        print('Количество токенов в документе :', count_token)
        # print('ЦЕНА запроса:', 0.0004 * (count_token / 1000), ' $')

        self.db.save_local(os.path.join(faiss_db_dir, f"db_initial_am_{self.file_name[:35]}"))

    print('\nЦЕНА запроса создания базы индексов для пункта "Обо мне":', 0.0004 * (count_tokens / 1000), ' $')


# загрузим файл со всеми инструкциями от HR отдела
import gdown

# Задаем прямую ссылку на файл (ссылку, которая начинается с "https://drive.google.com/uc?...")
# файл 'merged_resume.txt' доступен по ссылке:  https://drive.google.com/file/d/105lVzXUxliTGU_xgOAL4cz_tqVkNfnOb/view?usp=sharing
file_url = 'https://drive.google.com/uc?id=105lVzXUxliTGU_xgOAL4cz_tqVkNfnOb'

# Путь для сохранения файла в корневой папке
output_path = '/content/'  # Можете изменить путь, если нужно сохранить файл в другой папке

# Загрузите файл
gdown.download(file_url, output_path)

# @title Создаем общие чанки из всех инструкций от HR отдела
projects_dir = '/content/'

Promt_copywriter = "https://docs.google.com/document/d/1598tY9blVpyjlZ6YjdT3b2F9zDzNQvgZi7vWGYRRFPY/edit?usp=sharing"
Promt_assistant = "https://docs.google.com/document/d/1tf6HgJvQ05SVSSV59zEKtk-3gkY9pkfAsHizWctwJbU/edit?usp=sharing"


# Создаем объект для дообучения chatGPT
curator_we = WorkerОpenAI(system_promt = Promt_copywriter, # системный промт
                          system_promt_assistant = Promt_assistant, # промт помощника
                          save_project = projects_dir)     # путь для сохранения готовых файлов

curator_am = WorkerОpenAI(system_promt = Promt_copywriter, # системный промт
                          system_promt_assistant = Promt_assistant, # промт помощника
                          save_project = projects_dir)     # путь для сохранения готовых файлов

# Формируем базу по файлу txt
# # путь к материалам
doc_txt_dir = projects_dir
file_name = 'merged_resume.txt' # в файле merged_resume.txt последовательно скопированы все инструкции от HR


curator_we.create_embedding_one_file_we(doc_txt_dir = doc_txt_dir,   # путь к материалам
                                        file_ = file_name,            # какой файл берем
                                        faiss_db_dir = projects_dir)     # путь для сохранения исходной базы
curator_am.create_embedding_one_file_am(doc_txt_dir = doc_txt_dir,   # путь к материалам
                                        file_ = file_name,            # какой файл берем
                                        faiss_db_dir = projects_dir)     # путь для сохранения исходной базы
