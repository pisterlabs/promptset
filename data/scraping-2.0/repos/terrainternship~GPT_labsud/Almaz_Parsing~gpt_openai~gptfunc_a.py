import os.path

import tiktoken
import matplotlib.pyplot as plt
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import requests
import zipfile
from langchain.docstore.document import Document
from openai import AsyncOpenAI, OpenAI
import json
import re

class Chat_Manager:
    def __init__(self):
        self.db =None
        self.ix_fragments =None

    def num_tokens_from_string(self, string: str, encoding_name: str= "cl100k_base") -> int:
      """Возвращает количество токенов в строке"""
      encoding = tiktoken.get_encoding(encoding_name)
      num_tokens = len(encoding.encode(string))
      return num_tokens
    
    def hist(self, fragments):
      fragment_token_counts = [self.num_tokens_from_string(fragment.page_content, "cl100k_base") for fragment in fragments]
      plt.hist(fragment_token_counts, bins=20, alpha=0.5, label='Fragments')
      plt.title('Distribution of Fragment Token Counts')
      plt.xlabel('Token Count')
      plt.ylabel('Frequency')
      plt.show()

    def split_text(self, docs):
      from langchain.text_splitter import MarkdownHeaderTextSplitter
      headers_to_split_on = [
          ("#", "H1"),
          ("##", "H2"),
          ("###", "H3"),
      ]
      markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
      fragments =[]
      for ix, text in enumerate(docs):
          items = markdown_splitter.split_text(text)
          for i in items:
             i.metadata["ix"]=ix
          fragments.extend( items)
      return fragments

    def google_load_file(self, url: str, name_file: str):
        def download_file_from_google_drive(file_id: str, name_file: str):
            URL = f"https://drive.google.com/uc?id={file_id}&export=download"
            response = requests.get(URL, stream=True)
            print(URL)
            with open(name_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        def file_id_from_url(url: str) -> str:
            # Extract the document ID from the URL
            match_ = re.search('/file/d/([a-zA-Z0-9-_]+)', url)
            if match_ is None:
                raise ValueError('Invalid Google Docs URL')
            doc_id = match_.group(1)
            return doc_id

        doc_id = file_id_from_url(url)
        download_file_from_google_drive(doc_id, name_file)

    def load_file_content(self, url: str):
        response = requests.get(url) # Получение документа по url.
        response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
        return response.content


    def load_file(self, url: str, name_file: str):
        if os.path.exists( name_file):
            return
        if url.startswith("https://drive.google.com/"):
            self.google_load_file(url, name_file)
            return
        response = requests.get(url) # Получение документа по url.
        response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
        # Сохранение архива.
        with open(name_file, 'wb') as file:
            file.write(response.content)

    def load_db_vect(self, url: str, name_db: str):
        """ Функция загружает векторную Базу знаний."""
        # Скачивание архива Базы знаний
        name_db_zip = name_db + ".zip"
        self.load_file(url,name_db_zip)
        # Разархивирование Базы знаний.
        with zipfile.ZipFile(name_db_zip, 'r') as zip:
            zip.extractall(path=name_db)
        # Загрузка векторной Базы знаний.
        self.db = FAISS.load_local(name_db, OpenAIEmbeddings())
        return self.db

    def load_fragments(self, url:str, file_name:str):
        self.load_file(url,file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            self.ix_fragments = json.load(f)
        pass
    def search_with_score(self, search_str:str, limit_score:float = .8):
        docs = self.db.similarity_search_with_score(search_str, k=7)
        r = []
        score = 0
        def set_score(d, sc):
            d.metadata["score"]=sc
            return d
        for doc in docs:
            s = doc[1]
            if s==0:
                r.append( set_score(doc[0],s))
                break
            if score==0:
                if s<.2:
                    score = s*1.7
                elif s<.3:
                    score = s +.05
                else:
                    score = s +.01
                if score>limit_score:
                    score = limit_score
            if s<score:
                r.append( set_score(doc[0],s))
        return r

    def parse_searched(self, docs):
        res = []
        h1keys =set()
        for doc in docs:
            key = doc.metadata["key"] #-1
            fr = self.ix_fragments[key]
            t = fr["type"]
            # if t=="H2":
            #     pkey = fr["pkey"]
            #     h1keys.add(pkey)
            if t=="H1":
                h1keys.add(key)
        di = {}
        for doc in docs:
            key = doc.metadata["key"] #-1
            fr = self.ix_fragments[key]
            t = fr["type"]
            if t=="H1":
                # Добавляем все дочерние
                if key in di:
                    ar=di[key]
                else:
                    ar=[]
                    di[key]=ar
                content = [fr["H1"] ]
                if "content" in fr:
                    txt = fr["content"]
                    if txt:
                        content.append(txt )
                for fr_i in self.ix_fragments[key+1:]:
                    t2 = fr_i["type"]
                    if t2=="H1":
                        break
                    content.append(fr_i["H2"] )
                    if "content" in fr_i:
                        txt = fr_i["content"]
                        if txt:
                            content.append(txt)
                ar.append( Document(page_content= "\n".join(content),metadata=doc.metadata))
            elif t=="H2":
                key = fr["pkey"]
                if not key in h1keys:
                    content = [fr["H1"], fr["H2"]]
                    if key in di:
                        ar=di[key]
                    else:
                        ar=[]
                        di[key]=ar
                    if "content" in fr:
                        txt = fr["content"]
                        if txt:
                            content.append(txt)
                    ar.append( Document(page_content="\n".join(content),metadata=doc.metadata))
        for key in di:
            ar = di[key]
            for doc in ar:
                res.append(doc)
        return res

    def insert_newlines(self, text: str, max_len: int = 120) -> str:
        """ Функция форматирует переданный текст по длине
        для лучшего восприятия на экране."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) > max_len:
                lines.append(current_line)
                current_line = ""
            current_line += " " + word
        lines.append(current_line)
        return "\n".join(lines)

    def answer_index(self, model, system, topic: str, query, search_index, temp = 0, verbose_documents = 0,  verbose_price = 0, top_documents = 3, limit_score = 0.0):
        """ Основная функция которая формирует запрос и получает ответ от OpenAI по заданному вопросу
        на основе векторной Базы знаний. """

        # Выбор варианта вопроса. Если есть query, то вопрос задан из группового запроса и он имеет приоритет.
        question = query["question"] if bool(query) else topic
        # Выборка релевантных чанков.
        s_docs = self.search_with_score(question)
        docs = self.parse_searched(s_docs)

        message_content = ""            # Контекст для GPT.
        message_content_display = ""    # Контекст для вывода на экран.
        for i, doc in enumerate(docs):
            # Формирование контекста для запроса GPT и показа на экран отобранных чанков.
            message_content = message_content + f'Отрывок документа №{i+1}:{doc.page_content}'
            message_content_display = message_content_display + \
                f"\n Отрывок документа №{i+1}. Chank № {doc.metadata.get('key')}. "+\
                f"Score({str(doc.metadata.get('score'))})\n -----------------------\n{self.insert_newlines(doc.page_content)}\n"

            # Сбор информации для группого запроса.
            if bool(query):
                # Выделение из строки метаданных ссылки. Если нет - присваиваем пустую строку.
                link = ''
                # Заполнение запроса выбранными чанками.
                query[f"chank_{i+1}"] = f"""Chank № {doc.metadata.get('chank')}. Score({str(doc.metadata.get('score'))}).
{doc.page_content}.\n--------\n{link}"""

        # Вывод на экран отобранных чанков.
        if verbose_documents:
            print(message_content_display)

        # Отправка запроса к Open AI.
        completion = OpenAI().chat.completions.create(
            model = model[0],
            messages = [
                {"role": "system", "content": system } ,
                {"role": "user", "content": f"Документ с информацией для ответа пользователю : {message_content}.\n\nВопрос клиента: {question}"}
            ],
            temperature=temp
        )

        # Подсчет токенов и стоимости.
        prompt_tokens = completion.usage.prompt_tokens
        total_tokens = completion.usage.total_tokens
        price_promt_tokens = prompt_tokens * model[1]/1000
        price_answer_tokens = (total_tokens - prompt_tokens) * model[2]/1000
        price_total_token = price_promt_tokens + price_answer_tokens
        # Сбор информации для группого запроса.
        if bool(query):
            query["price_query"] = price_total_token
            query["price_question"] = price_promt_tokens
            query["price_answer"] = price_answer_tokens
            query["token_query"] = total_tokens
            query["token_question"] = prompt_tokens
            query["token_answer"] = total_tokens - prompt_tokens
        # Вывод на экран стоимости запроса.
        if (verbose_price):
            print('\n======================================================= ')
            print(f'{prompt_tokens} токенов использовано на вопрос. Цена: {round(price_promt_tokens, 6)} $.')
            print(f'{total_tokens - prompt_tokens} токенов использовано на ответ.  Цена: {round(price_answer_tokens, 6)} $.')
            print(f'{total_tokens} токенов использовано всего.     Цена: {round(price_total_token, 6)} $')
            print('======================================================= ')

        # Ответ OpenAI.
        return completion.choices[0].message.content


