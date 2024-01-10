from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter

import requests
import os
import re
import getpass
import openai
import tiktoken
from openai import OpenAI
import zipfile
from IPython import display
import timeit

import gspread                  # Импортируем API для работы с Google таблицами
# from google.colab import auth   # Импортируем модуль для аутентификации
# from google.auth import default # Импортируем модуль для работы с учетными данными


#@title Вспомогательные функции

def num_tokens_from_string(string: str) -> int:
    """Возвращает количество токенов в строке"""
    # Выбор кодировщика. `cl100k_base`используется для `gpt-4`, `gpt-3.5-turbo`, `text-embedding-ada-002`
    encoding = tiktoken.get_encoding("cl100k_base")
    # Разбивка строки на токены и подсчет из количества.
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_text(text, verbose=0):
    """ Функция разбивает текст на чанки. """
    # Шаблон MarkdownHeaderTextSplitter по которому будет делится переданный
    # текст в формате Markdown.
    headers_to_split_on = [ ("#",    "Header 1"),
                            ("##",   "Header 2"),
                            ("###",  "Header 3"),
                            ("####", "Header 4")
                        ]
    # Создаем экземпляр спилиттера.
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # Получаем список чанков.
    source_chunks = markdown_splitter.split_text(text)

    # Обработка чанков.
    chank_count = len(source_chunks)
    for number, chank in enumerate(source_chunks):
        # Добавление информации в метаданные чанка о его номере в базе.
        chank.metadata["chank"]=f'{number+1}/{chank_count}'
        # Вывод количества слов/токенов в фрагменте, если включен режим verbose.
        if verbose:
            count = num_tokens_from_string(chank.page_content)
            print(f"\n Chank#{number+1}/{chank_count}. Tokens in text = {count}\n {'-' * 20}\n{insert_newlines(str(chank))}\n{'=' * 20}")

    # Возвращение списка фрагментов текста.
    return source_chunks

def create_embedding(data, verbose=0):
    """Функция преобразует текстовую Базу знаний в векторную."""
    # Разбивка текста на чанки.
    source_chunks = []
    source_chunks = split_text(text=data, verbose=verbose)

    # Создание векторной Базы знаний на основе чанков.
    search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings(), )
    # Подсчет общего количества токенов во всех чанках.
    count_token = num_tokens_from_string(' '.join([x.page_content for x in source_chunks]))
    # Печать сводной информации по созданию векторной Базы знаний.
    print('\n==================== ')
    print('Количество токенов в документе :', count_token)
    # Стоимость эмбэндинга согласно прайса на 22.11.2023 - 0,0001/1К токенов.
    # https://openai.com/pricing#language-models
    print('ЦЕНА запроса:', 0.0001*(count_token/1000), ' $')
    print('==================== ')
    return search_index

def load_file(url: str):
    """ Функция загрузки документа по url как текст."""
    try:
        response = requests.get(url) # Получение документа по url.
        response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
        return response.text
    except Exception as e:
        print(e)

def load_search_indexes(url: str, verbose=0):
    """Функция загружает текстовую Базу знаний и преобразует ее в векторную."""
    try:
        return create_embedding(load_file(url), verbose=verbose)
    except Exception as e:
        print(e)

def insert_newlines(text: str, max_len: int = 120) -> str:
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

def filtred_docs (docs, limit_score):
    """ Функция удаляет из отобранных чанков чанки у которых score выше значения limit_score.
    При этом limit_score определяет гарантированно ошибочные чанки.
    Далее отбор чанков идет в зависимости от значения score первого не нулевого
    score. Если есть чанк с score=0 он оставляется единственным.
    Если limit_score = 0, то чанки не фильтруются."""
    if bool(limit_score):
        r = []
        score = 0
        def set_score(d, sc):
            d[0].metadata["score"]=sc
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
                    r.append(set_score(doc,s))
        print(f'Фильр пропустил чанк(ов): {len(r)} из {len(docs)}')
    else:
        r = docs
    return r

def answer_index(model, system, topic: str, query, search_index, temp = 0, verbose_documents = 0,  verbose_price = 0, top_documents = 3, limit_score = 0.0):
    """ Основная функция которая формирует запрос и получает ответ от OpenAI по заданному вопросу
    на основе векторной Базы знаний. """

    # Выбор варианта вопроса. Если есть query, то вопрос задан из группового запроса и он имеет приоритет.
    question = query["question"] if bool(query) else topic
    # Выборка релевантных чанков.
    docs = filtred_docs(search_index.similarity_search_with_score(question, k=top_documents), limit_score)

    """ Этот блок закоментирован. Рассматривается возможность даполнительного
    запроса с фильтрацией чанков по метаданным. Пока нереализовано."""
    # for i, doc in enumerate(docs):
    #     header1 = doc[0].metadata.get('Header 1')
    #     print(f'{i}. {header1}. Score: {doc[1]}')
    #     if header1=='Оценка' or header1=='Экспертиза':
    #         #docs = search_index.similarity_search_with_score(question, k=top_documents, )
    #         #print(f'{i}. {header1}. Score: {doc[1]}')
    #         pass


    message_content = ""            # Контекст для GPT.
    message_content_display = ""    # Контекст для вывода на экран.
    for i, doc in enumerate(docs):
        # Формирование контекста для запроса GPT и показа на экран отобранных чанков.
        message_content = message_content + f'Отрывок документа №{i+1}:{doc[0].page_content}'
        message_content_display = message_content_display + f"\n Отрывок документа №{i+1}. Chank № {doc[0].metadata.get('chank')}. Score({str(round(doc[1], 3))})\n -----------------------\n{insert_newlines(doc[0].page_content)}\n"

        # Сбор информации для группого запроса.
        if bool(query):
            # Выделение из строки метаданных ссылки. Если нет - присваиваем пустую строку.
            search_link_h1 = re.search(r'\[(.*?)\]', doc[0].metadata.get('Header 1')) if bool(doc[0].metadata.get('Header 1')) else ""
            search_link_h2 = re.search(r'\[(.*?)\]', doc[0].metadata.get('Header 2')) if bool(doc[0].metadata.get('Header 2')) else ""
            search_link_h3 = re.search(r'\[(.*?)\]', doc[0].metadata.get('Header 3')) if bool(doc[0].metadata.get('Header 3')) else ""
            search_link_h4 = re.search(r'\[(.*?)\]', doc[0].metadata.get('Header 4')) if bool(doc[0].metadata.get('Header 4')) else ""
            # Выбор самой внутренней ссылки. Если несколько ссылок, то самая внутренняя ссылается на расположение чанка на сайте.
            link = ''
            if bool(search_link_h4):
                link = search_link_h4.group(1)
            elif bool(search_link_h3):
                link = search_link_h3.group(1)
            elif bool(search_link_h2):
                link = search_link_h2.group(1)
            elif bool(search_link_h1):
                link = search_link_h1.group(1)
            # Заполнение запроса выбранными чанками.
            query[f"chank_{i+1}"] = f"Chank № {doc[0].metadata.get('chank')}. Score({str(round(doc[1], 3))}).\n{doc[0].page_content}.\n--------\n{link}"

    # Вывод на экран отобранных чанков.
    if (verbose_documents):
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

#question_normalization
def question_normalization(text):
    """ Функция нормализует текст вопроса. Удаляет лишние пробелы,
    символы, знаки. Делает первое слово с заглавной буквы."""

    # Удаление символов "!?.,-" из текста
    #text = re.sub(r'[-!?_]', '', text) # '[-!?.,_]'
    # Разделение текста на слова
    words = text.split()
    # # Проход по каждому слову
    # for i in range(len(words)):
    #     # Если слово состоит из заглавных букв и длина больше 1 символа, считаем это аббревиатурой
    #     if words[i].isupper() and len(words[i]) > 1:
    #         continue  # Пропускаем аббревиатуры
    #     else:
    #         words[i] = words[i].lower()  # Преобразуем в нижний регистр
    # # Преобразуем первое слово в строке к верхнему регистру
    #words[0] = words[0].capitalize()
    # Объединяем слова обратно в строку
    return ' '.join(words)

def load_bd_text(url: str, verbose=0):
    """ Функция загружает текстовую Базу знаний и
        преобразует ее в векторную."""
    response = requests.get(url) # Получение документа по url.
    response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
    return create_embedding(response.text, verbose=verbose)

def load_bd_vect(name, url: str, verbose=0):
    """ Функция загружает векторную Базу знаний."""
    # bd_index
    name_bd = name+'.zip'
    # Скачивание архива Базы знаний
    response = requests.get(url) # Получение документа по url.
    response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
    # Сохранение архива.
    with open(name_bd, 'wb') as file:
        file.write(response.content)
    # Разархивирование Базы знаний.
    with zipfile.ZipFile(name_bd, 'r') as zip:
        zip.extractall()
    # Загрузка векторной Базы знаний.
    bd = FAISS.load_local(name, OpenAIEmbeddings())

    if verbose :
        docs = bd.similarity_search_with_score('', k=10000)
        docs_sorted = sorted(docs, key=lambda x: int(x[0].metadata.get('chank').split('/')[0]))
        for doc in docs_sorted:
            count = num_tokens_from_string(doc[0].page_content)
            print(f"\n Chank#{doc[0].metadata.get('chank')}. Tokens in text = {count}\n {'-' * 20} \n{insert_newlines(str(doc))}\n{'=' * 20}")
        print()
    return bd

def load_bd (name, url_vect: str, url_text: str, verbose=0):
    """ Функция организует очередность загрузки Базы знаний.
        Сначала идет загрузка векторной базы, если она не загружается,
        то загружается база в текстовом формате и потом преобразуется в векторную."""
    try:
        bd = load_bd_vect(name, url_vect, verbose)
        print("Загрузка векторной Базы знаний выполнена успешно.")
        return bd
    except Exception as e:
        print("По указанной ссылке векторной Базы знаний нет.")
        print(e)
        print("\nИдет загрузка текстовой Базы знаний...")
        try:
            bd = load_bd_text(url_text, verbose)
            print("\nЗагрузка текстовой Базы знаний выполнена успешно.")
            return bd
        except Exception as e:
            print("\nПо указанной ссылке текстовой Базы знаний нет.")
            print(e)
            print("\nОшибка загрузки!!")


def archive_bd(folder_to_zip, output_filename, index_db):
#@title Архивирование Базы знаний (при необходимости).

# В дальнейшем архив нужно поместить на GitHub
    # Сохранение папки с векторной Базой знаний.
    index_db.save_local(folder_to_zip)
    # Архивирование папки с векторной Базой знаний
    with zipfile.ZipFile(output_filename, 'w') as zip:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                zip.write(os.path.join(root, file))
    print(f'База знаний - заархивирована. Имя файла - {output_filename}.')


# Данные по названиям модели и стоимости токена на 22.11.2023.
# https://openai.com/pricing#language-models
# Псевдоним = ['Имя модели', 'Цена токена - вопроса', 'Цена токена - ответа'].
MODEL_GPT_4_1106_PREVIEW = ['gpt-4-1106-preview', 0.01, 0.03]   # 128K tokens
MODEL_GPT_3_5_TURBO_1106 = ['gpt-3.5-turbo-1106', 0.001, 0.002] #  16K tokens

MODEL_COST = {x[0]:x for x in [MODEL_GPT_4_1106_PREVIEW, MODEL_GPT_3_5_TURBO_1106]}



