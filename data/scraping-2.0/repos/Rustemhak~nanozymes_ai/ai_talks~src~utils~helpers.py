import base64
import os
import random
from pathlib import Path
from typing import List, Tuple

import requests

import streamlit as st
import pandas as pd

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from src.utils.YandexGPT import YandexLLM

import openai
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFium2Loader

api_key = os.environ['api_key']
folder_id = "b1g6krtrd2vcbunvjpg6"

# openai.api_key = os.environ.get('OPENAI_API_KEY', None)

# llm_model = 'gpt-4'
# chat = ChatOpenAI(temperature=0.0, model=llm_model, max_tokens=1024)
# embeddings = OpenAIEmbeddings()


def render_svg(svg: Path) -> str:
    """Renders the given svg string."""
    with open(svg) as file:
        b64 = base64.b64encode(file.read().encode("utf-8")).decode("utf-8")
        return f"<img src='data:image/svg+xml;base64,{b64}'/>"


def get_files_in_dir(path: Path) -> List[str]:
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files


def get_random_img(img_names: list[str]) -> str:
    return random.choice(img_names)

def input_fields(*argv):
    match argv:
        case ("K_m",):
            K_m = st.text_input(r"Введите значение $$K_m$$ в диапазоне от 0 до 3820 mM")
            return K_m, None
        case ("V_max",):
            V_max = st.text_input(r"Введите значение $$V_{max}$$ в диапазоне от 0 до 59885 $$\frac{mM}{s}$$")
            return None, V_max
        case ("K_m", "V_max"):
            K_m = st.text_input(r"Введите значение $$K_m$$ в диапазоне от 0 до 3820 mM")
            V_max = st.text_input(r"Введите значение $$V_{max}$$ в диапазоне от 0 до 59885 $$\frac{mM}{s}$$")
            return K_m, V_max
        case _:
            return None, None
    

def get_search_data(df: pd.DataFrame, K_m: str, V_max: str):
    """Function for search data in csv file for given K_m and V_max

    Args:
        K_m (str): field K_m in csv file
        V_max (str): field V_max in csv file
    
    Returns:
        str: search most similar strings from csv file
    """
    distance = None
    match K_m, V_max:
        case (_, None):
            # print("K_m", K_m)
            distance = lambda value: abs(float(value['Km, mM']) - float(K_m))
        case (None, _):
            # print("V_max", V_max)
            distance = lambda value: abs(float(value['Vmax, mM/s']) - float(V_max))
        case _:
            # print("K_m, V_max", K_m, V_max)
            # distance = lambda value: abs(float(value['Km, mM']) - float(K_m))
            distance = lambda value: np.sqrt((float(value['Vmax, mM/s']) - float(V_max)) ** 2 + (float(value['Km, mM']) - float(K_m)) ** 2)
            # distance = lambda value: abs(float(value['Km, mM']) - float(K_m)) + abs(float(value['Vmax, mM/s']) - float(V_max))
    # Создаем список для хранения расстояний между значениями и целевыми значениями
    distances = []

    # Проходимся по каждому значению в данных
    for _, value in df.iterrows():
        # Вычисляем расстояние между значениями K_m и V_max
        try:
            # print("value", value)
            # print("value['Vmax, mM/s']", value['Vmax, mM/s'])
            _distance = distance(value)
            # Добавляем расстояние и значение в список distances
            distances.append((_distance, value))
        except ValueError:
            continue

    # Сортируем список distances по расстоянию в порядке возрастания
    distances.sort(key=lambda x: x[0])

    # Возвращаем топ N наиболее похожих значений
    return [(distance, value) for distance, value in distances[:5]]


# def get_search_data(df: pd.DataFrame, K_m: str, V_max: str) -> List[pd.DataFrame]:
#     # Преобразуем K_m и V_max в числа
#     K_m = float(K_m) if K_m else None
#     V_max = float(V_max) if V_max else None

#     # Извлекаем значения K_m и V_max из данных
#     data = df[['Km, mM', 'Vmax, mM/s']]
#     data.columns = ['Km, mM', 'Vmax, mM/s']
#     # Убираем значения "no"
#     data = data[data['Km, mM'] != 'no']
#     data = data[data['Vmax, mM/s'] != 'no']
#     print("data: ", data)
#     print("+" * 20)
#     # Нормализируем данные
#     normalize_data = normalize(data, axis=0, norm='l2')
#     # Вычисляем скалярное произведение
#     similarities = cosine_similarity(data)
#     print("data['Km, mM']", data['Km, mM'])
#     print("data['Km, mM'].mean()", data['Km, mM'].mean())
#     print("data['Vmax, mM/s'].mean()", data['Vmax, mM/s'].mean())
#     # Нормализуем искомый вектор
#     if K_m is not None and V_max is not None:
#         finded_vec = normalize(pd.DataFrame([float(K_m), float(V_max)]), axis=0, norm='l2')
#     elif K_m is not None:
#         finded_vec = normalize(pd.DataFrame([float(K_m), data['Vmax, mM/s'].mean()]), axis=0, norm='l2')
#     elif V_max is not None:
#         finded_vec = normalize(pd.DataFrame([data['Km, mM'].mean(), float(V_max)]), axis=0, norm='l2')
#     # Ищем топ 5 похожих значений
#     print("finded_vec", finded_vec)
#     finded_vec = np.array(finded_vec).reshape(1, -1)
#     results = []
#     print(pd.DataFrame(finded_vec).shape, pd.DataFrame(data).shape)
#     distance = cosine_similarity(finded_vec, normalize_data)[0]
#     print("distance", distance, len(distance))
#     # distance.sort()
#     top_k_indices = np.argpartition(distance, 5)[:5]
#     print("top_k_indices", top_k_indices)
    
#     # Добавляем расстояние и значение в results
#     for k in top_k_indices:
#         print("k", k)
#         results.append((distance[k], df.iloc[k]))
#     # print("results", results)
#     return results




# def get_chatgpt_pdf_syntes(df: pd.DataFrame) -> str:
#     """Function for get chatgpt pdf syntes paragraph

#     Args:
#         df (pd.DataFrame): row from csv file that need to get syntes

#     Returns:
#         str: description of syntes from chatgpt
#     """
#     print("In get_chatgpt_pdf_syntes: ", df)
#     print("+" * 20)
#     print("type(df)", type(df))
#     try:
#         df.reset_index(inplace=True, drop=True)
#         index = 0
#         formula, size, link = df.loc[index, 'formula'], int(df.loc[index, 'length, nm']), df.loc[index, 'link']
#         paper_filename=link.split('/')[-1]+'.pdf'
#         print("paper_filename", paper_filename)
#         try:
#             loader = PyPDFium2Loader("ai_talks/assets/pdf/" + paper_filename) # mode="elements"
#         except ValueError:
#             data = 'We apologize, but our service could not find the information in the original article: ' + df.loc[index, 'link']
#             raise ValueError
#         else:
#             print("NOT ERROR IN LOADER")
#             data = loader.load()
#         db = DocArrayInMemorySearch.from_documents(
#             data,
#             embeddings
#         )
#         retriever = db.as_retriever()
#         qa_stuff = RetrievalQA.from_chain_type(
#             llm=chat,
#             chain_type = "stuff", # "stuff"  "map_reduce"  "refine"  "map_rerank"
#             retriever=retriever,
#             verbose=False
#         )
#         # query =  "How is synthesis carried out in this article?"
#         # \n\nSynthesis искать это
#         # какие реагенты и оборудование. попробовать агентов. если ты не нашёл слово синтез то попробуй поискать словосочетания с ключевыми словами ...
#         query = f'What is needed for synthesis of {size} nm or other size {formula} NPs? NPs means nanoparticles. If the article does not describe how the synthesis was carried out, but a link is given to another article where it is said about it, then give a link to the article as an answer. If the article does not say anything about synthesis, then answer it. Answer as fully as possible, try to take the maximum of the original text from the article. Your answer should consist of several paragraphs and be quite voluminous, while preserving the source text as much as possible'
#         response = qa_stuff.run(query)
#         return response
#     except BaseException:
#         data = 'We apologize, but our service could not find the information in the original article: ' + df.loc[index, 'link']
#         return data
        
#     # return data

def get_chatgpt_pdf_syntes(df: pd.DataFrame) -> str:
    """Function for get chatgpt pdf syntes paragraph

    Args:
        df (pd.DataFrame): row from csv file that need to get syntes

    Returns:
        str: description of syntes from chatgpt
    """
    data = ""
    try:
        df.reset_index(inplace=True, drop=True)
        index = 0
        formula, size, link = df.loc[index, 'formula'], int(df.loc[index, 'length, nm']), df.loc[index, 'link']
        data = df.loc[index, 'synthesis']
    except BaseException:
        data = 'We apologize, but our service could not find the information in the original article: ' + df.loc[index, 'link']
    return data
        

instructions = """Представь себе, что ты умный помощник для помощи химикам и биологам Nanozymes. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника по статье."""

chat = YandexLLM(api_key=api_key, folder_id=folder_id,
                instruction_text = instructions)


def chat_YGPT(query, path_file=""):
    # Промпт для обработки документов
    try:
        document_prompt = langchain.prompts.PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        data = ""
        try:
            if path_file != "":
                loader = PyPDFium2Loader("ai_talks/assets/pdf/" + path_file) # mode="elements"
                data = loader.load()
        except BaseException:
            data = ""
        # Промпт для языковой модели
        document_variable_name = "context"
        stuff_prompt_override = """
        Пожалуйста, посмотри на текст ниже и ответь на вопрос на русском языке, используя информацию из этого текста.
        Текст:
        -----
        {context}
        -----
        Вопрос:
        {query}"""
        prompt = langchain.prompts.PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query"]
        )

        # Создаём цепочку
        llm_chain = langchain.chains.LLMChain(llm=chat, prompt=prompt)
        chain = langchain.chains.StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        class A:
            pass
        data = [A()]
        data[0].page_content = data
        data[0].metadata = {"metadata": "metadata"}
        response = chain.run(input_documents=data, query="How is synthesis carried out in this article?")
    except BaseException as err:
        response = "We apologize, but our service could not find the information in the original articles."
        print("In chat_YGPT: ", err)
    return response
    