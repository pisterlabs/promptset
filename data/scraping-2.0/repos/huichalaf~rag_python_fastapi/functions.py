import json
import openai
import dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
import time as tm
from PIL import Image
import io
import sys
from hashlib import sha256
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def calculate_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


async def change_filename(filename):
    filename = filename.replace(" ", "_")
    filename = filename.replace(":", "_")
    filename = filename.replace("(", "_")
    filename = filename.replace(")", "_")
    filename = filename.replace("?", "_")
    filename = filename.replace("¿", "_")
    filename = filename.replace("!", "_")
    filename = filename.replace("¡", "_")
    filename = filename.replace(";", "_")
    filename = filename.replace(",", "_")
    filename = filename.replace("/", "_")
    return filename

async def get_user(user):
    #dividimos por ;
    user = user.split(";")[0]
    #quitamos user=
    user = user.replace("user=", "")
    user = user.replace("%40", "@")
    return user
    
async def get_user_input(text):
    #buscamos donde dice Answer this question:
    index = text.find("Answer this question:")
    #tomamos el texto desde ahí hasta el final
    text = text[index:]
    return text

async def imagen_a_bytesio(ruta_imagen):
    imagen = Image.open(ruta_imagen)
    buffer = io.BytesIO()
    imagen.save(buffer, format='PNG')
    return buffer.getvalue()

async def identify_numbers(text):
    text_list = list(text)
    numbers = []
    number = ''
    for t in text_list:
        if t.isdigit():
            number += t
        elif t == '.':
            number += t
        elif number != '':
            numbers.append(float(number))
            number = ''
    if number != '':
        numbers.append(float(number))
    return numbers

async def identify_operators(text):
    operators = []
    
    if '**' in text:
        operators.append('**')
        text = text.replace('**', '  ')
    
    text_list = list(text)
    for t in text_list:
        if t in ['+', '-', '*', 'x', '/', '^', 'sqrt', '√']:
            operators.append(t)
    return operators

async def pre_process_math_prompt(prompt):
    
    prompt_lista = list(prompt)
    prompt_to_return = ''
    for i in range(len(prompt_lista)):
        if prompt_lista[i].isdigit():
            prompt_to_return += prompt_lista[i]
            if i+1 < len(prompt_lista):
                if prompt_lista[i+1] in ['+', '-', '*', 'x', '/', '^', 'sqrt', '√']:
                    prompt_to_return += ' '
        elif prompt_lista[i] == '.':
            prompt_to_return += prompt_lista[i]
        elif prompt_lista[i] in ['+', '-', 'x', '/', '^', 'sqrt', '√']:
            prompt_to_return += ' ' + prompt_lista[i] + ' '
        elif prompt_lista[i] == '*' and prompt_lista[i+1] == '*':
            prompt_to_return += ' ' + prompt_lista[i] + prompt_lista[i+1] + ' '
        elif prompt_lista[i] == '*' and prompt_lista[i+1] != '*' and prompt_lista[i-1] != '*':
            prompt_to_return += ' ' + prompt_lista[i] + ' '
        else:
            if prompt_lista[i] != '*':
                prompt_to_return += prompt_lista[i]
            else:
                pass
    return prompt_to_return

async def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    result = await openai.Embedding.acreate(input=[text], model=model)
    result_data = result['data'][0]['embedding']
    return result_data

async def embed_functions():
    with open("functions.json", "r") as json_file:
        data = json.load(json_file)
    functions = data
    embeddings = []
    for function in functions:
        function_name = function["name"]
        function_description = function["description"]
        function_text = function_name + " " + function_description
        embeddings.append(get_embedding(function_text))
    functions_embeddings = pd.DataFrame()
    functions_embeddings["name"] = [function["name"] for function in functions]
    functions_embeddings["description"] = [function["description"] for function in functions]
    functions_embeddings["embedding"] = embeddings
    functions_embeddings.to_csv("embed_functions.csv", index=False)
    return True

async def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))

async def read_embeddings_from_csv(file_path):
    functions_embeddings = pd.read_csv(file_path)
    embeddings = functions_embeddings["embedding"].tolist()
    embeddings = [eval(embedding) for embedding in embeddings]
    names = functions_embeddings["name"].tolist()
    descriptions = functions_embeddings["description"].tolist()
    return names, descriptions, embeddings

async def get_functions(prompt):
    embedding_prompt = await get_embedding(prompt)
    with open("functions.json", "r") as json_file:
        data = json.load(json_file)
    functions = data
    names, descriptions, functions_embeddings = await read_embeddings_from_csv("embed_functions.csv")
    similarities = []
    for embedding in functions_embeddings:
        similarities.append(await cosine_similarity(embedding_prompt, embedding))
    #filtramos por todas las que tengan similarities menor a 0.7
    functions_past = functions
    functions = [function for function, similarity in zip(functions, similarities) if similarity > 0.725]
    similarities_past = similarities
    similarities = [similarity for similarity in similarities if similarity > 0.725]
    if len(functions) == 0:
        #retornamos la funcion mas similar    
        max_similarity = max(similarities_past)
        index = similarities_past.index(max_similarity)
        function = functions_past[index]
        return [function]
    try:
        max1 = max(similarities)
        index1 = similarities.index(max1)
        function1 = functions[index1]
        similarities.pop(index1)
        functions.pop(index1)
    except:
        return False
    try:
        max2 = max(similarities)
        index2 = similarities.index(max2)
        function2 = functions[index2]
        similarities.pop(index2)
        functions.pop(index2)
    except:
        return [function1]
    try:
        max3 = max(similarities)
        index3 = similarities.index(max3)
        function3 = functions[index3]
        similarities.pop(index3)
        functions.pop(index3)
    except:
        return function1, function2
    try:
        max4 = max(similarities)
        index4 = similarities.index(max4)
        function4 = functions[index4]
        similarities.pop(index4)
        functions.pop(index4)
    except:
        return function1, function2, function3
    try:
        max5 = max(similarities)
        index5 = similarities.index(max5)
        function5 = functions[index5]
        similarities.pop(index5)
        functions.pop(index5)
    except:
        return function1, function2, function3, function4
    
    return function1, function2, function3, function4, function5

async def rename_by_hash(path, text, user):
    files_path = os.getenv("FILES_PATH")
    extension = path.split(".")[-1]
    if type(text) == list:
        text = " ".join(text)
    hash_object = sha256(text.encode())
    hex_dig = hash_object.hexdigest()
    new_name = hex_dig+"."+extension
    return new_name

async def get_all_text(data):
    keys = list(data.keys())
    text = []
    for i in keys:
        text.extend(data[i]['text'].tolist())
    return text

async def get_all_embeddings(data):
    keys = list(data.keys())
    embeddings = []
    for i in keys:
        embeddings.extend(data[i]['embedding'].tolist())
    return embeddings

async def cosine_similarity(a, b):
    result = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return result

async def divide_text_str(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
    )
    text_pre_pure = text_splitter.create_documents([text])
    text_pure = text_splitter.split_text(text)
    return text_pure