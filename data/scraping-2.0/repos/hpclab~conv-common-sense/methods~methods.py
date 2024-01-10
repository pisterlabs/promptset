import pickle
import ast
import re

def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        print(val)
        return val

def convert_column_from_text_to_list(column:list):
    return [[re.sub('^\s+','',str(entity)) for entity in literal_return(str(x))] for x in column]


def write_dict(dictionary_data,output_path):
    with open(output_path,'wb') as file:
        pickle.dump(dictionary_data, file)

def load_dict(input_path):
    with open(input_path,"rb") as file:
        data = pickle.load(file)
    return data

def write_list(lista,outpath):
    with open(outpath, 'w') as f:
        for el in lista:
            f.write("%s\n" % el)
            
def read_list(path):
    with open(path, 'r') as f:
        lines= f.readlines()
        lines = [literal_return(x) for x in lines]
    return lines

import openai
import time
openai.api_key = 'insert_api_key'
def query_chatgpt(query):
    try:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0.6,
        max_tokens=150,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
        )
        return response.choices[0].text.replace('\n','')
    except:
        time.sleep(5)
        try:query_chatgpt(query)
        except:
            time.sleep(10)
            try:query_chatgpt(query)
            except:
                time.sleep(15)
                query_chatgpt(query)

