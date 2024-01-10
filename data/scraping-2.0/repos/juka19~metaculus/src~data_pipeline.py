import sqlite3
import urllib.request
import json
import re

from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import SequentialChain, LLMChain
from langchain.callbacks import FileCallbackHandler
from airflow import DAG
from airflow.operators.python import PythonOperator
from loguru import logger
import pandas as pd

logfile = "logs/metaculus.log"
logger.add(logfile, colorize=True, enqueue=True, backtrace=True, diagnose=True, level="DEBUG")
handler = FileCallbackHandler(logfile)

def read_data(sql_path):
    conn = sqlite3.connect(sql_path)
    data = pd.read_sql('select * from metaculus', conn)
    return data


def get_additional_data(id):
    endpoint = f'https://www.metaculus.com/api2/questions/{id}/'
    url = urllib.request.urlopen(endpoint)
    response = url.read()
    data = json.loads(response)
    data['description']
    
    if data['possibilities']['type'] == 'continuous':
        return [data['description'], data['publish_time'], data['resolve_time'], data['active_state'], data['possibilities']]
    else:
        return [data['description'], data['publish_time'], data['resolve_time'], data['active_state']]


def remove_markdown_links(text):
    pattern = r'\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, r'\1', text)


task1 = PythonOperator(
    dag='metaculus_prep',
    task_id='read_data',
    python_callable=read_data
)

task2 = PythonOperator(
    dag='metaculus_prep',
    task_id='get_additional_data',
    python_callable=get_additional_data
)

task3 = PythonOperator(
    dag='metaculus_prep',
    task_id='remove_markdown_links',
    python_callable=remove_markdown_links
)







def langchain_pipeline(record):
    



data2 = pd.read_pickle('data/metaculus_questions.pkl')

data['id'] = data.url.str.extract('(\/\d{3,5})')
data['id'] = data.id.str.lstrip('\/')
data = data.dropna(subset=['id'])
data['id'] = data.id.astype(int)

data2 = data2.rename(columns={"title": 'title2', 'resolution': 'resolution2', 'url': 'url2'})
data2 = data2.drop_duplicates(subset=['title2'])

df = pd.merge(
    data,
    data2,
    how="inner",
    on="id"
)




tttt = [t for t in df.text if len(t) > 10]







with open("assets/HF_API_TOKEN.txt", "r") as f:
    hf_token = f.read()

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"pad_token_id": 11,
    "max_length": 1500,
    "do_sample": True,
    "top_k": 10,
    "num_return_sequences": 1,
    "trust_remote_code": True}
)









