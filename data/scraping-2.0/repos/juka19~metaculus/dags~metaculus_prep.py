import logging
import sqlite3
import urllib.request
import json
import re
import os
from datetime import datetime, timedelta
import time 

from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import SequentialChain, LLMChain
from langchain.callbacks import FileCallbackHandler
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.log.logging_mixin import LoggingMixin
from langchain.schema import prompt_template
from huggingface_hub import login
from numpy import random, rec
import pandas as pd
from cerberus import Validator
from fp.fp import FreeProxy
from fp.errors import FreeProxyException
from sqlalchemy.engine import result

with open("../../assets/HF_API_TOKEN.txt", "r") as f:
    hf_token = f.read()

# hf_token = os.environ['HF_API_TOKEN']

schema = {
    'id': {'type': 'integer', 'coerce': int},
    'url': {'type': 'string', 'regex': 'https:\/\/www.metaculus.com\/questions\/\d{3,5}\/.*'},
    'title': {'type': 'string'},
    'res_criteria': {'type': 'string'},
    'resolution': {'type': 'string'},
    'forecast_type': {'type': 'string', 'allowed': ['binary', 'date_range', 'numerical']},
    'description': {'type': 'string'},
    'possibilities': {
        'type': ['dict', 'string'],
        'allow_unknown': True,
        'schema': {
            'scale': {
                'type': 'dict',
                'allow_unknown': True,
                'schema': {
                    'max': {'type': ['number', 'string']},
                    'min': {'type': ['number', 'string']}
                }
            }
        }
    }
}
v = Validator(schema, allow_unknown=True)

def read_data(sql_path):
    conn = sqlite3.connect(sql_path)
    data = pd.read_sql('select * from metaculus', conn)
    data['id'] = data.url.str.extract('(\/\d{3,5})')
    data['id'] = data.id.str.lstrip('\/')
    data = data.dropna(subset=['id'])
    data['id'] = data.id.astype(int)
    return data

def remove_markdown_links(text):
    pattern = r'\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, r'\1', text)

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def query_api_proxy(id):
    xyc = True
    prox_list = FreeProxy().get_proxy_list(3)
    while xyc:
        if len(prox_list) == 0: 
            time.sleep(60)
            prox_list = FreeProxy().get_proxy_list(3)
        try:
            prox = prox_list.pop() 
            proxy_support = urllib.request.ProxyHandler({'http': prox, 'https': prox})
            opener = urllib.request.build_opener(proxy_support)
            urllib.request.install_opener(opener)
            req = urllib.request.urlopen(f'https://www.metaculus.com/api2/questions/{id}/', timeout=3).read()
            result = json.loads(req)
            xyc = False
        except Exception as e:
            continue
    return result

def query_api(id):
    endpoint = f'https://www.metaculus.com/api2/questions/{id}/'
    url = urllib.request.urlopen(endpoint, timeout=10)
    response = url.read()
    result = json.loads(response)
    return result

def get_additional_data(data):
    records = []
    for id in data['id']:
        try:
            LoggingMixin().log.info(f'Querying API for id {id}')
            result = query_api(id)
        except Exception as e:
            LoggingMixin().log.info(f'Querying API with proxy for id {id}')
            LoggingMixin().log.error(e)
            result = query_api_proxy(id)
        if result['possibilities']['type'] == 'continuous':
                record = {
                    'description': result['description'],
                    'publish_time': result['publish_time'],
                    'resolve_time': result['resolve_time'],
                    'active_state': result['active_state'],
                    'possibilities': result['possibilities'],
                }
        else:
            record = {
                'description': result['description'],
                'publish_time': result['publish_time'],
                'resolve_time': result['resolve_time'],
                'active_state': result['active_state'],
                'possibilities': ''
            }
        record = {**record, **data[data['id'] == id].to_dict('records')[0]}
        record['description'] = remove_markdown_links(record['description'])
        record['description'] = striphtml(record['description']).strip().replace('\n', ' ').replace('\r', '')
        records.append(record)
    return records

def validate_input(data):
    for record in data:
        LoggingMixin().log.info(f'Validating record {record["id"]}')
        if not v.validate(record):
            raise ValueError(v.errors)
    return data


def langchain_pipeline(record, llm=None):
    if llm is None:
        llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            huggingfacehub_api_token=hf_token,
            model_kwargs={"pad_token_id": 11,
            "max_length": 10000,
            "max_tokens": 10000,
            "do_sample": True,
            "top_k": 10,
            "num_return_sequences": 1,
            "trust_remote_code": True}
        )
    template = """
    For each instruction, write a high-quality description about the most capable and suitable agent to answer the instruction. In second person perspective.
    [Instruction]: Make a list of 5 possible effects of deforestation.
    [Agent Description]: You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.
    [Instruction]: Identify a descriptive phrase for an eclipse.
    [Agent Description]: You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.
    [Instruction]: {question}
    [Agent Description]:
    """.strip()
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    agent_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key='agent_descr'
    )
    
    if record['forecast_type'] == 'binary':
        
        template = """
        {agent_descr}
        Given above identity background, please answer the following instruction. The expert takes all information into consideration and answers this question either with Yes or with No, nothing else.
        [Additional context information]: {context}
        [Question]: {question}
        [Expert Prediction (Yes/No)]:
        """.strip()
        prompt_template = PromptTemplate(
            input_variables=["agent_descr", "question", "context"],
            template=template
        )
        
        answer_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_key='answer'
        )
        
        overall_chain = SequentialChain(
            chains=[agent_chain, answer_chain],
            input_variables=["question", "context"],
            output_variables=["agent_descr", "answer"]
        )
        if record['description'] == '':
            context = record['res_criteria']
        else:
            context = record['description']
        args = {"question": record['title'], "context": context}
    
    elif record['forecast_type'] == 'date_range':
        
        template = """
        {agent_descr}
        Now given above identity background, please answer the following instruction. You are required to answer with an exact date estimate in the format of YY-MM-DD.
        The maximum date is {max} and the minimum date is {min}.
        [Additional context information]: {context}
        [Question:] {question}
        [Expert prediction (YY-MM-DD)]:
        """.strip()
        prompt_template = PromptTemplate(
            input_variables=["agent_descr", "question", "context", "max", "min"],
            template=template
        )
        
        answer_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_key='answer'
        )
        
        overall_chain = SequentialChain(
            chains=[agent_chain, answer_chain],
            input_variables=["question", "context", "max", "min"],
            output_variables=["agent_descr", "answer"]
        )
        if record['description'] == '':
            context = record['res_criteria']
        else:
            context = record['description']
        args = {"question": record['title'], "context": context, 
                "max": record["possibilities"]["scale"]["max"], "min": record["possibilities"]["scale"]["min"]}
    
    elif record['forecast_type'] == 'numerical':
        
        template = """
        {agent_descr}
        Now given above identity background, please answer the following instruction. You are required to answer with an numerical estimate. The expert only answers with the estimated number.
        The maximum number is {max} and the minimum number is {min}.
        [Additional context information]: {context}
        [Question]: {question}
        [Expert prediction (Number)]:
        """.strip()
        prompt_template = PromptTemplate(
            input_variables=["agent_descr", "question", "context", "max", "min"],
            template=template
        )
        
        answer_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_key='answer'
        )
        
        overall_chain = SequentialChain(
            chains=[agent_chain, answer_chain],
            input_variables=["question", "context", "max", "min"],
            output_variables=["agent_descr", "answer"]
        )
        if record['description'] == '':
            context = record['res_criteria']
        else:
            context = record['description']
        args = {"question": record['title'], "context": context, 
                "max": record["possibilities"]["scale"]["max"], "min": record["possibilities"]["scale"]["min"]}
    
    result = overall_chain(args)
    
    result['id'] = record['id']
    
    return result

def run_langchain_pipeline(data, llm):
    results = []
    for record in data:
        LoggingMixin().log.info(f'Running langchain pipeline for record {record["id"]}')
        result = langchain_pipeline(record, llm=llm)
        results.append(result)
    return results

v2 = Validator(
    {
        'agent_descr': {'type': 'string'},
        'answer': {'type': ['string', 'number']},
        'context': {'type': 'string'},
        'question': {'type': 'string'},
        'id': {'type': 'integer'}
    },
    allow_unknown=True
)

def validate_output(data):
    for output in data:
        LoggingMixin().log.info(f'Validating output for record {output["id"]}')
        if not v2.validate(output):
            raise ValueError(v2.errors)
    return data

def write_to_json(record, output_path):
    a = []
    if not os.path.isfile(output_path):
        a.append(record)
        with open(output_path, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(output_path) as feedsjson:
            feeds = json.load(feedsjson)
        feeds.append(record)
        with open(output_path, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

def write_json(data, output_path):
    for record in data:
        write_to_json(record, output_path)
    LoggingMixin().log.info(f'Wrote {len(data)} records to {output_path}')


llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={'max_num_batch_tokens':4096},
    huggingfacehub_api_token=hf_token,
)

dag_id = f'metaculus_prep'
dag = DAG(
    dag_id=dag_id,
    schedule_interval='@once', 
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'start_date': datetime.today(),
        },
    catchup=False,
)
with dag:
    task1 = PythonOperator(
        task_id='read_data',
        python_callable=read_data,
        op_kwargs={'sql_path': '../../data/metaculus.db'}
    )
    task2 = PythonOperator(
        task_id='get_additional_data',
        python_callable=get_additional_data,
        op_kwargs={'data': task1.output}
    )
    task3 = PythonOperator(
        task_id='validate_input',
        python_callable=validate_input,
        op_kwargs={'data': task2.output}
    )
    task7 = PythonOperator(
        task_id='write_json_full',
        python_callable=write_json,
        op_kwargs={'data': task3.output, 'output_path': '../../data/all_data2.json'}
    )
    task4 = PythonOperator(
        task_id='run_langchain_pipeline',
        python_callable=run_langchain_pipeline,
        op_kwargs={'data': task3.output, 'llm': llm}
    )
    task5 = PythonOperator(
        task_id='validate_output',
        python_callable=validate_output,
        op_kwargs={'data': task4.output}
    )
    task6 = PythonOperator(
        task_id='write_json_preds',
        python_callable=write_json,
        op_kwargs={'data': task5.output, 'output_path': '../../data/json_output2.json'}
    )
    task1 >> task2 >> task3 >> task7 >> task4 >> task5 >> task6