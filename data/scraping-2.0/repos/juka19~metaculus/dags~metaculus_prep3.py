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

from metaculus_prep import *
from metaculus_prep2 import read_json

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


llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={'max_num_batch_tokens':4096, 'max_tokens': 4096},
    huggingfacehub_api_token=hf_token
)
dag_id = f'metaculus_prep3'
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
        task_id='read_json',
        python_callable=read_json,
        op_kwargs={'json_path': '../../data/all_data2.json'}
    )
    task2 = PythonOperator(
        task_id='validate_input',
        python_callable=validate_input,
        op_kwargs={'data': task1.output}
    )
    task3 = PythonOperator(
        task_id='run_langchain_pipeline',
        python_callable=run_langchain_pipeline,
        op_kwargs={'data': task2.output, 'llm': llm}
    )
    task4 = PythonOperator(
        task_id='validate_output',
        python_callable=validate_output,
        op_kwargs={'data': task3.output}
    )
    task5 = PythonOperator(
        task_id='write_json_preds',
        python_callable=write_json,
        op_kwargs={'data': task4.output, 'output_path': '../../data/json_output3.json'}
    )
    task1 >> task2 >> task3 >> task4 >> task5