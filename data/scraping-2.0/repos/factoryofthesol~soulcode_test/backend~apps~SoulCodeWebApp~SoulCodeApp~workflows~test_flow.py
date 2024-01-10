import django
import pandas as pd
from workflows.models import Hash
from workflows.marvinai import MarvinAI
from workflows.models import data
from prefect import flow,task
from typing import Optional
from marvin.prompts.library import System, User, ChainOfThought
import csv
import os
import openai

django.setup()

# from django.contrib.auth.models import User

@task
def run():
    file = open('C:/Users/dell/DjangoPrefectMarvin/django-prefect-example/workflows/Text Stands (PromptPathway).xlsx')
    sheet_name = "SCT#"
    # read_file = csv.reader(file)
    df = pd.read_excel(file, sheet_name=sheet_name)

    for index, row in df.iterrows():
        Hash.objects.create(ID = row[0],Input = row[1],Tags = row[2])


@task
def process_input(input_data):
    tool, value = input_data.split("#")
    return f"Processed {tool}: {value}"


@flow
def my_flow():
    inputs = Hash.objects.values('ID').distinct('Inputs')
    tags = Hash.objects.values('ID').distinct('Tags')
    processed_results = process_input.map(inputs)
    for result in processed_results:
        print(result)


class ExpertSystem(System):
        # content: str = (
        #     "You are a world-class expert on {{ topic }}. "
        #     "When asked questions about {{ topic }}, you answer correctly."
        # )
    content: result[0]
    topic: tags[0]

    prompt = (
            ExpertSystem(topic= tags[0])
            | User(input[0]))
            | ChainOfThought()  # Tell the LLM to think step by step
    )


     return prompt.dict()

##Example prompt generated with inputs
'''
[
    {
        'role': 'system',
        'content': 'You are a world-class expert on python. 
                    When asked questions about python, you 
                    answer correctly.'
    },
    {
        'role': 'user',
        'content': 'I need to know how to write a function to
                    find the nth Fibonacci number.'
    },
    {  'role': 'assistant', 
        'content': "Let's think step by step."
    }
]

'''

prompt_new = prompt.dict()

@task
for value in prompt_new[0].keys:
    my_list = []
    my_list.append(value)

    return my_list

##Storing generated prompts in postgres again by mapping ids for generated prompts

@task
def add_task_to_db(Add_prompts):
    connection = psycopg2.connect(
        database='Hashdb',
        user="postgres",
        password="909090",
        host="localhost",
        port="5432"
    )
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Hash VALUES (my_list);", (Add_prompts,))
    connection.commit()
    cursor.close()
    connection.close()



@task
def fetch_tasks_from_db():
    connection = psycopg2.connect(
        database='Hashdb',
        user="postgres",
        password="909090",
        host="localhost",
        port="5432"
    )
    )
    cursor = connection.cursor()
    cursor.execute("SELECT id, my_list FROM hash")
    tasks = cursor.fetchall()
    cursor.close()
    connection.close()
    return tasks

##Marvin ai uses chatgpt to generate texts to create images from it and pass it over to replica api

@task
def generate_chat_response(my_list):
    openai.api_key = "your_openai_api_key"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=my_list,
        max_tokens=100
    )
    return response.choices[0].text

@task
def process_with_marvin(data):
    model = MarvinAI()
    prediction = model.predict(data)
    return prediction

with Flow("DjangoPrefectFlow") as flow:
    data = my_list
    chat_response = generate_chat_response(my_list)
    prediction = process_with_marvin(chat_response)


