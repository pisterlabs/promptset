import openai
import os
import datetime
from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API key from the environment
api_key = os.getenv('OPENAI_API_KEY')
print(api_key)
dag_path = os.getenv('dag_file_path')
print(dag_path)
# Initialize the OpenAI API client
openai.api_key = api_key

import random

def enhance_prompt(user_prompt):
    enhanced_prompt = f"""Generate the Python code for an Apache Airflow DAG that creates a machine learning pipeline using the {user_prompt} algorithm. The DAG should include the following tasks with appropriate dependencies.\n
    take a dataset from skelaern and implement the DAG, the dag name should relevant to the algorithm name. generate a new DAG code similar to :{ref_dag}
"""
    return enhanced_prompt
def file_name(prompt):
    return prompt + '.txt'
def generate_airflow_dag(prompt):
    enhanced_prompt = enhance_prompt(prompt)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=enhanced_prompt,
        max_tokens=3000,  # Adjust as needed
        n=1,  # Number of responses to generate
        stop=None,  # You can specify a stop sequence to end the response
    )
#     response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": enhanced_prompt}
#     ]
# )
    # Specify the file path where you want to save the generated DAG

    current_date = datetime.date.today()
    file_path = str(current_date)+"_"+prompt+ ".py"
    file_path = dag_path+file_path

    # Save the generated DAG to the specified file path
    save_to_file(file_path, response.choices[0].text.strip())

    print(f"Generated Airflow DAG saved to: {file_path}")
    return file_path

def save_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)



ref_dag = """
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'John Doe',
    'depends_on_past': False,
    'start_date': datetime(2020, 1, 1)
}

dag = DAG('machine_learning_pipeline', default_args=default_args, schedule_interval=None)

# download link: https://www.kaggle.com/uciml/pima-indians-diabetes-database
def download_dataset():
    #Downloads the dataset from Kaggle
    # Code to download the dataset
    pass
def process_data():
    #Processes the data
    # Code to process the data
    pass
def split_data():
    #Splits the data into training and testing dataset
    # Code to split the data
    pass
def train_model():
    
    #Trains the model
    # Code to train the model using decision tree algorithm
    pass
def evaluate_model():
    #Evaluates the model
    # Code to evaluate the model
    pass

# Define tasks
download_task = PythonOperator(task_id='download_dataset', python_callable=download_dataset, dag=dag)

process_data_task = PythonOperator(task_id='process_data', python_callable=process_data, dag=dag)

split_data_task = PythonOperator(task_id='split_data', python_callable=split_data, dag=dag)

train_model_task = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

eval_task = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)

# Set task sequence
download_task >> process_data_task >> split_data_task >> train_model_task >> eval_task
"""
