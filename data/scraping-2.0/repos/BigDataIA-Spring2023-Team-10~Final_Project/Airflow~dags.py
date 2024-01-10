from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from google.cloud import storage
from google.oauth2 import service_account
import os
import pymysql
import openai
import json
import pendulum
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.hooks.base_hook import BaseHook
from google.auth.credentials import AnonymousCredentials
from airflow.models.param import Param
from dotenv import load_dotenv
import pandas as pd
load_dotenv()


openai.api_key = os.getenv("open_api_key")
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 28),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bigdata7245-finalproject',
    default_args=default_args,
    description='Big Data 7245 Final Project',
    schedule_interval=timedelta(minutes=5),
    catchup=False
)

#Util functions
def init_gcp_bucket():
    # Get the credentials from Airflow Admin Connection'
    your_gcp_keys = { 
        "type": os.environ.get('type'),
        "project_id": os.environ.get('project_id'),
        "private_key_id": os.environ.get('private_key_id'),
        "private_key": os.environ.get('private_key').replace('\\n', '\n'),
        "client_email": os.environ.get('client_email'),
        "client_id": os.environ.get('client_id'),
        "auth_uri": os.environ.get('auth_uri'),
        "token_uri": os.environ.get('token_uri'),
        "auth_provider_x509_cert_url": os.environ.get('auth_provider_x509_cert_url'),
        "client_x509_cert_url": os.environ.get('client_x509_cert_url')
    }
    
    # Set the credentials
    credentials = service_account.Credentials.from_service_account_info(your_gcp_keys)
    storage_client = storage.Client(credentials=credentials)
    return storage_client

def list_all_file():
    files_list=[]
    storage_client=init_gcp_bucket()
    # Get the bucket object
    bucket = storage_client.get_bucket(os.getenv("bucket_name")) 
    # Iterate through all the blobs in the bucket
    for blob in bucket.list_blobs(prefix='raw_reviews/'):
        # Print the name of the file
        print(str(blob.name).split('/')[1].split('.')[0])
        files_list.append(str(blob.name).split('/')[1].split('.')[0])
    return files_list[1:]

def write_database(restaurant_id):
    try:  
        conn = pymysql.connect(
                host = os.getenv("host"), 
                user = os.getenv("user"),
                password = os.getenv("password"),
                db = os.getenv("db"))
        cursor = conn.cursor()
        sql_insert=f"INSERT INTO restaurant_request_process ( restaurant_id ) VALUES (%s);"
        record=(restaurant_id)
        cursor.execute(sql_insert,record)
        conn.commit()
        cursor.close()
    except Exception as error:
        print("Failed to insert record into table {}".format(error))

def move_reviews(**kwargs):
    review_files=list_all_file()
    for file in review_files:
        storage_client=init_gcp_bucket()
        bucket = storage_client.get_bucket(os.getenv("bucket_name"))
        blob_name = f"raw_reviews/{file}.csv"
        source_blob = bucket.blob(blob_name)
        # copy to new destination
        bucket.copy_blob(source_blob, bucket, f"processed_reviews/{file}.csv")
        # delete in old destination
        source_blob.delete()
        os.remove(f"{file}.csv")
        
def get_review_files(**kwargs):
    review_files=list_all_file()
    storage_client=init_gcp_bucket()
    for file in review_files:
        bucket = storage_client.get_bucket(os.getenv("bucket_name"))
        blob_name = f"raw_reviews/{file}.csv"
        blob=bucket.blob(blob_name)
        blob.download_to_filename(f"{file}.csv")
    ti = kwargs['ti']
    ti.xcom_push(key='file_name', value=review_files)
    
    
def upsert_text_embedings(**kwargs):
    import pandas as pd
    import pinecone
    index_name = os.environ.get('pinecone_index')
    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=os.environ.get('pinecone_api_key'),
        environment=os.environ.get('pinecone_env')  # find next to api key in console
    )
    # check if 'openai' index already exists (only create index if not)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    # connect to index
    index = pinecone.Index(index_name)
    MODEL = os.environ.get('openai_embedings')
    files = list_all_file()
    print(files)
    for file in files:
        print(file)
        df = pd.read_csv(f"{file}.csv")
        df['stars'] = df['stars'].astype(str)
        batch_size = 32 
        resturant_id=file.split('-id-')[1]
        for i in range(0, len(df), batch_size):
            # set end position of batch
            i_end = min(i+batch_size, len(df))
            # get batch of lines and IDs
            lines_batch = df['text'][i: i+batch_size].to_list()
            ids_batch = [str(n) for n in range(i, i_end)]
            # create embeddings
            res = openai.Embedding.create(input=lines_batch, engine=MODEL)
            embeds = [record['embedding'] for record in res['data']]
            # prep metadata and upsert batch
            meta = [{'business_id': df['business_id'][j], 'text': df['text'][j], 'stars': df['stars'][j], 'date': df['date'][j]} for j in range(i, i_end)]
            #meta = df['business_id'][i: i_end].reset_index(drop=True).apply(lambda x: {'business_id': x}).to_list()
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert),namespace=resturant_id)
        write_database(resturant_id)
    
t1 = PythonOperator(
    task_id='download_file',
    python_callable=get_review_files,
    dag=dag,
    provide_context=True,
)

t2 = PythonOperator(
    task_id='upsert_embedings',
    python_callable=upsert_text_embedings,
    dag=dag,
    provide_context=True,
)
t3 = PythonOperator(
    task_id='move_reviews',
    python_callable=move_reviews,
    op_kwargs={'folder': 'processed_reviews/'},
    dag=dag,
    provide_context=True,
)

t1 >> t2 >> t3 