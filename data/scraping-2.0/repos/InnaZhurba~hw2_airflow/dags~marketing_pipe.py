from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import easyocr
import ssl
from urllib.parse import urljoin
from openai import OpenAI
import time


ssl._create_default_https_context = ssl._create_unverified_context

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'marketing_material_pipeline',
    default_args=default_args,
    schedule_interval=None, 
)

def get_all_image_addresses(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')
        image_addresses = [urljoin(url, img['src']) for img in img_tags if 'src' in img.attrs]

        results = []
        for address in image_addresses: 
            if address[-3:] == 'jpg':
                results.append(address)
            if len(results) == 10:
                break
            
        print (results)

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

def perform_ocr(**kwargs):
    url = "https://www.whosmailingwhat.com/blog/best-direct-mail-marketing-examples/"
    image_adresses = get_all_image_addresses(url)
    
    reader = easyocr.Reader(['en'])
    results = []
    for image_url in image_adresses:
        print (image_url)
        try:
            result = reader.readtext(image_url)
            results.append(result)
        except:
            print('image not found')
            continue
        
    return results

def extract_company_info_from_ocr(ocr_result):
    results = []

    for result in ocr_result:
        result_ = str(result)

        client = OpenAI(api_key="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")

        chat_completion = client.chat.completions.create(
            messages=[
                {"role" : "user",  
                "content" : "Return all the company information (name, contacts etc.) from the data: " + result_,
            }], model = "gpt-3.5-turbo",
        )
        
        generated_text = chat_completion
        print(generated_text)
        
        results.append(generated_text)


    return results

def extract_company_info(**kwargs):
    ocr_result = kwargs['task_instance'].xcom_pull(task_ids='perform_ocr_task')
    
    company_infos = []
    for result in ocr_result:
        
        company_info = extract_company_info_from_ocr(result)
        company_infos.append(company_info)
        
        print(company_info)
        
    return company_infos

perform_ocr_task = PythonOperator(
    task_id='perform_ocr_task',
    python_callable=perform_ocr,
    provide_context=True,
    dag=dag,
)

extract_company_info_task = PythonOperator(
    task_id='extract_company_info_task',
    python_callable=extract_company_info,
    provide_context=True,
    dag=dag,
)

perform_ocr_task >> extract_company_info_task 
