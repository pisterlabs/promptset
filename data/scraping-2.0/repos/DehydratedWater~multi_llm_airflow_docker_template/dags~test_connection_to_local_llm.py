from airflow import DAG
from airflow.operators.python import PythonOperator
from langchain.chat_models import ChatOpenAI
from datetime import datetime, timedelta

default_args = {
    'owner': 'dehydratedwater',
    'depends_on_past': False,
    'start_date': datetime(2020, 1, 1),
}


def request_result_from_llm():
    model = "models/llama-2-13b-chat.Q5_K_M.gguf"

    llm = ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base="http://llm-server:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=2000,
                    model_kwargs={
                        "logit_bias": {},
                    },
                    streaming=True,
                    )
    
    full_result = ""
    
    for chunk in llm.stream("Write 10 point list about cats: "):
        print(chunk.content, end="", flush=True)
        full_result += chunk.content

    return full_result

with DAG(
    dag_id='test_single_local_llm_dag',
    default_args=default_args,
    description='Test Local LLM DAG',
    catchup=False,
    schedule_interval='@once',
) as dag:
    llm_operator = PythonOperator(
        task_id='test_local_llm_task',
        python_callable=request_result_from_llm,
    )

    llm_operator
