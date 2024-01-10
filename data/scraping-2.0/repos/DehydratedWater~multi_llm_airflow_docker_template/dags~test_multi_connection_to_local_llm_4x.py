import asyncio
import time
from airflow import DAG
from airflow.operators.python import PythonOperator
from langchain.chat_models import ChatOpenAI
from datetime import datetime, timedelta

default_args = {
    'owner': 'dehydratedwater',
    'depends_on_past': False,
    'start_date': datetime(2020, 1, 1),
}

prompt = """
[INST] 
Provided below segment of the text, list all specialistic terms, and ideas contained in this text, that list will be later used for mining relations between concepts in order of building graph of semantic relations. 

Example:
Format data as a JSON containing {"<<name of category>>": ["idea_1", "term_2", ...], "<<name of category>>": ["term_1", "term_2", "idea_3", ...]}. Categories should be general.

Text to extract categories:
for about 20 years the problem of properties of short - term changes of solar activity has been considered extensively .", "many investigators studied the short - term periodicities of the various indices of solar activity . several periodicities were detected , but the periodicities about 155 days and from the interval of @xmath3 $ ] days ( @xmath4 $ ] years ) are mentioned most often . first of them was discovered by @xcite in the occurence rate of gamma - ray flares detected by the gamma - ray spectrometer aboard the _ solar maximum mission ( smm ) . his periodicity was confirmed for other solar flares data and for the same time period @xcite .
Use text above to extract concepts, ideas, people, ect, find name for the category and format them into flat JSON containing lists. Return JSON within 
json {...}
[/INST]
Before creating JSON with the results I will list categories contained inside provided text and give them more compact and universal name:
"""

async def request_result_from_llm():
    model = "models/llama-2-13b-chat.Q4_K_M.gguf"

    llm1 = ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base="http://llm-server-1:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=2000,
                    request_timeout=500,
                    max_retries=1,
                    model_kwargs={
                        "logit_bias": {},
                    },
                    streaming=False,
                    )
    
    llm2 = ChatOpenAI(temperature=0.6,
                    model=model, 
                    openai_api_base="http://llm-server-2:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=2000,
                    request_timeout=500,
                    max_retries=1,
                    model_kwargs={
                        "logit_bias": {},
                    },
                    streaming=False,
                    )
    
    llm3 = ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base="http://llm-server-3:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=2000,
                    request_timeout=500,
                    max_retries=1,
                    model_kwargs={
                        "logit_bias": {},
                    },
                    streaming=False,
                    )
    
    llm4 = ChatOpenAI(temperature=0.6,
                    model=model, 
                    openai_api_base="http://llm-server-4:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=2000,
                    request_timeout=500,
                    max_retries=1,
                    model_kwargs={
                        "logit_bias": {},
                    },
                    streaming=False,
                    )
    start_time = time.time()

    iterations = 10

    async def run_iteration(llm, prompt, iterations):
        results = []
        for i in range(iterations):
            p = await llm.apredict(prompt)
            results.append(p)
            print(p)
            print('Iteration: ', i)
        return results

    p1 = run_iteration(llm1, prompt, iterations)
    p2 = run_iteration(llm2, prompt, iterations)
    p3 = run_iteration(llm3, prompt, iterations)
    p4 = run_iteration(llm4, prompt, iterations)
    results = await asyncio.gather(p1, p2, p3, p4)
    print(results[0])
    print(results[1])
    print(results[2])
    print(results[3])



    return f"Test iterations [count {iterations}] done in: {(time.time() - start_time)/60:.2f} minutes"


def run_async():
   loop = asyncio.get_event_loop()
   result = loop.run_until_complete(request_result_from_llm())
   return result

with DAG(
    dag_id='test_multi_local_llm_dag_4x',
    default_args=default_args,
    description='Test Multi Local LLM DAG',
    catchup=False,
    schedule_interval='@once',
) as dag:
    llm_operator = PythonOperator(
        task_id='test_multi_local_llm_task',
        python_callable=run_async,
    )

    llm_operator
