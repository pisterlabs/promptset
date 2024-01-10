import os
import openai
import chromadb

from extract_info import get_info
from create_database import get_collection

from dotenv import load_dotenv, find_dotenv
from chromadb.utils import embedding_functions

path = "path/to/your/.env/file"

_ = load_dotenv(find_dotenv(filename=path))

openai.api_key = os.environ['OPENAI_API_KEY']

def get_answer(
        customer_need_response: str,
        customer_time_response: str,
        customer_level_response: str
    ) -> str:
    overall_prompt = get_overall_prompt(
        customer_need_response,
        customer_time_response,
        customer_level_response
    )

    output_dict = get_info(
        input=overall_prompt,
        path=path
    )

    need = output_dict["need"]
    min_time = output_dict["min_time"]
    max_time = output_dict["max_time"]
    level = output_dict["level"]

    min_time = 0 if min_time == '-' else int(min_time)
    max_time = 9999 if (max_time == '-' or max_time == 'inf') else int(max_time)

    results = get_results_from_db(need)

    filtered_results = []
    filtered_without_time = []
    filtered_without_level = []

    for result in results:
        result_t = result["time"]
        result_l = result["level"]
        if int(result_t) >= min_time and int(result_t) <= max_time:
            if level == '-':
                filtered_results.append(result)
            elif result_l == level:
                filtered_results.append(result)
            else:
                filtered_without_level.append(result)
        else:
            filtered_without_time.append(result)

    answer = ""
    
    if len(filtered_results):
        answer += "\nİşte senin için bulduğumuz eğitimler:\n"
        for i, result in enumerate(filtered_results):
            answer += f"{i+1}. {result['header']}\n\n{result['description']}\n\nUzunluk: {result['time']} dk\n\nSeviye: {result['level']}"
            answer += '\n' + '-'*80 + '\n'
    elif len(filtered_without_level):
        answer += "\nİstediklerini tam olarak karşılayan bir eğitim bulamadık ama farklı seviyelerdeki eğitimlerimizi deneyebilirsin:\n"
        for i, result in enumerate(filtered_without_level):
            answer += f"{i+1}. {result['header']}\n\n{result['description']}\n\nUzunluk: {result['time']} dk\n\nSeviye: {result['level']}"
            answer += '\n' + '-'*80 + '\n'
    elif len(filtered_without_time):
        answer += "\nİstediklerini tam olarak karşılayan bir eğitim bulamadık ama farklı uzunluklardaki eğitimlerimizi deneyebilirsin:\n"
        for i, result in enumerate(filtered_without_time):
            answer += f"{i+1}. {result['header']}\n\n{result['description']}\n\nUzunluk: {result['time']} dk\n\nSeviye: {result['level']}"
            answer += '\n' + '-'*80 + '\n'
    else:
        answer += "İsteklerine uygun bir eğitim bulamadık. Üzgünüz.\n"

    return answer
    
def get_results_from_db(
        str_to_embed: str,
        n: int = 4
    ) -> list[dict]:

    client = None
    collection = None

    client = chromadb.PersistentClient(path='./')

    if len(client.list_collections()):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            model_name="text-embedding-ada-002"
        )
        collection = client.get_collection(name="collection", embedding_function=openai_ef)
    else:
        collection = get_collection()

    result_dict = collection.query(
        query_texts=[str_to_embed],
        n_results=n
    )

    metadatas = result_dict["metadatas"]

    descriptions = result_dict["documents"]

    result = []

    for i in range(n):
        dict = {
            "header" : metadatas[0][i]["header"],
            "description" : descriptions[0][i],
            "time" : metadatas[0][i]["time"],
            "level" : metadatas[0][i]["level"]
        }

        result.append(dict)

    return result

def get_overall_prompt(
    need_response: str,
    time_response: str,
    level_response: str
    ) -> str:
    
    prompt = need_response + '\n' + time_response + '\n' + level_response

    return prompt
