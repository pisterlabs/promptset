from datetime import datetime

from langchain import OpenAI
from tqdm import tqdm
from requests.exceptions import ConnectionError

from ubix.common.log_basic import logging
#from route.router import route_meta, default_chain, router_chain
from route.router import default_chain, router_chain
import route.router as router
from langchain.utils.math import cosine_similarity
import os
import re
import pickle
import pdb
import numpy as np
import requests

def get_route_name(question):
    stop = None if isinstance(router_chain.llm, OpenAI) else ["\n"]
    route_name_original = router_chain.run(input=str(question), stop=stop)
    route_name = route_name_original.strip().split(':')[-1].strip()
    #route_name = route_name if route_name in route_meta else "other"
    print(f"route_name:{route_name_original}=>{route_name}")
    return route_name

def get_answer(question: str, history, query_type="auto"):
    start_route = datetime.now()
    answer = ""
    try:
        start = datetime.now()
        route_meta = router.get_route_meta()

        route_name = get_route_name(question) if query_type == "auto" else query_type
        logging.info("***********************")
        logging.info(f"step one: choose the proper route {route_name}, current route:{list(route_meta.keys())}")
        duration_route = (start-start_route).total_seconds()
        logging.info(f'>>>: Begin ask <{route_name}> about question: <{question}>, route cost:{duration_route} sec')

        if route_name in route_meta:
            if "query" in route_name:
                chain_list = route_meta[route_name]
                answer = {}
                sql_dct = {}
                answer_dct = {}
                answer_chain_sap = chain_list[0].run(question)
                sql_dct["sap"] = answer_chain_sap["sql"]
                answer_dct["sap"] = answer_chain_sap["answer"]

                answer_chain_bone = chain_list[1].run(question)
                sql_dct["bone"] = answer_chain_bone["sql"]
                answer_dct["bone"] = answer_chain_bone["answer"]

                answer["sql"] = sql_dct
                answer["answer"] = answer_dct
            else:
                answer = route_meta[route_name].run(question)
                if "other" in route_name:
                    try:
                        if answer.startswith("?\n"):
                            answer = answer.split("\n")[1]
                    except:
                        pass
        else:
            answer = default_chain.run(question)
    except ConnectionError as ex:
        answer = f"❌:Connection issue:{type(ex)},{str(ex)}"
        route_name = "error"
    #print("step three: the final result is", answer, "\n")
    end = datetime.now()
    duration = (end-start).total_seconds()
    #logging.info(f'🔴: End ask route:<{route_name}> about question <{question}>, cost:{duration} sec, answer: {answer}')
    #logging.info("⏰" + f" Route:{route_name} cost:{duration_route:.0f} seconds, Query: cost:{duration:.0f}sec seconds")
    return route_name, answer


if __name__ == '__main__':
    import os
    for _ in tqdm(range(1), desc="Answer testing"):
        if os.environ.get("LLM_TYPE", None) == "din":
            question_list = [
                ("What is machine learning", "auto"),
                ("what are our revenues by product for the past 13 years", "auto"),
                ]
            for question, query_type in question_list:
                print("question: ", question)
                get_answer(question, None, query_type=query_type)
        else:
            question_list = [
                    ("What is black body radiation", "auto"),
                    ("what is the maximum total in this table?", "query"),
                    ("how many records are there in this table?", "query"),
                    ]
            for question, query_type in question_list:
                get_answer(question, None, query_type=query_type)
"""
 
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. LLM_TYPE=gglm python ubix/common/answer.py

RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=. LLM_TYPE=vllm python ubix/common/answer.py
PYTHONPATH=. LLM_TYPE=gglm python ubix/common/answer.py

✅ PYTHONPATH=. LLM_TYPE=tgi python -u ubix/common/answer.py
✅ PYTHONPATH=. LLM_TYPE=din python -u ubix/common/answer.py


"""

