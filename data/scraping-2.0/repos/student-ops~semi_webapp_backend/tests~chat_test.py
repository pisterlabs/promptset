import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from langchain import OpenAI
from src.llama import utils
from pprint import pprint 
from src.llama import chat
import redis
import pickle
import uuid
import pdb
from llama_index.evaluation import QueryResponseEvaluator

r = redis.Redis(host='localhost', port=6379, db=0)

def LlamaChat(bot_name, question,uuid):
    model_name = "text-davinci-003"
    model_temperature = 0

    api_key = os.getenv("OPENAI_API_KEY")

    index = chat.InitIndex(bot_name=bot_name)

    print(index)
    llm = utils.get_llm(model_name, model_temperature, api_key)
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=llm))
    if (index == None):
        yield f"BOT not found eorror"
        return

    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=2,
        service_context=service_context
    )
    response = query_engine.query(question)
    response_txt = ""
    for text in response.response_gen:
        response_txt += text
        yield text
    res = response.get_response()
    res.response = response_txt
    pprint(res)
    r.set(uuid, pickle.dumps(res))

    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",streaming=True))
    # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    # evaluator = QueryResponseEvaluator(service_context=service_context)
    # eval_result = evaluator.evaluate_source_nodes(response=res,query=question)
    # print(eval_result)
    
    

import pdb  # デバッガーのインポート

if __name__ == "__main__":
    question = "どのような人材を目指していますか"
    question += " 日本語で回答して"
    bot_name = "faculty"
    try:
        response_gen = LlamaChat(bot_name, question)
        if response_gen is not None:
            for text in response_gen:
                print(text)
        else:
            print("No response received.")

    except Exception as e:
        print("An error occurred: ", str(e))
    # InitIndex("faculty")
