import pickle
import redis
import pprint

# Get the pickled object
r = redis.Redis(host='localhost', port=6379, db=0)
pickled_object = r.get('eff3e567-9008-4fa6-9569-89cdce3ac2cb')

# Unpickle the object
res = pickle.loads(pickled_object)

# Create a pprint instance and print the object
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(res)
question = "どのような人材を目指していますか"
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
from llama_index.evaluation import QueryResponseEvaluato

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
evaluator = QueryResponseEvaluator(service_context=service_context)
eval_result = evaluator.evaluate_source_nodes(response=res,query=question)
print(eval_result)