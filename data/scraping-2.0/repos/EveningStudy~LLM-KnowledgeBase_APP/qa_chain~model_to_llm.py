import sys

sys.path.append("../llm")
from llm.spark_llm import Spark_LLM
from langchain.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key


def model_to_llm(model: str = None, temperature: float = 0.0, appid: str = None, api_key: str = None,
                 Spark_api_secret: str = None):
    """
    星火：model,temperature,appid,api_key,api_secret
    百度问心：model,temperature,api_key,api_secret
    智谱：model,temperature,api_key
    OpenAI：model,temperature,api_key
    """
    # if model in ["Spark-1.5", "Spark-2.0", "Spark-3.0"]:
    if api_key == None or appid == None and Spark_api_secret == None:
        api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
    llm = Spark_LLM(model=model, temperature=temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
    return llm
