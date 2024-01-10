
import os
import openai
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from llm_utils.read_config import load_config


config = load_config() 
os.environ["https_proxy"] = "http://localhost:7890"
 
def llm_quary(csv_path, prompt):
    #os.environ["https_proxy"] = "https://localhost:7890"
    openai.api_base = config['openai_base']
    openai.api_key = config['openai_api_key']
    print(csv_path) 
    df = pd.read_csv(csv_path, encoding='gbk')
    llm = OpenAI(api_token=openai.api_key)

    pandas_ai = PandasAI(llm)
    response = pandas_ai.run(df, prompt='请回答：'+prompt+'。回答的同时请判断是否需要生成图片或者表格，如有必要请生成图片以及处理之后的表格，将图片和表格保存至result_data目录下')
    print(response)
    return response
