import ast
import configparser
import json
import os
import openai
import pandas as pd
import pinecone
import time

from datetime import date, datetime
from openai.embeddings_utils import distances_from_embeddings
from tqdm.auto import tqdm

config = configparser.ConfigParser()
config.read('default.cfg')
d_conf = config['DEFAULT']

# Init openai param
openai.api_type = d_conf['api_type']
openai.api_base = d_conf['api_base'] 
openai.api_version = d_conf['api_version']

deployment_id = d_conf['fast_llm_model_deployment_id']
temperature = int(d_conf['temperature'])

# Define index name
index_name = 'research-assistant'

max_tokens = int(d_conf['max_tokens'])
max_len=6000

date_format_str= '%Y-%m-%d'
debug = False

def create_context(index, v_filter, question, max_len=max_len):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
            input=question, 
            deployment_id=d_conf['embedding_model_depleyment_id'],
            #engine='text-embedding-ada-002',
            )['data'][0]['embedding']

    # get relevant contexts (including the questions)
    response = index.query(q_embeddings, top_k=10, filter=v_filter, include_metadata=True)

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    #for i, row in df.sort_values('distances', ascending=True).iterrows():
    for item in response['matches']:

        # Add the length of the text to the current length
        cur_len += item['metadata']['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(item['metadata']['text'])

    # Return the context
    return '\n\n###\n\n'.join(returns)


def answer_question(
    index,
    v_filter,
    question="简要介绍一下轻联这个产品",
    max_len=max_len,
    debug=debug,
    max_tokens=max_tokens,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(index, v_filter, question, max_len=max_len)

    prompt = f"根据下面的上下文回答提问，上下文会包含多行内容，最后以---结束;如果不能根据上下文回答提问，请回答:\"没有找到相关信息\"\n\n上下文：\n{context}\n----\n\n提问：{question}\n回答："
    # If debug, print the raw model response
    if debug:
        print(f"\n---prompt start:\n{prompt}\n---prompt end\n\n")

    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
                deployment_id=deployment_id,
                messages=[{'role': 'system', 'content':prompt}],
                temperature=temperature,
                #max_tokens=max_tokens+max_len,
        )
        #print(response)
        return response.choices[0].message.content

    except Exception as e:
        print(e)
        return ""


def format_date_filter(json_to_load):
    try:
        brace_index = json_to_load.index("{")
        maybe_fixed_json = json_to_load[brace_index:]
        last_brace_index = maybe_fixed_json.rindex("}")
        maybe_fixed_json = maybe_fixed_json[: last_brace_index + 1]
        return json.loads(maybe_fixed_json)
    except (json.JSONDecodeError, ValueError) as e:
        print(e)
        return {}


def convert_date_format(date_string):
    date_obj = datetime.strptime(date_string, date_format_str)
    return int(time.mktime(date_obj.timetuple()))


def generate_date_filter(question, debug=debug):
    prompt = '如果提问中包含了检索日期范围等信息，则返回如下json格式：\n{"start_date": "检索开始日期", "end_date": "检索结束日期"}\n同时检索开始时间和检索结束时间转换类似:"2023-07-15"格式。今天的日期是：' + date.today().strftime(date_format_str) + '。\n如果提问中不包含检索日期范围，则返回：{}。\n需要特别注意的是：你只需要提取检索日期范围，并返回指定的json格式，不需要要对提问中的其他的内容做解答。\n\n提问：\n' + question + '\n\n回答：'
       
    # If debug, print the raw model response
    if debug:
        print(f"\n--prompt start:\n{prompt}\n---prompt end.\n\n")

    # Create a completions using the question and context
    response = openai.ChatCompletion.create(
                deployment_id=deployment_id,
                messages=[{'role': 'system', 'content':prompt}],
                temperature=temperature,
                )
    

    llm_filter = response.choices[0].message.content
    raw_filter = format_date_filter(llm_filter)

    v_filter = {}
    if len(raw_filter.get('start_date','')) > 0:
        gte_filter = {'p_date_ut': {'$gte': convert_date_format(raw_filter['start_date'])}}
        v_filter = gte_filter

    if len(raw_filter.get('end_date','')) > 0:
        lte_filter = {'p_date_ut': {'$lte': convert_date_format(raw_filter['end_date'])}}
        v_filter = {'$and': [gte_filter, lte_filter]}

    # If debug, print the raw model response
    if debug:
        print(f"llm generate date filter: {llm_filter}\n")
        print(f"json format date filter: {raw_filter}\n")
        print(f"\njson format for vector filter: {v_filter}\n")

    return v_filter


def main():
    # Initialize connection to Pinecone
    pinecone.init(api_key=d_conf['pinecone_api_key'], environment=d_conf['pinecone_api_env'])

    # Connect to the index and view index stats
    index = pinecone.Index(index_name)

    questions = [
        '腾讯轻联6月份发布了哪些产品功能特性，请用列表的方式进行总结，字数控制在200字以内。',
        '腾讯轻联最近一个月发布了哪些产品功能特性，请用列表的方式进行总结，字数控制在200字以内。',
        '腾讯轻联最近一个季度，新增了哪些客户案例，请列举出客户名称以及客户使用场景。']

    for q in questions:
        ask_question(index, q)


def ask_question(index, question):
    v_filter = generate_date_filter(question)

    response = answer_question(index, v_filter, question, debug=debug)
    print(f"\n提问：\n{question}")
    print(f"\n回答：\n{response}")


if __name__ == '__main__':
    main()

