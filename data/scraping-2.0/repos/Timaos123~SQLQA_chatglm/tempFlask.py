from flask import Flask
from flask_restful import Resource, Api, reqparse,request
import pandas as pd
import json
from flask import jsonify
import requests
import openai
import pandas as pd
import datetime
import random
import json
import argparse
import joblib
import traceback
import time
from elasticsearch import Elasticsearch
import numpy as np
# 导入所需的库
import jieba
import numpy as np
from gensim import corpora, models, similarities
import re
import sqlite3
import datetime
import time
import uuid as uuid
import zhipuai

app = Flask(__name__)
api = Api(app)
openai.api_key=""

# 初始化Elasticsearch
es = Elasticsearch(hosts=["http://127.0.0.1:9200"])

# 自调用
def flaskReq(reqUrl,data):
    url = "http://127.0.0.1:6006{}".format(reqUrl)

    payload = json.dumps(data)
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': '127.0.0.1:6006',
        'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.json()["data"]


@app.route("/chatglm/generate_content", methods=["POST"])
def generate_content_glm():

    try:
        data = json.loads(request.get_data(as_text=True))
    except Exception as e:
        data = "Input parameter is not in JSON format"
        return jsonify({"code": 500, "msg": f"request data is not json format: %s" % str(e)})
    
    prompt = data.get("prompt", None)
    application = data.get("application", "GPT")
    
    # 接口API KEY
    API_KEY = ""
    # 公钥
    PUBLIC_KEY = ""
    ability_type = "chatglm_qa_6b"
    # 引擎类型
    engine_type = "chatglm_6b"
    # 请求参数样例
    import uuid
    uuid = uuid.uuid1()
    request_task_no = str(uuid).replace("-", "")

    zhipuai.api_key = API_KEY
    response = zhipuai.model_api.async_invoke(
        model="chatglm_pro",
        prompt=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.55,
        top_p=0.7,
        incremental=True 
    )
    taskID=response["data"]["task_id"]
    zhipuai.api_key = API_KEY
    while True:
        response = zhipuai.model_api.query_async_invoke_result(taskID)
        print(response)
        try:
            if response["data"]["task_status"]=="SUCCESS":
                break
        except:
            return {"code": 200, "detail": "", "data": response["msg"]}
        time.sleep(2)

    print(response)
    # else:
    #     print("获取token失败，请检查 API_KEY 和 PUBLIC_KEY")
    return {"code": 200, "detail": "", "data": response["data"]["choices"][0]["content"]}

from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer
from transformers import AutoModel,AutoTokenizer
import http.client
import json
import requests
import torch
import pandas as pd
from new_A0_try_input_Text2SQL import insert_into_es
import re
import traceback

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_text(bert_model, bert_tokenizer, text):
    inputs = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

def search_similar_vectors(query_text, indexName="sqlqa_index", 
                                        simVal="content",
                                        simVec="content_vector",
                                        simType="column",top_k=5):
    # 返回近似字段
    # simType: column/value
    
    model = AutoModel.from_pretrained("./dependent_service/models--junnyu--roformer_chinese_sim_char_base")
    tokenizer = AutoTokenizer.from_pretrained("./dependent_service/models--junnyu--roformer_chinese_sim_char_base",trust_remote_code=True)
#     query_vector=encode_text(model, tokenizer, query_text)
    input_ids = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True)['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
    query_vector=outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        
    if simType=="column":
        body = {
                    "size": 5,
                    "query": {
                        "function_score": {
                            "query": {"match_all": {}},
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, doc['col_name_vec']) + \
                                                cosineSimilarity(params.query_vector, doc['value_vec'])+\
                                                1.0",
                                    "params": {"query_vector": query_vector.flatten().tolist()}
                                }
                            }
                        }
                    },
                    "_source": ["col_name", "value"],  # 返回指定字段
                    "aggs": {
                        "deduplicate": {
                            "terms": {
                                "field": "col_name",
                                "size":5
                            },
                            "aggs": {
                                "top_hits": {
                                    "top_hits": {
                                        "size": 1  # 每个分组返回的文档数量，这里设为1代表只选择每组中的第一个文档
                                    }
                                }
                            }
                        }
                    }
                }
    else:
        body = {
                    "size": 5,
                    "query": {
                        "function_score": {
                            "query": {"match_all": {}},
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, doc['col_name_vec']) + \
                                                cosineSimilarity(params.query_vector, doc['value_vec'])+\
                                                1.0",
                                    "params": {"query_vector": query_vector.flatten().tolist()}
                                }
                            }
                        }
                    },
                    "_source": ["col_name", "value"],  # 返回指定字段
                    "aggs": {
                        "deduplicate": {
                            "terms": {
                                "field": "value",
                                "size":5
                            },
                            "aggs": {
                                "top_hits": {
                                    "top_hits": {
                                        "size": 1  # 每个分组返回的文档数量，这里设为1代表只选择每组中的第一个文档
                                    }
                                }
                            }
                        }
                    }
                }
    
    buckets = es.search(index=indexName, body=body)
    for bucketItem in buckets["aggregations"]["deduplicate"]["buckets"]:
        print("bucketItem:",[{"col_name":row["_source"]["col_name"],"value":row["_source"]["value"]} for row in bucketItem["top_hits"]["hits"]["hits"]])
    response=[row["_source"] for bucketItem in buckets["aggregations"]["deduplicate"]["buckets"]
                              for row in bucketItem["top_hits"]["hits"]["hits"]]
    response=[{"col_name":row["col_name"],"value":row["value"]} for row in response]
    print(simType,json.dumps(response,indent=4,ensure_ascii=False))
#     print(simType,response)# [0]["_source"]
    return response


def search_similar_text(query_text, 
                        indexName="sqlqa_index", 
                        simVal="col_name",
                        simType="column", top_k=5):
    # simType: column/value
    if simType=="column":
        body = {
            "size": 0,
            "aggs": {
                "deduplicated_values": {
                    "terms": {
                        "field": simVal,
                        "size": top_k
                    },
                    "aggs": {
                        "top_hits": {
                            "top_hits": {
                                "size": 1  # 每个分组返回的文档数量，这里设为1代表只选择每组中的第一个文档
                            }
                        }
                    }
                }
            },
            "query": {
                "match": {
                    simVal: {
                        "query": query_text,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
        # # print(body)
        # response = es.search(index=indexName, body=body)["aggs"]["hits"]["hits"]
        # response=[row["_source"] for row in response]
    else:
        body = {
            "size": 0,
            "aggs": {
                "deduplicated_values": {
                    "terms": {
                        "field": simVal,
                        "size": top_k
                    },
                    "aggs": {
                        "top_hits": {
                            "top_hits": {
                                "size": 1  # 每个分组返回的文档数量，这里设为1代表只选择每组中的第一个文档
                            }
                        }
                    }
                }
            },
            "query": {
                "match": {
                    simVal: {
                        "query": query_text,
                        "fuzziness": "2",
                        "prefix_length": 1,
                        "max_expansions": 50
                    }
                }
            }
        }
    buckets = es.search(index=indexName, body=body)["aggregations"]["deduplicated_values"]["buckets"]
    response=[]
    for bucketItem in buckets:
        response+=bucketItem["top_hits"]["hits"]["hits"]
    response=[row["_source"] for row in response]
    return response

def get_related_columns(user_text,indexName="sqlqa_index"):
    # 基于文本的向量近似度和关键词近似度处理逻辑
    
    # 近似列
    colList1=search_similar_vectors(user_text, 
                                    indexName=indexName,
                                    simType="column",
                                    simVec="col_name_vec",
                                    simVal="col_name", top_k=3)
    colList2=search_similar_text(user_text, 
                                    indexName=indexName, 
                                    simType="column",
                                    simVal="col_name",
                                    top_k=3)
    colList=colList1+colList2
    print("colList:",colList1,colList2)
    colList=[colItem["col_name"] for colItem in colList]
    colList=list(set(colList))
    
    return colList

def get_related_values(user_text,indexName="sqlqa_index"):
    # 在ES中查询与单引号内最相近的5个值的逻辑
    # 近似值
    valList1=search_similar_vectors(user_text, 
                                    indexName=indexName,
                                    simType="value",
                                    simVec="value_vec",
                                    simVal="value", top_k=3)
    valList2=search_similar_text(user_text, 
                                    indexName=indexName,
                                    simVal="value",
                                    simType="value",
                                    top_k=3)
    valList=valList1+valList2
    valList=[(row["col_name"],row["value"]) for row in valList]
    valList=list(set(map(lambda row:str(row),valList)))
    valList=list(map(lambda row:eval(row),valList))
    valList=list(filter(lambda row:row[1]!="num",valList))
    valDict=dict(valList)
    
    return valDict

def retrieve_sample_table(myDf,relevant_fields,sampleN=5):
    # 从数据库中获取样本表格的逻辑
    real_relevant_fields=filter(lambda colItem:colItem in myDf.columns,relevant_fields)
    return myDf.loc[:,real_relevant_fields].sample(min(sampleN,myDf.shape[0]))

def chatGLM(prompt):
    import requests
    import json

#     url = "http://127.0.0.1:6006/beauty_industry_doc_qa"
    url = "http://127.0.0.1:6520/chatglm/generate_content"

    payload = json.dumps({
        "prompt":prompt
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': '127.0.0.1:6520',
        'Connection': 'keep-alive'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

#     print(response.text)
    
    return response.json()["data"]

def construct_base_sql(user_text,sample_table,table_name="school_df",falseQueryList=[]):
    # 基于样本表格构建基础的SQL的逻辑
    prompt="我们拥有如下数据：\n {}:\n{}\n".format(table_name,sample_table.to_markdown()) + \
            "请根据以上数据以及用户问题：'{}'\n".format(user_text)+ \
            "构建SQL解答用户问题。\n"+\
            "构建SQL的时候注意使用和返回原表的字段（{}），不要使用*，答案只需要sql，使用GROUP BY的时候注意你的聚合函数和使用的字段是什么，不需要别的解释，请用以下格式生成SQL：\n".format(" 或 ".join(map(lambda colItem:"`{}`".format(colItem),list(sample_table.columns))))+\
            "生成的SQL为：```你生成的SQL```\n"+\
            ("你之前生成过如下错误SQL：\n"+";\n".join(falseQueryList)+"\n请换一种思路写你的SQL\n" if len(falseQueryList)>0 else "")+"\n注意原表内容\n"+\
            "用户问题是：'{}'\n".format(user_text)+ \
            "生成的SQL为："
    print("prompt:",prompt)
    SQLResult=chatGLM(prompt)
    return SQLResult

def check_quotes(base_sql):
    if "'" in base_sql:
        return True
    else:
        return False

def construct_new_table(sample_table, attr_val_dict):
    # 构建新的样例表格的逻辑
    print("attr_val_dict:",attr_val_dict)
    matchEvalStr="|".join(["(sample_table['{}']=='{}')".format(k,v) 
                                   for k,v in attr_val_dict.items()
                                      if k in sample_table.columns])
    print("matchEvalStr:",matchEvalStr)
    if len(matchEvalStr)>0:
        new_table=sample_table.loc[eval(matchEvalStr),:]
    else:
        new_table=sample_table
    return new_table

def reconstruct_sql(base_sql, user_text):
    # 重构SQL，结合用户输入的文本的逻辑
    # ...
    return reconstructed_sql

import duckdb
def execute_sql(reconstructed_sql,myDf,table_name="school_df"):
    # 执行SQL检索的逻辑
    # df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})

    con = duckdb.connect()
    con.register(table_name, myDf)

    result = con.execute(reconstructed_sql.replace("\\n",""))
    df_result = result.fetchdf()
    
    return df_result

def generate_answer(user_text, newSQL,sample_table,query_result,table_name="school_df"):
    # 结合用户输入的问题和SQL检索结果生成回答的逻辑
    prompt="我们拥有如下数据：\n {}:\n{}\n".format(table_name,sample_table.to_markdown()) + \
            "请根据以上数据以及用户问题：'{}'\n".format(user_text)+ \
            "构建SQL解答用户问题。\n"+\
            "构建SQL的时候注意使用和返回原表的字段（{}），注意原表内容，不要使用*，答案只需要sql，使用GROUP BY的时候注意你的聚合函数和使用的字段是什么，不需要别的解释，请用以下格式生成SQL：\n".format(" 或 ".join(map(lambda colItem:"`{}`".format(colItem),list(sample_table.columns))))+\
            "生成的SQL为：```你生成的SQL```\n"+\
            "用户问题是：'{}'\n".format(user_text)+ \
            "生成的SQL为：```{}```\n".format(newSQL)+\
            "所得结果为：\n{}".format(query_result.to_markdown())+\
            "根据以上数据，通过自然语言的方式，回答用户问题'{}'，回答格式如下：\n".format(user_text)+\
            "你的回答是：```你的答案```\n"+\
            "你的回答是："
    answer=chatGLM(prompt)
    return answer

from fuzzywuzzy import process
def find_most_similar_string(str1, string_list):
    return process.extractOne(str1, string_list)[0]

@app.route("/chatglm/sqlqa",methods=["post"])
def sqlqa():
    # 用户输入文本
    data = json.loads(request.get_data(as_text=True))
    prompt = data.get("prompt", None)
    
    user_text=prompt
    
    # 示例DataFrame
    myDf = pd.DataFrame(
        [["张三","清华大学","北京"],
        ["张四","清华大学","南京"],
        ["张五","清华大学","赣州"],
        ["张六","北京大学","北京"],
        ["张七","北京大学","南京"],
        ["张八","对外经济贸易大学","赣州"],
        ["张九","河海大学","北京"]],columns=["student_name","school_name","district"])
    indexName = "try_student_school_index"  # 设置Elasticsearch索引的名称
    id_col_name = "student_name"  # 用户提供的id列名
    table_name="school_df"
    
#     insert_into_es(myDf,es,indexName,id_col_name)

    # 处理输入文本，获取相关字段
    relevant_fields = get_related_columns(user_text,indexName=indexName)

    # 从数据库中获取样本表格
    sample_table = retrieve_sample_table(myDf,relevant_fields)

    # 构建基础的SQL查询语句
    falseQueryList=[]
    while True:
        if len(falseQueryList)>3:
            break
        try:
            base_sql = construct_base_sql(user_text,sample_table,table_name=table_name,falseQueryList=falseQueryList)
            base_sql=base_sql.replace("\"","").replace("`","").replace("，",",")

            if "SELECT" in base_sql:
                base_sql=base_sql[base_sql.index("SELECT"):]
            elif "select" in base_sql:
                base_sql=base_sql[base_sql.index("select"):]
                
            # 判断SQL中是否存在单引号
            if check_quotes(base_sql):
                # 在ES中查询与单引号内最相近的5个值
                attr_val_dict = get_related_values(user_text,indexName=indexName)

                # 构建新的样例表格
                new_table = construct_new_table(sample_table, attr_val_dict)
                
                # 重构SQL查询语句，结合用户输入的文本
                sqlKVList=re.findall("\S*\s+=\s+'.*?'",base_sql)
                for sqlKVItem in sqlKVList:
                    k,v=sqlKVItem.split("=")
                    k=k.strip()
                    if "." in k:
                        k=k.split(".")[1]
                    v=v.replace("'","").strip()
                    newVList=new_table[k].values.flatten().tolist()
                    newV=find_most_similar_string(v,newVList)
                    base_sql=base_sql.replace(v,newV)

                reconstructed_sql = base_sql
            else:
                new_table = sample_table
                reconstructed_sql = base_sql
                
            print("table:",new_table)
            print("SQLQuery:",reconstructed_sql)
            # 执行SQL查询
            query_result = execute_sql(reconstructed_sql,myDf,table_name=table_name)
            break
        except duckdb.BinderException as de:
            traceback.print_exc()
            falseQueryList.append(reconstructed_sql)
            falseQueryList=list(set(falseQueryList))
        except duckdb.ParserException as dp:
            traceback.print_exc()
            falseQueryList.append(reconstructed_sql)
            falseQueryList=list(set(falseQueryList))
        except:
            traceback.print_exc()
            

    # 生成回答
    answer = generate_answer(user_text,reconstructed_sql, sample_table,query_result,table_name=table_name)

    # 输出结果
    if "你的回答是：" in answer:
        answer=answer.split("你的回答是：")[1]
        answer=answer.replace("\"","")
        
    print("回答:", answer)
        
    return {
        "data":{
            "SQL":reconstructed_sql,
            "table":query_result.to_markdown(),
            "answer":answer
        },
        "status":200
    }

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=6520,debug=False)