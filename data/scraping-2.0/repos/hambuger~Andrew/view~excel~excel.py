import openai
from flask import request, render_template, Blueprint, session
import pandas as pd
from openai import ChatCompletion, OpenAIError
import uuid
from config.global_logger import logger
from database_util.es.es import es, bulk_insert
import json
import numpy as np
import math
import threading
import time
from database_util.redis.redis_client import api_key_manager
from openai_util.prompt import get_excel_2_es_result_prompt, get_excel_2_es_mapping_prompt_v2

# 存储上传的Excel文件的数据
dataframe = None

excel_route = Blueprint('excel', __name__)


def delete_indices_with_prefix(prefix):
    all_indices = es.cat.indices(index='*', h='index', s='index', format='json')
    indices_to_delete = [index_info['index'] for index_info in all_indices if index_info['index'].startswith(prefix)]
    for index in indices_to_delete:
        response = es.indices.delete(index=index)
        if response["acknowledged"]:
            logger.debug(f"Index '{index}' deleted successfully.")
        else:
            logger.debug(f"Failed to delete index '{index}'.")


def deal(sid):
    if api_key_manager.get_key_value(sid) and time.time() - float(api_key_manager.get_key_value(sid)) < 60 * 5:
        task = threading.Timer(60 * 5, deal, args=[sid])
        task.start()
        return
    delete_indices_with_prefix(sid)


@excel_route.route('/excel')
def home():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return render_template('excel.html')


@excel_route.route('/excel/upload', methods=['POST'])
def upload_file():
    global dataframe
    session_id = session['sid']
    session.modified = True
    file_name = request.files['file'].filename.replace('.xlsx', '').replace('.xls', '')
    session['file_name'] = file_name
    session.modified = True
    print(session_id)
    if request.method == 'POST':
        try:
            file = request.files['file']  # 获取上传的文件对象
            # 处理上传的 Excel 文件数据
            if file and file.filename.endswith('.xls') or file.filename.endswith('.xlsx'):
                # 执行相关的 Excel 数据处理逻辑
                # 例如，使用 pandas 读取 Excel 文件数据
                dataframe = pd.read_excel(file, header=None)
                # 创建一个空 list 来保存数据
                data_list = []
                dataframe = dataframe.fillna(0)
                # 遍历 dataframe 的每一行
                # 取第一行数据
                field_names = dataframe.head(1).values.tolist()[0]
                for index, row in dataframe.iterrows():
                    # 将每一行的数据转换为 list 并添加到 data_list
                    data_list.append(row.tolist())
                try:
                    index_name = session_id + '_' + file_name
                    # 如果es不存在索引，创建索引
                    if not es.indices.exists(index=index_name):
                        # 调用chatgpt3.5模型，传入对话列表
                        message = get_excel_2_es_mapping_prompt_v2(data_list[:10])
                        logger.debug("message{}".format(message))
                        openai.api_key = api_key_manager.get_openai_key()
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": message}],
                            temperature=0,
                        )
                        content = response['choices'][0]['message']['content']
                        logger.debug("content{}".format(content))
                        mapping = json.loads(content)
                        logger.debug("mapping{}".format(mapping))
                        es.indices.create(index=index_name, body=mapping)
                        session['mappings'] = mapping
                        session.modified = True
                        task = threading.Timer(60 * 5, deal, args=(session_id,))
                        task.start()
                    actions = []
                    data_list = data_list[1:]
                    for index, item in enumerate(data_list):
                        processed_item = []
                        for y, value in enumerate(item):
                            if isinstance(value, float) and math.isnan(value):
                                processed_item.append(np.nan_to_num(value))
                            elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                                try:
                                    processed_item.append(json.loads(value))
                                except (ValueError, SyntaxError):
                                    logger.debug("无法将 {} 解析为列表，将保存原数据".format(value))
                                    processed_item.append(value)
                            else:
                                processed_item.append(value)

                        data_dict = {field_names[i]: processed_item[i] for i in range(len(field_names))}

                        actions.append({
                            "_index": index_name,
                            "_source": data_dict
                        })
                        if index % 100 == 0:
                            bulk_insert(actions)
                            actions = []
                    # 批量插入剩余的数据
                    if len(actions) > 0:
                        bulk_insert(actions)
                    return "success"
                except OpenAIError as e:
                    logger.debug(e)
                    return "failed"
                except Exception as e:
                    logger.debug(e)
                    return "failed"
            else:
                return 'Invalid file format. Please upload an Excel file.'
        except Exception as e:
            logger.debug(e)
            return 'Error processing file.'
    else:
        return 'Invalid request method.'


@excel_route.route('/excel/chat', methods=['POST'])
def chat():
    # 获取用户的输入
    api_key_manager.update_key_value(session['sid'], time.time())
    message = request.form['message']
    logger.debug(message)
    if not session.get('file_name') or not session.get('mappings'):
        return "请先上传数据或者请等待数据处理完成"
    # 使用 GPT-3 生成回应
    mappings = session['mappings']
    openai.api_key = api_key_manager.get_openai_key()
    response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": get_excel_2_es_result_prompt(mappings, message)}
        ],
        temperature=0,
    )
    # 返回 GPT-3 的回应
    dslStr = response['choices'][0]['message']['content']
    query = json.loads(dslStr)
    logger.debug("query{}".format(query))
    indexList = es.search(index=session['sid'] + '_' + session['file_name'], body=query)
    sourceList = []
    if indexList['hits']['total']['value'] == 0:
        return "没有找到相关数据"
    for hit in indexList['hits']['hits']:
        sourceList.append(hit['_source'])
    result = '<br>'
    for idx, source in enumerate(sourceList, 1):
        result += f"{idx}. "
        for key, value in source.items():
            result += f"{key}:{value} "
        result += '<br>'
    return result
