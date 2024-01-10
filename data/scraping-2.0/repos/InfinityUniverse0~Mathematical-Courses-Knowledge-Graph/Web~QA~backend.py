'''backend.py
    This file contains the backend code for the web application.
    复杂的后端处理函数代码
    被views.py调用
'''

import openai
from django.conf import settings
openai.api_key = settings.OPENAI_API_KEY

# 导入neo4j_db/course_graph.py 中的 CourseGraph类
import sys
sys.path.append("..")
from neo4j_db.course_graph import CourseGraph

# 和AI交互的接口，输入格式：[message1, message2, ...]
def chat(message_list: list, role: str):
    try:
        messages = [{"role": "system", "content": '你是一个' + role}]
        for msg in message_list:
            messages.append({"role": "user", "content": msg})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=messages,
        )
        completion = response['choices'][0]['message']['content']
    except openai.error.InvalidRequestError: # 超出Token上限或模型参数错误
        completion = "AI内部错误, 请重试!"
    except openai.error.RateLimitError: # 超出3条/min的请求限制
        completion = '系统繁忙，请稍等片刻再进行提问~~'
    return completion

# Node查询的cursor转化为list节点集
def nodes_to_list(cursor_node, table=False):
    '''将py2neo.Node对象转换为点集'''
    node_list = []
    cursor = cursor_node
    for node in cursor:
        node = node['n']
        if node is None:
            continue
        node_dict = {
            'name': node['name'],
            'label': list(node.labels)[0],
        }
        if table:
            node_dict.update({
                'refer': node['references'],
                'intro': node['intro']
            })
        node_list.append(node_dict)
    return node_list

# Path查询的cursor转化为list边集
def paths_to_list(cursor_path):
    '''将py2neo.Path对象转换为边集'''
    path_list, node_list = [], []
    cursor = cursor_path
    for p in cursor:
        if p['p'] is None:
            continue
        path = p['p']
        start_node, end_node = path.start_node, path.end_node
        node_list.append({'n': start_node})
        node_list.append({'n': end_node})
        rel = list(path.types())[0]
        path_dict = {
            'source': start_node['name'],
            'target': end_node['name'],
            'relation': rel,
        }
        path_list.append(path_dict)
    return path_list, nodes_to_list(node_list)

# 节点集去重
def unique_nodes(node_list):
    unique_list = []
    for node in node_list:
        if node not in unique_list:
            unique_list.append(node)
    return unique_list

# 边集去重
def unique_paths(path_list):
    unique_list = []
    edges = []
    for path in path_list:
        edge = (path['source'], path['target'], path['relation'])
        if edge not in edges:
            unique_list.append(path)
            edges.append(edge)
    return unique_list
