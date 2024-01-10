import os
from langchain.vectorstores import Chroma
import sys
from langchain.embeddings import FakeEmbeddings
import json


def add2json(user_id, file_name, begin_index, end_index):
    if os.path.exists('data.json'):
        # 读取JSON文件
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        if user_id in data:
            new_file = {
                'file_name': file_name,
                'begin_index': begin_index,
                'end_index': end_index
            }
            data[user_id].append(new_file)

            # 将修改后的数据写回文件
            with open('data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        else:
            data[user_id] = [{
                'file_name': file_name,
                'begin_index': begin_index,
                'end_index': end_index
            }]
            # 将修改后的数据写回文件
            with open('data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    else:
        data = {
            user_id: [{
                'file_name': file_name,
                'begin_index': begin_index,
                'end_index': end_index
            }]
        }

        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)


def delete_from_json(user_id, file_name):
    with open('data.json', 'r') as f:
        data = json.load(f)

    for i in range(0, len(data[user_id])):
        if data[user_id][i]['file_name'] == file_name:
            data[user_id].pop(i)
            break

    # 将更新后的 JSON 数据写回文件中
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def delete_from_vdb(user_id, file_name):
    begin_index = None
    end_index = None
    with open('data.json', 'r') as f:
        data = json.load(f)

    for i in range(0, len(data[user_id])):
        if data[user_id][i]['file_name'] == file_name:
            begin_index = data[user_id][i]['begin_index']
            end_index = data[user_id][i]['end_index']
            break

    embeddings = FakeEmbeddings(size=1352)
    persist_directory = os.path.join(sys.path[0], 'db')
    print(persist_directory)
    memorydb = Chroma(persist_directory=persist_directory,
                      embedding_function=embeddings, collection_name=user_id)
    # 删除选择文件数据
    ids = [file_name+str(i) for i in range(begin_index, end_index)]
    memorydb.delete(ids)
