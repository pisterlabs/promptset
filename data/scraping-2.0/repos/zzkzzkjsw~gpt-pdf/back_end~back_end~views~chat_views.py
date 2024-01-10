from flask import Blueprint, request, redirect, session
from ..chatgpt_api.user_input import handle_user_input, handle_user_file_input
from flask_login import login_required
import json
from PyPDF2 import PdfReader
from typing import List
from pathlib import Path
import os

chat_views = Blueprint('chat_views',__name__)

@chat_views.route("/user/<chat_uuid>",methods=['GET'])
def hello_world(chat_uuid):
    
    # 返回json数据的方法
    data = {
        "name":"zhangsan",
        "age":18
    }
    # 第一种
    response = json.dumps(data)  # 将python的字典转换为json字符串
    return response,200,{"Content-Type":"application/json"}

@login_required
@chat_views.route("/chat-process",methods=['POST'])
def handle_chat():
    # # 返回json数据的方法
    # data = {
    #     "role":"assistant",
    #     "text":"你好啊我是4002",
    #     "id":"789",
    #     "parentMessageId":"666"
    # }
    
    request_data = request.get_json()
    folder = 'cache/files'


    if os.path.exists(folder+'/result.json'):
        print('开始处理文件回答')
        with open(folder+'/result.json', 'r', encoding='UTF-8') as f: 
            obj=json.load(f)
            file_text_embedding_list = obj['file_text_embedding_list']
            file_text_list = obj['file_text_list']
        response_data = handle_user_file_input(request_data,file_text_embedding_list,file_text_list)
    else:
        print('开始处理正常回答')
        response_data = handle_user_input(request_data)

    response = json.dumps(response_data)  # 将python的字典转换为json字符串
    return response,200,{"Content-Type":"application/json"}

@chat_views.route("/file",methods=['POST'])
def handle_file():
    # # 返回json数据的方法
    # data = {
    #     "role":"assistant",
    #     "text":"你好啊我是4002",
    #     "id":"789",
    #     "parentMessageId":"666"
    # }
    # "multipart/form-data"
    # uuid = request.form.get('uuid')  # 获取表单uuid字段
    print('我已收到文件')
    file = request.files['file']
    file_text = extract_text_from_file(file)

    folder = 'cache/files'
    Path(folder).mkdir(parents=True, exist_ok=True)


    file_text_list = chunks(file_text)

    file_text_embedding_list = get_embeddings(file_text_list)
    print('我正在处理文件')
    with open(folder+'/result.json', 'w',encoding='utf-8') as handle2:
        json.dump({"file_text_list":file_text_list,"file_text_embedding_list":file_text_embedding_list}, handle2, ensure_ascii=False, indent=4)

    response_data = {}
    response_data['detail'] = ''
    response_data['role'] = 'assistant'
    response_data['id'] = '1'
    response_data['parentMessageID'] = ''
    response_data['text'] = '我已收到文件！'

    response = json.dumps(response_data)  # 将python的字典转换为json字符串
    return response,200,{"Content-Type":"application/json"}

# Extract text from a file based on its mimetype
def extract_text_from_file(file):
    """Return the text content of a file."""
    if file.mimetype == "application/pdf":
        # Extract text from pdf using PyPDF2
        reader = PdfReader(file)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text()
    elif file.mimetype == "text/plain":
        # Read text from plain text file
        extracted_text = file.read().decode("utf-8")
        file.close()
    # elif file.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    #     # Extract text from docx using docx2txt
    #     extracted_text = docx2txt.process(file)
    else:
        # Unsupported file type
        raise ValueError("Unsupported file type: {}".format(file.mimetype))

    return extracted_text

def chunks(text:str) -> List[str]:

    WORDS_MAX_LENGTH = 200
    words = text.split()
    text_list = []
    start = 0
    while start < len(words):
        end = start + WORDS_MAX_LENGTH
        chunk = " ".join(words[start:end])
        text_list.append(chunk)
        start = end
    return text_list

def get_embedding(text):
    import openai
    # 设置 API Key，申请地址：https://platform.openai.com/account/api-keys
    openai.api_key = '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' 
    # 设置组织，查看地址：https://platform.openai.com/account/org-settings
    openai.organization = "org-jy2eL6AhJYOSnDydGNQpPcro"
    EMBEDDING_DIM = 1536
    model = "text-embedding-ada-002"
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result['data'][0]['embedding']

def get_embeddings(file_text_list:List)->List[List[float]]:
    embedding_list = []
    for segment in file_text_list:
        result = get_embedding(segment)
        embedding_list.append(result)
    return embedding_list

