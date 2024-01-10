# ! /usr/bin/python3.11
# ! -*- coding:UTF-8 -*-
from io import BytesIO

import pandas as pd
import pymysql
from flask import Blueprint, request, jsonify, send_file

from dataprocess.ConvertToExcel import convert_to_excel
from model.transformer import Transformer
from model.gpt import Gpt
from model.llama import Llama
from dataprocess.ImportToMysql import import_to_mysql
from openai import OpenAI
# from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import os

os.environ["OPENAI_API_KEY"] = "sk-uQzvGpP0SZmjBm8J918c590782Cc4e93A2715dC3286fD9C8"

# llama = Llama(model_path='../../../PycharmProjects/roberta-gat/llama-2-7b.Q4_K_M.gguf')

client = OpenAI(base_url="https://d2.xiamoai.top/v1")

sens_tan_md = SentenceTransformer('all-MiniLM-L6-v2')

roberta_gat = None

# 连接数据库
db = pymysql.connect(host='localhost', user='root', passwd='lish145210', port=3306, db='rdreaserch')

data_process_blueprint = Blueprint('data', __name__)

@data_process_blueprint.route('/api/embedding', methods=['POST'])
def get_embedding():
    # 获取POST请求中的参数
    model_name = request.form.get('model')
    sentences = request.form.get('sentences')
    # 选择模型
    if model_name == "GPT":
        return Gpt.embedding(client, sentences)
    elif model_name == "Llama2":
        return Llama.embedding(llama, sentences)
    elif model_name == "sentence-transformer":
        return Transformer.embedding_by_transformer(sens_tan_md, sentences)

@data_process_blueprint.route('/api/import_mysql', methods=['POST'])
def importToMysql():
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        file_content = file.read().decode('utf-8')
        paper = file_content.split("\nER\n")

        import_to_mysql(db, paper)
        return jsonify({'message': 'File successfully processed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# TODO 向neo4j中导入数据
@data_process_blueprint.route('/api/import_neo4j', methods=['POST'])
def import_neo4j():
    pass

# TODO 向milvus中导入数据
@data_process_blueprint.route('/api/import_milvus', methods=['POST'])
def import_milvus():
    pass

# TODO 摘要序列标注
@data_process_blueprint.route('/api/abstract_segment', methods=['Post'])
def abstract_segment():
    model_name = request.form.get('model')
    abstract = request.form.get('abstract')
    if model_name == "GPT":
        return Gpt.abstract_segmentation(client, abstract)
    elif model_name == "Llama":
        return Llama.abstract_segmentation(llama, abstract)
    elif model_name == "RobertaGAT":
        return Transformer.abstract_segmentation(roberta_gat, abstract)


# TODO ner
@data_process_blueprint.route('/api/ner', methods=['POST'])
def ner():
    text = request.form.get('text')
    model = request.form.get('model')

    if model == 'gpt':
        return Gpt.ner(client, text)
    if model == 'llama':
        return Llama.ner(llama, text)
    if model == 'bert':
        return Transformer.ner(text)

# TODO txt 转成格式化excel
@data_process_blueprint.route('/api/txt_to_excel', methods=['POST'])
def txt_to_excel():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        df = convert_to_excel(file)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)

        return send_file(output,
                         download_name='data.xlsx',
                         as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        # 如果出现任何异常，返回错误信息
        return jsonify({'error': str(e)}), 500

@data_process_blueprint.route('/api/question_answering', methods=['POST'])
def question_answering():
    model = request.form.get('model')
    question = request.form.get('question')
    if model == "GPT":
        return Gpt.question_answering(client, question)
    if model == "llama":
        return Llama.question_answering(llama, question)

@data_process_blueprint.route('/api/classification', methods=['POST'])
def classification():
    model = request.form.get('model')
    text = request.form.get('text')
    if model == "GPT":
        return Gpt.classification(client, text)
    if model == "llama":
        return Llama.classification(llama, text)
