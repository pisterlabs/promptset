import openai
import logging  # 加载日志功能
import csv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import pdfplumber
from tqdm import tqdm
from queue import Queue
import os
from datetime import datetime
import time
import json

# 配置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # 配置日志级别

script_dir = os.path.dirname(os.path.abspath(__file__))
api_key_file_path = os.path.join(script_dir, 'key.txt')
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")

def initialize_model(api_key, model_name="gpt-3.5-turbo", temperature=0.7):
    chat_model =ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=temperature)
    return chat_model
def format_model_response(model_response):
    formatted_questions = []
    for question in model_response:
        formatted_question = {
            'title': question.get('title', ''),
            'type': question.get('type', ''),
            'option_list': question.get('option_list', []),
            'answer': question.get('answer', ''),
            'explain': question.get('explain', ''),
            'del_flag': question.get('del_flag', '0'),
            'create_by': question.get('create_by', 'admin'),
            'create_time': question.get('create_time', ''),
            'update_by': question.get('update_by', 'admin'),
            'update_time': question.get('update_time', '')
        }
        formatted_questions.append(formatted_question)
    return formatted_questions

def extract_text_from_pdf(pdf_path):
    text_blocks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"开始提取 PDF 文件文本，共 {total_pages} 页。")
            for page in tqdm(pdf.pages, desc="提取进度"):
                text = page.extract_text()
                if text:  # 确保提取到的文本非空
                    text_blocks.append(text)
    except Exception as e:
        logger.error(f"处理 PDF 文件时出错: {e}")
        return []

    return text_blocks


def generate_prompt(text_block):
    formatted_prompt = (
        "提取文档中某一个段落的内容，理解它并将它设计成一个选择题，每题有三个选项（A、B、C）。请返回字符串并按照以下格式提供每个问题的信息：\n"
        "- title: 根据文档中的某个段落随机编写一道选择题，例如‘航线上的天气变化都包括什么？’注意这里写入的不是一个简短的标题而是一个引用自文档中某段原文改写而来的完整的问题，"
        "题目和答案之间的对应关系应当清洗明确\n"
        "- type: 题目类型，例如 '单选题'\n"
        "- option_list: 一个包含三个选项的列表，每个选项都是一个字典，包含 'content' 和 'id' 键。例如: "
        '[{"content":"选项A内容","id":"A"}, {"content":"选项B内容","id":"B"}, {"content":"选项C内容","id":"C"}]\n'
        "- answer: 正确答案的ID，例如 'A'\n"
        "- explain: 答案解释\n"
        "- del_flag: 删除标志，用于表示题目是否已被删除，例如 '0' 代表未删除\n"
        "- create_by: 题目创建者，例如 'admin'\n"
        "- create_time: 题目创建时间，格式为 'MM/DD/YYYY HH:MM:SS'\n"
        "- update_by: 题目最后更新者，例如 'admin'\n"
        "- update_time: 题目最后更新时间，格式为 'MM/DD/YYYY HH:MM:SS'\n\n"
        "这里是知识点内容：\n"
        f"{text_block}"
    )

    return formatted_prompt

def parse_model_response(response_str):
    question = {}
    for line in response_str.strip().split('\n'):
        key, value = line.split(':', 1)  # 分割键和值
        key = key.strip().lower().replace(" ", "_")  # 格式化键名
        value = value.strip()
        if key == 'option_list':
            value = json.loads(value)  # 将选项列表字符串转换为列表
        question[key] = value
    return question

def generate_questions_from_text_blocks(text_blocks, chat_model):
    questions_queue = Queue()
    for index, text_block in enumerate(text_blocks):
        prompt = generate_prompt(text_block)
        try:
            response = chat_model.invoke([HumanMessage(content=prompt)])
            logger.info(f"处理第 {index + 1}/{len(text_blocks)} 个文本块")
            logger.info(f"模型返回的内容: {response.content}")

            # 直接解析 JSON 字符串
            question = json.loads(response.content)
            questions_queue.put(question)
            logging.info(f"{(question)}成功保存")
            # 等待1秒后继续下一个循环
            time.sleep(1)
        except Exception as e:
            logger.error(f"生成第 {index + 1} 个文本块的问题时出错: {e}")

    return questions_queue





def save_to_csv(questions_queue, csv_file_path):
    fieldnames = ['title', 'type', 'option_list', 'answer', 'explain', 'del_flag', 'create_by', 'create_time',
                  'update_by', 'update_time']

    # 检查队列是否为空
    if questions_queue.empty():
        logger.info("问题队列为空，不创建 CSV 文件。")
        return

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while not questions_queue.empty():
            question = questions_queue.get()
            logging.info(question)
            # 可以在这里添加对 question 内容的额外检查
            writer.writerow(question)


def main():
    # 模型初始化
    model = initialize_model(api_key)

    # PDF 文本提取
    pdf_path = os.path.join(script_dir, "uploaded_files/运动驾驶员执照理论考试知识点试行（初级飞机）.pdf")
    text_blocks = extract_text_from_pdf(pdf_path)
    if not text_blocks:
        logger.info("未能成功提取 PDF 文件内容。")
        return

    # 生成问题
    questions_queue = generate_questions_from_text_blocks(text_blocks, model)

    # 确保保存目录存在
    embedding_files_dir = os.path.join(script_dir, "Embedding_Files")
    if not os.path.exists(embedding_files_dir):
        os.makedirs(embedding_files_dir)

    # 保存问题到 CSV 文件
    csv_file_path = os.path.join(embedding_files_dir, "questions.csv")
    save_to_csv(questions_queue, csv_file_path)

if __name__ == "__main__":
    main()
