import os
import openai
import logging

from flask import Flask
from gpt_format import LlamaFormat, LlamaChatFormat, GPTJFormat, GPTFormat
from utils import load_model, load_reference
from flask_wtf.csrf import CSRFProtect

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')


ref_data = load_reference(os.getenv('REF_DATA'))
formaters = {
    'llama': LlamaFormat,
    'llama-chat': LlamaChatFormat,
    'gpt-j': GPTJFormat,
    'custom': GPTFormat
}

def init_log():
    logger = logging.getLogger('chat_logger')
    logger.setLevel(logging.DEBUG)

    # 創建一個文件處理器，並設置日誌級別為DEBUG
    file_handler = logging.FileHandler('/chat_log/chat.log')
    file_handler.setLevel(logging.INFO)

    # 創建一個格式化器，以設置日誌消息的格式
    formatter = logging.Formatter('"%(asctime)s", %(message)s')
    file_handler.setFormatter(formatter)

    # 將文件處理器添加到Logger中
    logger.addHandler(file_handler)

init_log()
app = Flask(__name__)
csrf = CSRFProtect(app)

class Config(object):

    GPT_MODEL = load_model(os.getenv('OPENAI_ENGINE'))

    REF_DATA = ref_data
    FORMATTER = formaters['custom']

app.config.from_object('environment.Config')
