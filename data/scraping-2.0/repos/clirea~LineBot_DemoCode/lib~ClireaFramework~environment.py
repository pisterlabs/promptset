from sqlalchemy.ext.declarative import declarative_base
import pytz
import logging
import os
import tiktoken
import json
import openai
from openai import OpenAI
from tiktoken.core import Encoding
from datetime import datetime
from .config import DebugConfig
jst = pytz.timezone('Asia/Tokyo')
now = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
Base = declarative_base()
# ロガーの作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ハンドラの作成
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# フォーマッタの作成
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# ロガーにハンドラを追加
logger.addHandler(ch)


_DEBUG = False

ChannelAccessToken =  ""
ChannelSecret =  ""
OpenaiApiKey = ""
BacketName = ""
FileAge =""
GptModel = ""
GptFunctionModel = ""
MaxTokens:int = 0
encoding: Encoding = None

#RELEASE
if _DEBUG == False:
    ChannelAccessToken =  os.environ.get('LINE_ACCESS_TOKEN')
    ChannelSecret =  os.environ.get('LINE_CHANNEL_SECRET')
    OpenaiApiKey = os.environ.get('OPENAI_APIKEY')
    BacketName = os.environ.get('BACKET_NAME')
    FileAge = os.environ.get('FILE_AGE')
    GptModel = os.environ.get('GPT_MODEL')
    GptFunctionModel = os.environ.get('GPT_FUNCTION_MODEL')
    MaxTokens:int = 2500
    encoding: Encoding = tiktoken.encoding_for_model(GptModel)
    ReplyCount = 3
    
    username = os.environ.get('MYSQL_USER')
    password = os.environ.get('MYSQL_PASS')
    server = os.environ.get('MYSQL_HOST')
    database = os.environ.get('MYSQL_DB_NAME')
    unix_socket_path = os.environ.get('INSTANCE_UNIX_SOCKET')

#DEBUG
if _DEBUG == True:
    config = DebugConfig

    ChannelAccessToken = config.get("LINE_ACCESS_TOKEN")
    ChannelSecret = config.get("LINE_CHANNEL_SECRET")
    OpenaiApiKey = config.get("OPENAI_APIKEY")
    BacketName = config.get("BACKET_NAME")
    FileAge = config.get("FILE_AGE")
    GptModel = config.get("GPT_MODEL")
    GptFunctionModel = config.get("GPT_FUNCTION_MODEL")
    MaxTokens = config.get("MAX_TOKENS")
    encoding = tiktoken.encoding_for_model(GptModel)
    ReplyCount = 3

    username =config.get('MYSQL_USER')
    password = config.get('MYSQL_PASS')
    server =config.get('MYSQL_HOST')
    database = config.get('MYSQL_DB_NAME')
    unix_socket_path = config.get('INSTANCE_UNIX_SOCKET')
                                  
# openai.api_key = OpenaiApiKey
client = OpenAI(api_key=OpenaiApiKey)