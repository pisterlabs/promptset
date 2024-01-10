import openai
from dotenv import load_dotenv
import os
from lib.cache import *


load_dotenv()  # 加载.env文件中的环境变量
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv('OPENAI_API_KEY')
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')


redis_handle = RedisCache(redis_host,redis_port, redis_password)
redis_handle.create_pool()
redis_handle.get_redis()
