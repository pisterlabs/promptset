import os
import random

from dotenv import load_dotenv

from common.log import LogUtils
from config.settings import OPENAI_API_KEYS

# 加载 .env 文件
load_dotenv()

loggerOpenAI = LogUtils.new_logger("openai-Key")


class OpenAIUtilsKey:

    @staticmethod
    def get_key_in_config():
        _list_keys = OPENAI_API_KEYS
        if not _list_keys:
            loggerOpenAI.error("The OpenAI Key Configure Item Were Not Found in The Configuration File.")
        else:
            if len(_list_keys) == 1 and _list_keys[0] == "":
                loggerOpenAI.error("The OpenAI Key Has Not Been Configured in The Configuration File.")
            else:
                return random.choice(_list_keys)
        return

    @staticmethod
    def get_key_in_env():
        api_keys = os.environ.get('OPENAI_API_KEYS')

        if not api_keys:
            return None

        _list_keys = api_keys.split(",")
        _list_keys = [key.strip() for key in _list_keys if key.strip()]

        if not _list_keys:
            return None

        return random.choice(_list_keys)
