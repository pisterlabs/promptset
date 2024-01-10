import traceback
from datetime import datetime
from typing import List

import pytz
from openai import OpenAI

from .constants import *
from .gpt_model_enum import GptModelDefines


def completion_top_append_file(
        current_content: str,
        file_path: str | None = HISTORY_PROMPT):
    # 将字符串写入文件的最顶端
    with open(file_path, 'r+', encoding=UTF8) as file:
        original_content = file.read()
        file.seek(0, 0)
        file.write(current_content + '\n' + original_content)


def get_current_time() -> str:
    """
    :return: 获取UTC8的当前系统时间(年-月-日 时:分:秒)
    """
    return datetime.now(pytz.timezone('Asia/Shanghai')) \
        .strftime('%Y-%m-%d %H:%M:%S')


def get_chat_completion(prompt: str,
                        client: OpenAI | None = None,
                        gpt_model_define: GptModelDefines | None = GptModelDefines.GPT3_5_TURBO):
    """
    get GPT Response Chat Completion
    :param prompt: user prompt; Required
    :param client: OpenAI;
        Optional, Default is Official Client
    :param gpt_model_define:
        GptModelDefines Enum; Optional,	Default is 'gpt-3-turbo'
    :return: `openai.resources.chat.Completions`
    """
    if client is None:
        client = official_client()
    completion = client.chat.completions.create(
        model=get_model_name(gpt_model_define),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CHINESE},
            {"role": "system", "content": SYSTEM_PROMPT_FORMAT},
            {"role": "user", "content": prompt}
        ]
    )
    return completion


def get_file_content(file_path: str) -> str:
    """
    Read the file's content and strip it based on the provided file path.

    :param file_path: path of file, stored api key
    :return: api key
    """
    try:
        with open(file_path, 'r', encoding=UTF8) as file:
            content = file.read()
        return content.strip()
    except FileNotFoundError:
        traceback.print_exc()
        print("file not found!please check file path: %s" % file_path)


def official_api_key() -> str:
    return get_file_content(OFFICIAL_API_FILE)


def proxy_api_key() -> str:
    return get_file_content(PROXY_API_FILE)


def get_model_name(model_enum: GptModelDefines) -> str:
    return model_enum.value


def official_client():
    return OpenAI(api_key=official_api_key())


def official_safe_client():
    return OpenAI(base_url=get_file_content(OFFICIAL_SAFE_URL), api_key=official_api_key())

def proxy_client():
    return OpenAI(
        api_key=get_file_content(PROXY_API_FILE),
        base_url=get_file_content(PROXY_URL)
    )


def get_model_list() -> List[str]:
    client = official_client()
    models = client.models.list()
    model_names = list()
    for model in models:
        model_names.append(model.id)
    model_names.sort()
    return model_names
