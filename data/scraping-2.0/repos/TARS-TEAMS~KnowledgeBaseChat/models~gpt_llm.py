import configparser
from typing import List, Dict, Optional

import openai
from langchain.llms.base import LLM

from configs.model_config import *
from utils import torch_gc


class ChatGPT(LLM):
    max_token: int = 10000
    temperature: float = 0.8
    top_p = 0.9
    # history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              streaming: bool = STREAMING):  # -> Tuple[str, List[List[str]]]:
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0,
            max_tokens=4096,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\"\"\""]
        )
        res = response.choices[0].message.content
        torch_gc()
        history += [[prompt, res]]
        yield res, history
        torch_gc()

    def load_model(self,
                   model_name_or_path: str = "THUDM/chatglm-6b",
                   llm_device=LLM_DEVICE,
                   use_ptuning_v2=False,
                   use_lora=False,
                   device_map: Optional[Dict[str, int]] = None,
                   **kwargs):
        openai.api_key = get_properties("OPENAI_API_KEY")


def get_properties(option_name):
    return get_prop('default', option_name)


def get_prop(section, option_name):
    # 初始化API KEY
    config = configparser.RawConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    properties_file = os.path.join(current_dir, '../configs/environment.properties')
    print(properties_file)
    config.read(properties_file)

    value = ''
    try:
        value = config.get(section, option_name)
    except configparser.NoSectionError:
        print('No such section')
    except configparser.NoOptionError:
        print('No such option')

    return value


if __name__ == "__main__":
    llm = ChatGPT()
    llm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL], llm_device=LLM_DEVICE, )
    last_print_len = 0
    for resp, history in llm._call("你好", streaming=True):
        logger.info(resp[last_print_len:])
        last_print_len = len(resp)
    for resp, history in llm._call("你好", streaming=False):
        logger.info(resp)
    pass
