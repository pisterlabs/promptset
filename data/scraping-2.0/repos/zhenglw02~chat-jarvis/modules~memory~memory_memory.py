import logging
import openai
import json
import time
from datetime import datetime

from config import system_config
from modules.memory.memory_interface import AbstractMemory
from modules.model.chat_item import ChatItem
from modules.long_memory.long_memory_interface import AbstractLongMemory
from modules.long_memory.long_memory_interface import LongMemoryItem

openai.api_key = system_config.MEMORY_SUMMARY_OPENAI_API_KEY
openai.api_base = system_config.MEMORY_SUMMARY_OPENAI_API_BASE


class MemoryMemory(AbstractMemory):
    def __init__(self):
        self._logger = None
        self._messages = None
        self._long_memory = None

    def init(self, logger: logging.Logger, long_memory: AbstractLongMemory):
        self._logger = logger
        self._messages = []
        self._long_memory = long_memory

    def save(self, message: ChatItem):
        self._messages.append(message)
        if len(self._messages) > system_config.MEMORY_MAX_LENGTH:
            self._logger.debug("memory is max, summary them")
            self._extract_long_memory()
            _new_messages = []
            if system_config.MEMORY_SUMMARY_ENABLE:
                history = ""
                for message in self._messages[1:]:
                    history += "【{}】：{}\n".format(message.role, message.content)
                response = openai.ChatCompletion.create(
                    model=system_config.MEMORY_SUMMARY_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个资深的文字工作者."},
                        {"role": "user", "content": "你的任务是总结下面的对话内容:\n{}.请按照以下的格式总结上面的对话内容：\n"
                                                    "【用户和贾维斯正在做的事情】：\n"
                                                    "【其他内容】：".format(history)}
                    ],
                )
                message = response.choices[0].message
                _new_messages.append(ChatItem.new("user", message.content))
            self._messages = _new_messages

    def load_recent(self):
        return self._messages

    def clear_recent(self):
        self._extract_long_memory()
        self._messages = []

    def _extract_long_memory(self):
        self._logger.debug("extract long memory")
        history = ""
        for message in self._messages[1:]:
            history += "【{}】：{}\n".format(message.role, message.content)
        response = openai.chat.completions.create(
            model=system_config.MEMORY_SUMMARY_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个优秀的管家，我是你的主人."},
                {"role": "user", "content": "你的任务是从下面的【对话内容】中提取你需要长期记住的信息，主要是关于我的个人喜好，生活习惯等管家需要记住的内容，以及我要求你记住的其他内容。\n"
                                            "【对话内容】:\n{}\n"
                                            "你应该按照json列表的形式输出需要长期记住的信息，下面是一个示例：\n"
                                            "[\"用户的名字是liwei。\", \"用户的出生年份是1997年。\", \"用户要求管家记住以后说话要简洁一点。\"]\n"
                                            "如果上面的【对话内容】中没有需要长期记住的信息，你应该返回空列表：[]\n"
                                            "你应该直接返回文字内容，不能进行函数调用。".format(history)}
            ],
        )
        message = response.choices[0].message
        long_memory_items = json.loads(message.content)
        items = []
        for item in long_memory_items:
            items.append(LongMemoryItem.new(content=item, id=str(time.time_ns()),
                                            metadata={"datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}))
        self._long_memory.save(items)
