import logging
import openai

from config import system_config

from plugin.plugin_interface import AbstractPlugin, PluginResult
from jarvis.jarvis import Jarvis

openai.api_key = system_config.SUMMARY_FILE_OPENAI_API_KEY
openai.api_base = system_config.SUMMARY_FILE_OPENAI_API_BASE


class FileQueryPlugin(AbstractPlugin):
    def valid(self) -> bool:
        return True

    def __init__(self):
        self._logger = None

    def init(self, logger: logging.Logger):
        self._logger = logger

    def get_name(self):
        return "file_query"

    def get_chinese_name(self):
        return "文档问答"

    def get_description(self):
        return "文档问答接口，本接口可以针对一个文件或下载的网页内容进行特定问题的查询。"

    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要提问的文件路径，应该是绝对路径。",
                },
                "query": {
                    "type": "string",
                    "description": "要提问的问题。",
                }
            },
            "required": ["file_path", "query"],
        }

    def run(self, jarvis: Jarvis, args: dict) -> PluginResult:
        query = args.get("query")
        file_path = args.get("file_path")
        with open(file_path, "r") as f:
            content = f.read()
        response = openai.ChatCompletion.create(
            model=system_config.SUMMARY_FILE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "你是一个资深的文字工作者."},
                {"role": "user",
                 "content": "你的任务是根据下面的【输入内容】，回答【问题】\n【输入内容】: {}\n【问题】：{}".format(content,
                                                                                                          query)}
            ],
        )
        message = response.choices[0].message
        self._logger.debug("file query result: \n{}\n".format(message.content))
        return PluginResult.new(
            result="问题的答案如下：\n{}\n你应该优先考虑把上面的总结内容直接返回给用户。".format(message.content),
            need_call_brain=True)
