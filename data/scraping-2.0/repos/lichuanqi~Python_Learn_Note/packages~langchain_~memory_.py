from abc import ABC, abstractmethod
from typing import Dict,List,Any,Optional,Tuple,Union,Mapping
from pprint import pprint
from pydantic import BaseModel, Extra, Field, root_validator

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage


class NewMessageHistory(ChatMessageHistory):
    def wenxin_format(self):
        """把历史消息记录转换为百度文心格式
        
        格式 - 单轮请求
        [
            {"role":"user","content":"介绍一下你自己"}
        ]
        
        格式 - 多轮请求示例
        [
            {"role":"user","content":"请介绍一下你自己"},
            {"role":"assistant","content":"我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"},
            {"role":"user","content": "我在上海，周末可以去哪里玩？"}
        ]
        """
        wenxins = []

        if len(self.messages)%2 != 1:
            print('请确保消息数量为奇数')
            return 
        
        for i in range(0, len(self.messages)-1, 2):
            wenxins.append({"role": "user", "content": self.messages[i].content})
            wenxins.append({"role": "assistant", "content": self.messages[i+1].content})
        
        # 添加最后一条问题
        wenxins.append({"role": "assistant", "content": self.messages[-1].content})

        return wenxins
        


    def chatbot_format(self):
        """把历史消息记录转换为Gradio的Chatbot格式
        
        格式
        [
            ["请介绍一下你自己", "百度公司开发的人工智能语言模型"],
            [..., ...]
        ]
        """
        chatbots = []

        if len(self.messages)%2 == 1:
            print('请确保消息数量为偶数')
            return 

        for i in range(0, len(self.messages), 2):
            chatbots.append([self.messages[i].content, self.messages[i+1].content])

        return chatbots


class VectorySearchMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "vectory search"


class NewMessageHistoryV2(ChatMessageHistory):
    def add_vectory_search_message(self, message):
        self.messages.append(VectorySearchMessage(content=message))

    def wenxin_format(self):
        """把历史消息记录转换为百度文心格式
        
        格式 - 单轮请求
        [
            {"role":"user","content":"介绍一下你自己"}
        ]
        
        格式 - 多轮请求示例
        [
            {"role":"user","content":"请介绍一下你自己"},
            {"role":"assistant","content":"我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"},
            {"role":"user","content": "我在上海，周末可以去哪里玩？"}
        ]
        """
        wenxins = []

        if len(self.messages)%3 != 2:
            print('请确保消息数量为3n+2条')
            return 
        
        for i in range(0, len(self.messages)-2, 3):
            wenxins.append({"role": "user", "content": self.messages[i].content})
            wenxins.append({"role": "assistant", "content": self.messages[i+2].content})
        
        # 添加最后一条问题
        wenxins.append({"role": "assistant", "content": self.messages[-2].content})
        return wenxins
        
    def chatbot_format(self):
        """把历史消息记录转换为Gradio的Chatbot格式
        
        格式
        [
            ["请介绍一下你自己", "百度公司开发的人工智能语言模型"],
            [..., ...]
        ]
        """
        chatbots = []

        if len(self.messages)%3 != 0:
            print('请确保消息数量为3n条')
            return 

        for i in range(0, len(self.messages), 3):
            chatbots.append([self.messages[i].content, self.messages[i+2].content])

        return chatbots


def test_message_history():
    # messageHistory = NewMessageHistory()
    messageHistory = NewMessageHistoryV2()

    # 增加点测试数据
    messageHistory.add_user_message('你好啊')
    messageHistory.add_vectory_search_message('检索内容1')
    messageHistory.add_ai_message('还行')
    messageHistory.add_user_message('你好啊2')
    messageHistory.add_vectory_search_message('检索内容2')
    messageHistory.add_ai_message('还行2')

    # 增加一个用户消息
    question = '我又来了'
    messageHistory.add_user_message(question)
    messageHistory.add_vectory_search_message('检索内容3')

    # 转换成百度文心一言的格式
    pprint(messageHistory.wenxin_format())
    # 转换成chatbot格式回显
    # pprint(messageHistory.chatbot_format())

    # 拿到返回值后增加一个大模型消息
    answer = '还是欢迎'
    messageHistory.add_ai_message(answer)

    # 转换成chatbot格式回显
    pprint(messageHistory.chatbot_format())


if __name__ == '__main__':
    test_message_history()