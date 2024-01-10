from abc import ABC, abstractmethod

import requests
import os

from langchain_community.chat_message_histories import MongoDBChatMessageHistory, FileChatMessageHistory

from simpleaichat import prompt


###基于您的需求，可以对 CustomOutputParser 类进行扩展或修改，以实现特定的逻辑：当响应中包含 action 和 actionInput 时，截取 actionInput 以上的回复加入到上下文中，并执行 action 调用的函数。然后，将函数的输出结果添加到观察结果中，并连同上下文再次发送请求，直到响应中出现 finalAnswer。
# 设置环境变量（仅用于测试，实际部署时更换）
os.environ['OPENAI_API_KEY'] = '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'


class BaseAIGenerator(ABC):
    """AI文本生成器的基类。"""

    def __init__(self):
        self.history = []  # 初始化一个空的历史记录列表

    def generate(self, instruction: str) -> str:
        """生成文本的方法，需要在子类中实现。
        Args:
            instruction (str): 输入提示。
        Returns:
            str: 生成的文本。
        """
        generated_text = self._generate_text(instruction)  # 假设的内部方法来生成文本
        self._update_history(instruction, generated_text)  # 更新历史记录
        return generated_text

    def generate_with_rag(self, instruction: str, context: str, query: str) -> str:
        """生成带有额外查询的文本的方法，需要在子类中实现。
        Args:
            instruction (str): 输入提示。
            context (str): 上下文。
            query (str): 查询问题。
        Returns:
            str: 生成的文本。
        """
        generated_text = self._generate_text_with_rag(instruction, context, query)  # 假设的内部方法
        self._update_history(query, generated_text)  # 更新历史记录
        return generated_text

    @abstractmethod
    def _config_llm(self):
        """内部方法：在子类中实现具体的文本生成逻辑。"""
        raise NotImplementedError

    @abstractmethod
    def _generate_text(self, instruction: str) -> str:
        """内部方法：在子类中实现具体的文本生成逻辑。"""
        raise NotImplementedError

    def _generate_text_with_rag(self, instruction: str, context: str, query: str) -> str:
        """内部方法：在子类中实现具体的带额外查询的文本生成逻辑。"""
        raise NotImplementedError

    def _update_history(self, instruction: str, generated_text: str):
        """内部方法：更新历史记录。"""
        self.history.append({"user:": instruction, "tuji:": generated_text})

    def get_history(self):
        """获取当前的历史记录。"""
        return self.history


class LocalLLMGenerator(BaseAIGenerator):
    """使用本地语言模型的生成器。"""

    def _config_llm(self):
        model_url = "http://182.254.242.30:5001"
        url = f"{model_url}/v1/completions"
        # url = f"{model_url}/v1/chat/completions" ##chat模式
        headers = {"Content-Type": "application/json"}
        return url, headers

    def _generate_text(self, instruction: str) -> str:
        url = self._config_llm()[0]
        headers = self._config_llm()[1]
        data = {
            "prompt": instruction,
            # "message"[{  ##chat模式
            #     "role": "user",
            #     "content": instruction
            # }],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "seed": -1,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['text']
                # return data['choices'][0]['message'] ##chat模式
            else:
                raise Exception("响应中没有找到有效的 'choices' 数据")
        else:
            raise Exception(f"API 请求失败，状态码: {response.status_code}")

    def _generate_text_with_rag(self, instruction: str, context: str, query: str) -> str:
        url = self._config_llm()[0]
        headers = self._config_llm()[1]
        final_prompt = f"<|im_start|>{instruction}\n 参考资料:\n{context}\n{prompt.RAG}<|im_end|>\nuser:{query}\n兔叽:"
        data = {
            "prompt": final_prompt,
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "seed": -1,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['text']
            else:
                raise Exception("响应中没有找到有效的 'choices' 数据")
        else:
            raise Exception(f"API 请求失败，状态码: {response.status_code}")

    def generate(self, instruction: str) -> str:
        return super().generate(instruction)

    def generate_with_rag(self, instruction: str, context: str, query: str) -> str:
        return super().generate_with_rag(instruction, context, query)


class OpenAIGenerator(BaseAIGenerator):

    def _config_llm(self):
        model_url = "https://api.openai.com"
        url = f"{model_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
        }
        return url, headers

    def _generate_text(self, instruction: str) -> str:
        url = self._config_llm()[0]
        headers = self._config_llm()[1]
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": instruction}]
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and data['choices']:
                try:
                    return data['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"解析响应时出错: {e}")
            else:
                raise Exception("响应中没有找到有效的 'choices' 数据")
        else:
            raise Exception(f"API 请求失败，状态码: {response.status_code}")

    def generate(self, instruction: str) -> str:
        return super().generate(instruction)

    def generate_with_rag(self, instruction: str, context: str, query: str) -> str:
        return super().generate_with_rag(instruction, context, query)

