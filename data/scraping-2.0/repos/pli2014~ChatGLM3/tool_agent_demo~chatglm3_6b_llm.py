#从typing库中导入必要的函数和类型声明
from typing import Any, List, Mapping, Optional

#导入所需的类和接口
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json

# 语法解释：
# 1. 使用typing库中的相关类型进行类型声明
# 2. 使用继承实现自定义LLM类的功能扩展
# 3. 通过重写父类的方法以实现特定的功能需求
# 4. 使用@property装饰器很好地实现了对私有变量和方法的封装和保护
# 5. _identifying_params属性和_llm_type属性分别用于标识和记录各自对象的属性和类型信息。

#定义一个名为CustomLLM的子类，继承自LLM类
base_url = "http://localhost:6006" # 把这里改成你的IP地址

def create_chat_completion(model, messages, use_stream=False):
    data = {
        "model": model, # 模型名称
        "messages": messages, # 会话历史
        "stream": use_stream, # 是否流式响应
        "max_tokens": 100, # 最多生成字数
        "temperature": 0.8, # 温度
        "top_p": 0.8, # 采样概率
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        if use_stream:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        #print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            return content
    else:
        print("Error:", response.status_code)
        return None

class Chatglm3_6b_LLM(LLM):

    # 用于指定该子类对象的类型
    @property
    def _llm_type(self) -> str:
        return "Chatglm3_6b_LLM"

    # 重写基类方法，根据用户输入的prompt来响应用户，返回字符串
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        chat_messages = [
            {
            "role": "system",
            "content": "从现在开始扮演一个agent智能体"
            }
        ]
        chat_messages.append({"role": "user", "content": prompt})
        response = create_chat_completion("chatglm3-6b", chat_messages, use_stream=False)
        print(f"===llm answer with response: {response}")
        return response

    # 返回一个字典类型，包含LLM的唯一标识
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "Chatglm3_6b_LLM"}
