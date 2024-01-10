from langchain.llms import Tongyi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import dashscope


class LLMQuery:
    # 使用通义千问模型，回答用户的问题

    def __init__(self, template_path: str, api_key: str, model_name: str = "qwen-72b-chat"):
        dashscope.api_key = api_key

        with open(template_path, 'r', encoding="utf-8") as f:
            template = f.read()

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )

        self.llm = LLMChain(
            llm=Tongyi(dashscope_api_key=api_key, model_name=model_name),
            prompt=prompt,
            verbose=False,
            memory=ConversationBufferWindowMemory(return_messages=False)
        )

    def query(self, text: str):
        answer = self.llm.predict(human_input=text)
        return answer
