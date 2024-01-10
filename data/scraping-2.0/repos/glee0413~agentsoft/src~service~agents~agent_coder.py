from modules.agent import LLMAgent
from modules.message import Message
from langchain.prompts import ChatPromptTemplate
from prompt.dev_prompt import python_coder_template
from config.constant import ProfessionType

class CoderAgent(LLMAgent):
    def __init__(self, name):
        super().__init__(name,profession=ProfessionType.PT_EXPERT_PYTHON)
        self.python_hat = ChatPromptTemplate.from_template(python_coder_template)
    
    async def Conclude(self,content:str):
        # 总结的函数
        prompt_value = {'demand':content}
        response = await self.llm.ainvoke(python_coder_template,prompt_value)
        return response
    
    def stop(self):
        return
    
    def stop(self):
        # 停止Agent
        pass

def test_llmagent():
    agent = CoderAgent('Python Expert')
    agent.launch()
    pass

if __name__ == "__main__":
    test_llmagent()