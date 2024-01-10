from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic_settings import BaseSettings

from langchain.chat_models import QianfanChatEndpoint
from langchain.llms import QianfanLLMEndpoint
from langchain_core.output_parsers import StrOutputParser

from langchain.chains import LLMChain


import os
import asyncio

from loguru import logger

class BaiduConfig(BaseSettings):
    QIANFAN_AK: str
    QIANFAN_SK: str
    
    class Config:
        env_file = ".env_baidu"

class ModelCenter:
    def __init__(self) -> None:
        self.llm_model = {}
        self.chat_model = {}
        
        self.llm_model['qianfan'] = self.qianfan()

        
    def qianfan(self):
        self.baidu_config = BaiduConfig()
        os.environ["QIANFAN_AK"] = self.baidu_config.QIANFAN_AK
        os.environ["QIANFAN_SK"] = self.baidu_config.QIANFAN_SK
        qianfan_llm = QianfanLLMEndpoint(streaming=True)
        
        return qianfan_llm
    
    def ask(self,question:str, model = 'qianfan'):
        answer = self.llm_model['qianfan'](question)
        return answer
    
    async def aask(self,question:str, model = 'qianfan'):
        answer = await self.llm_model['qianfan'].agenerate(prompts=[question])
        #print(answer)
        # print(type(answer))
        # print(answer)
        #output_parser = StrOutputParser()
        answer_str = answer.generations[0][0].text
        print('parse:',answer.generations[0][0].text)

        return answer_str
    
    async def ainvoke(self,prompt_template:str,prompt_value:dict,model = 'qianfan'):
        output_parser = StrOutputParser()
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm_model[model] | output_parser
        reply_content = await chain.ainvoke(prompt_value)
        # reply_content = output_parser.invoke(message)
        logger.info(f'prompt:{prompt}\n value:{prompt_value}\n reply:{reply_content}')
        
        # prompt_template = PromptTemplate.from_template(prompt)
        
        # chain = LLMChain(llm=self.llm_model[model], prompt=prompt)
        
        # response = await chain.arun(prompt_value)
        #return response


 
def test_modelmodel():
    model_center = ModelCenter()
    # answer = model_center.ask('请介绍一下鲁迅先生的简历')
    answer = model_center.aask('请介绍一下鲁迅先生的简历')
    asyncio.get_event_loop().run_until_complete()
    
    
    return

async def async_test():
    model_center = ModelCenter()
    # model_center.aask('请介绍一下鲁迅先生的简历')
    task1 = asyncio.create_task(model_center.aask('请介绍一下鲁迅先生的简历'))
    #task2 = asyncio.create_task(model_center.aask('请介绍一下周树人先生的简历'))
    #await asyncio.gather(task1, task2)
    await asyncio.gather(task1)

async def chain_test():
    
    prompt_template = """
Given the user's name, write them a personalized greeting. 

User's name: {name}

Your response:
"""
    prompt_value = {'name':'孙悟空'}
    
    model_center = ModelCenter()
    task1 = asyncio.create_task(model_center.invoke(prompt_template,prompt_value))
    await asyncio.gather(task1)

if __name__ == "__main__":
    #test_modelmodel()
    
    asyncio.get_event_loop().run_until_complete(chain_test())
    #asyncio.run(async_test())