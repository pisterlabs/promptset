# -*- coding: utf-8 -*-
#  @file        - qa_chain.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - langchain实现大模型问答类
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

import sys
sys.path.append("../")

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from llm.model_to_llm import model_to_llm
from database.get_vectordb import get_vectordb

# 默认prompting模板
default_template_rq = """
    使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:
    """

class QAChain():
    """ 不带历史记录的问答链
    """
    #基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    

    def __init__(self, 
                 model:str, 
                 temperature:float=0.0, 
                 top_k:int=4,  
                 file_path:str=None, 
                 persist_path:str=None, 
                 appid:str=None, 
                 api_key:str=None,
                 Spark_api_secret:str=None,
                 Wenxin_secret_key:str=None, 
                 embedding = "openai",  
                 embedding_key = None, 
                 template=default_template_rq):
        """
        - model: 调用的模型名称
        - temperature: 温度系数，控制生成的随机性
        - top_k: 返回检索的前k个相似文档
        - file_path: 建库文件所在路径
        - persist_path: 向量数据库持久化路径
        - appid: 星火需要输入
        - api_key: 所有模型都需要
        - Spark_api_secret: 星火秘钥
        - Wenxin_secret_key: 文心秘钥
        - embeddings: 使用的embedding模型  
        - embedding_key: 使用的embedding模型的秘钥 (智谱或者OpenAI)
        - template: 可以自定义提示模板, 没有输入则使用默认的提示模板default_template_rq
        """
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template

        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=self.template)
        self.retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': self.top_k})  #默认similarity，k=4
        # 自定义 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        retriever=self.retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})

    def answer(self, question : str = None, temperature = None, top_k = 4):
        """
        核心方法，调用问答链
        arguments: 
            - question: 用户提问
        """
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
            
        if top_k == None:
            top_k = self.top_k

        result = self.qa_chain({"query": question, 
                                "temperature": temperature, 
                                "top_k": top_k})
        return result["result"]   
    
class QAChainWithHistory():
    """ 带历史记录的问答链
    """
    def __init__(self, 
                 model:str, 
                 temperature:float=0.0, 
                 top_k:int=4, 
                 chat_history:list=[], 
                 file_path:str=None, 
                 persist_path:str=None, 
                 appid:str=None, 
                 api_key:str=None, 
                 Spark_api_secret:str=None,
                 Wenxin_secret_key:str=None, 
                 embedding = "openai",
                 embedding_key:str=None):
        """
        - model: 调用的模型名称
        - temperature: 温度系数，控制生成的随机性
        - top_k: 返回检索的前k个相似文档
        - chat_history: 历史记录，输入一个列表，默认是一个空列表
        - history_len: 控制保留的最近 history_len 次对话
        - file_path: 建库文件所在路径
        - persist_path: 向量数据库持久化路径
        - appid: 星火
        - api_key: 星火、百度文心、OpenAI、智谱都需要传递的参数
        - Spark_api_secret: 星火秘钥
        - Wenxin_secret_key: 文心秘钥
        - embeddings: 使用的embedding模型
        - embedding_key: 使用的embedding模型的秘钥(智谱或者OpenAI) 
        """
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        #self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key


        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
      
    def clear_history(self):
        """ 清空历史记录
        """
        return self.chat_history.clear()

    
    def change_history_length(self,history_len:int=1):
        """ 保存指定对话轮次的历史记录
        输入参数：
            - history_len ：控制保留的最近 history_len 次对话
            - chat_history: 当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def answer(self, question:str=None, temperature = None, top_k = 4):
        """" 核心方法，调用问答链
        arguments: 
        - question: 用户提问
        """
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature

        llm = model_to_llm(self.model, temperature, self.appid, 
                           self.api_key, self.Spark_api_secret,
                           self.Wenxin_secret_key)

        #self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': top_k})  #默认similarity，k=4

        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever,

        )

        #print(self.llm)
        result = qa({"question": question,"chat_history": self.chat_history})       #result里有question、chat_history、answer
        answer =  result['answer']
        self.chat_history.append((question,answer)) #更新历史记录

        return answer, self.chat_history  #返回本次回答和更新后的历史记录
