from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.prompts import PromptTemplate

from langchain.llms import AzureOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import sentence_transformers
from  config.config import *


class QAProcessor():

    def __init__(self, emb_type=0, request=0, temperate=0, chain_history=[]):
        self.emb_type = emb_type
        self.temperate = temperate
        self.chain_history = chain_history
        self.request = request

    def get_embeddings(self):
        if self.emb_type == 1:
            embeddings = HuggingFaceEmbeddings(
                model_name="GanymedeNil/text2vec-large-chinese", )
            embeddings.client = sentence_transformers.SentenceTransformer(self.embeddings.model_name,
                                                                          device=EMBEDDING_DEVICE)
            return embeddings
        else:
            embeddings = OpenAIEmbeddings()
            return embeddings

    def get_memory(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
        return memory

    def get_prompt(self):
        input_v_list_holder = ["question", "chat_history", "related_context"]
        prompt_template = """假设你是 bohrium 客服机器人, 你需要回答用户关于 bohrium 相关问题,要求: --回答时参考历史聊天记录以及相关的软件工程, 高性能计算, 科学方面的相关文档记录;--用规范的语言来回答;--对于 特别复杂的问题, 你可以尝试 think step by step;--对于你不清楚的问题, 不要主观编造, 你可以回复 "抱歉, 我不清楚,请提供更多的技术细节";--尽可能用中文回复;
            历史对话记录:
            ------\n{chat_history}------\n
            参考文档如下:
            ------\n{related_context}------\n
            目前对话:
            Human: {input}
            AI:     
             """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_v_list_holder,
        )
        return prompt

    def get_db_context(self, db, query):
        pass

    def insert_db(self, db, query, responce):
        pass

    def get_llm_model(self):
        gpt_name = "gpt-35"
        llm = AzureOpenAI(deployment_name=gpt_name,
                          model_name="gpt-35-turbo", max_tokens=500, temperature=0)
        return llm

    def qa_chain(self, query, chat_history, related_context, memory, llm, prompt_template):
        # qa_chain = LLMChain(llm = llm, prompt = prompt_template)
        conversation_with_summary = ConversationalRetrievalChain(
            llm=llm,
            prompt=prompt_template,
            # We set a very low max_token_limit for the purposes of testing.
            memory=memory,
            verbose=True
        )
        res = conversation_with_summary.predict(
            question=query, chat_history=chat_history, related_context=related_context)
        return res
