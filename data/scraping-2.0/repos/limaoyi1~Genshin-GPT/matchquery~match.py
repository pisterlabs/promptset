# init 初始化一些常数
import re

import torch
from langchain import OpenAI, LLMChain
from langchain.agents import LLMSingleActionAgent, AgentOutputParser, AgentExecutor, initialize_agent, AgentType, \
    ZeroShotAgent
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.experimental import load_chat_planner, load_agent_executor, PlanAndExecute
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document, AgentAction, AgentFinish
from langchain.vectorstores import Chroma

from matchquery.dbtools import tools
from readconfig.myconfig import MyConfig
from pympler import tracker

config = MyConfig()

# tr = tracker.SummaryTracker()
metadata_field_info = [
    AttributeInfo(
        name="language",
        description="The language used by the character",
        type="string",
    ),
    AttributeInfo(
        name="npcName",
        description="The name of the character",
        type="string",
    ),
    AttributeInfo(
        name="type",
        description="Scenes where the character speak",
        type="string",
    )
]

# llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY,openai_api_base= config.OPENAI_BASE_URL)

document_content_description = "All dialogues of game characters"

embedding_model_dict = {
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "hinese-macbert": "hfl/chinese-macbert-base"
}

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 加载嵌入模型 ==============================================================
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict['hinese-macbert'],
                                   model_kwargs={'device': EMBEDDING_DEVICE})

# 加载VectorDB =============================================================
vectordb = Chroma(persist_directory="./resource/dict/v4", embedding_function=embeddings)

# 加载VectorDB =============================================================
vectordb_wiki = Chroma(persist_directory="./resource/dict/v1", embedding_function=embeddings)

vectordb_tools = Chroma(persist_directory="./resource/dict/tools", embedding_function=embeddings)


class MatchAnswer:
    role_name: str = None
    llm_questions: [] = []

    def __init__(self, role_name):
        self.role_name = role_name

    def match(self, raw_answer):
        # # 查看内存学习
        # print("查看内存信息")
        # print(tr.print_diff())
        # llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, openai_api_base=config.OPENAI_BASE_URL)
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=config.OPENAI_API_KEY,
                     openai_api_base=config.OPENAI_BASE_URL)
        # 加载检索器 ================================================================
        retriever = SelfQueryRetriever.from_llm(
            llm, vectordb, document_content_description, metadata_field_info, verbose=False, enable_limit=True
        )
        # 组合 多查询检索器 和 自我检索器
        # 减少数量优化内存
        template = f"""You are an AI language model assistant. Your task is to generate three 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions seperated by newlines.
            Original question: {raw_answer}"""
        questions = llm.predict(template)
        output_list = questions.split("\n")
        contents = []
        self.llm_questions = output_list
        for i in range(len(output_list)):
            print(output_list[i])
            # 本人看法
            query = f"""{self.role_name} said :```{output_list[i]}```"""
            documents = retriever.get_relevant_documents(query)
            for doc in documents:
                # 去重
                if doc.page_content not in contents:
                    contents.append(doc.page_content)
            # if i == len(output_list) - 1:
            #     row_docs = retriever.get_relevant_documents(raw_answer)
            #     for row_doc in row_docs:
            #         if row_doc.page_content not in contents:
            #             contents.append(row_doc.page_content)

            # retriever.
            # vectordb.
        # print("查看内存信息2")
        # print(tr.print_diff())
        # 在wikipedia 中检索
        return contents

    metadata_wiki_info = [
        AttributeInfo(
            name="theme",
            description="theme of Wiki",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="Source of Wiki Information",
            type="string",
        ),
        AttributeInfo(
            name="type",
            description="data type",
            type="string",
        )
    ]

    def matchWiki(self, raw_answer):
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=config.OPENAI_API_KEY,
                     openai_api_base=config.OPENAI_BASE_URL)
        # 加载检索器 ================================================================
        retriever = SelfQueryRetriever.from_llm(
            llm, vectordb_wiki, document_contents="Some wiki text", metadata_field_info=self.metadata_wiki_info,
            verbose=True, enable_limit=True
        )
        # retriever = vectordb_wiki.as_retriever(search_type="mmr",search_kwargs={"k": 1})
        # querys = self.llm_questions
        querys = ["two wiki about" + self.llm_questions[0] + f""" which theme is {self.role_name}""",
                  "two wiki about" + self.llm_questions[0], "two wiki about " + self.role_name]
        contents = []
        for q in querys:
            documents = retriever.get_relevant_documents(q)
            for doc in documents:
                # 去重
                if doc.page_content not in contents:
                    contents.append(doc.page_content)
        return contents

    def matchTools(self, raw_answer):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=config.OPENAI_API_KEY,
                         openai_api_base=config.OPENAI_BASE_URL)
        query = f"获取一些基础信息关于{self.role_name}的提问:{raw_answer}"
        # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
        #                          handle_parsing_errors="Check your output and make sure it conforms!")
        agent = self.get_agent()
        try:
            result = agent.run(query)
            return result
        except Exception as e:
            print(f"链式分析异常: {e}")
            return ""

    def get_agent(self):
        chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=config.OPENAI_BASE_URL,
            streaming=True,
            verbose=False
        )

        prefix = """你是一个优秀的COSPALYER,之所以优秀在于你会利用从文档数据库中获得的知识，通过分析对象说的话来获取大量的信息.
        你的目标是通过文本推理出的信息查询数据库,获得丰富背景信息和回答的建议,首先在我给你的数据库中搜索答案,如果失败可以尝试其他的可能有用的工具.
        当你仍然缺乏必要的信息时，你才能使用维基百科搜索来查找网络文章的结果.
        为了让用户更容易理解和阅读，你应该始终将最终答案作为要点提供。传入参数内容使用简体中文.
        
        You have access to the following tools: """

        FORMAT_INSTRUCTIONS = """Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the detailed,at most comprehensive result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer based on my observation
        Final Answer: the final answer to the original input question is the full detailed explanation from the Observation provided as bullet points."""

        suffix = """Begin!"

                {chat_history}
                Question: {input}
                {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=FORMAT_INSTRUCTIONS,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        def _handle_error(error) -> str:
            INSTRUCTIONS = """Use the following format:

                  Thought: you should always think about what to do
                  Action: the action to take, should be one of [{tool_names}]
                  Action Input: the input to the action  
                  Observation: the detailed, comprehensive result of the action
                  Thought: I now know the final answer based on my observation
                  Final Answer: the final answer to the original input question is the full detailed explanation from the Observation provided as bullet points."""

            ouput = str(error).removeprefix("Could not parse LLM output: `").removesuffix("`")

            response = f"Thought: {ouput}\nThe above completion did not satisfy the Format Instructions given in the Prompt.\nFormat Instructions: {INSTRUCTIONS}\nPlease try again and conform to the format."
            # print("error msg: ", response)
            return response

        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

        llm_chain = LLMChain(llm=chat_model, prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, handle_parsing_errors=_handle_error)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=_handle_error
        )

        return agent_chain





if __name__ == "__main__":
    answer = MatchAnswer("钟离")
    matchs = answer.match("早安")
    print(matchs)

