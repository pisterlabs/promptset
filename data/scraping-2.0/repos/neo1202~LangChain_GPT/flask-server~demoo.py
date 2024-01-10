import math, os, random, csv
from config import OPEN_API_KEY, PINECONE_KEY, SERP_API_KEY
from tabnanny import verbose
import pandas as pd
import numpy as np
# log and save
import json, logging, pickle, sys, shutil, copy
from argparse import ArgumentParser, Namespace
from pathlib import Path
from copy import copy
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import torch

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader

from langchain.vectorstores import Pinecone
import pinecone
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import OpenAI
from langchain.agents import AgentType
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import YoutubeLoader
import nltk
nltk.download('punkt')
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Union
import zipfile
import re
from langchain import SerpAPIWrapper, LLMChain, LLMMathChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory 

from langchain.document_loaders import WebBaseLoader
from serpapi import GoogleSearch

from langchain.chat_models import ChatOpenAI
from text2vec import SentenceModel
global_llm_chat, global_embeddings = None, None

def initialize():
    os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
    os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
    #OpenAI類默認對應 「text-davinci-003」版本：
    #OpenAIChat類默認是 "gpt-3.5-turbo"版本
    #OpenAI是即將被棄用的方法，最好是用ChatOpenAI
    #model = SentenceModel('shibing624/text2vec-base-chinese')
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese', model_kwargs={'device': EMBEDDING_DEVICE})  #768維度
    llm_chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") #GPT-3.5-turbo
    #llm_chat = ChatOpenAI(temperature=0, model_name="gpt-4")
    return llm_chat, embeddings

def process_and_store_documents(file_paths: List[str]) -> None:
    llm_chat, embeddings = initialize()
    def init_txt(file_pth: str):
        loader = TextLoader(file_pth)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=[" ", ",", "\n", "\n\n", "\t", ""]
        )
        split_docs_txt = text_splitter.split_documents(documents)
        return split_docs_txt
    
    def init_csv(file_pth: str):
        # my_csv_loader = CSVLoader(file_path=f'{file_pth}',encoding="utf-8", 
        #                           csv_args={'delimiter': ','
        # })
        loader = DirectoryLoader(f'{file_pth}', glob='**/*.csv', loader_cls=CSVLoader, silent_errors=True)
        documents = loader.load()
        split_docs_csv = documents #這份csv資料已經人為切割
        return split_docs_csv
    
    def init_xlsx(file_pth: str):
        loader = UnstructuredExcelLoader(file_pth,mode="elements")
        split_docs_xlsx = loader.load() 
        return split_docs_xlsx
    
    def init_pdf(file_pth: str):
        loader = PyPDFLoader(file_pth)
        split_docs_pdf = loader.load_and_split()
        return split_docs_pdf

    def init_word(file_pth: str):
        loader = Docx2txtLoader(file_pth)
        split_docs_word = loader.load()
        return split_docs_word
    
    def init_ustruc(file_pth: str):
        loader = UnstructuredFileLoader(file_pth)
        split_docs_ustruc = loader.load()
        return split_docs_ustruc
    
    pinecone.init(
        api_key=PINECONE_KEY,
        environment="us-west1-gcp-free"
    )
    index_name="demo-langchain" #768 #open ai embedding為1536向量

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine', #or dotproduct
            dimensions=768
        )
    doc_chunks = []
    for file_path in file_paths:
        txt_docs = init_txt(file_path)
        doc_chunks.extend(txt_docs)
    Pinecone.from_texts([t.page_content for t in doc_chunks], embeddings, index_name=index_name)
    
def get_my_agent():
    llm_chat, embeddings = initialize()
    pinecone.init(
        api_key= PINECONE_KEY,
        environment="us-west1-gcp-free"
    )
    index_name="demo-langchain" 
    index = pinecone.Index(index_name)
    print(f"\n我的資料庫現在有: {index.describe_index_stats()}筆向量\n") 
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    CONTEXT_QA_Template = """
    根據以下提供的信息，回答用戶的問題
    信息：{context}

    問題：{query}
    """
    CONTEXT_QA_PROMPT = PromptTemplate(
        input_variables=["context", "query"],
        template=CONTEXT_QA_Template,
    )
    class FugeDataSource:
        def __init__(self, llm:ChatOpenAI(temperature=0.1)):
            self.llm = llm

        def find_product_description(self, product_name: str) -> str:
            """模拟公司产品的数据库"""
            product_info = {
                "好快活": "好快活是一個營銷人才平台，以社群+公眾號+小程序結合的運營模式展開，幫助企業客戶連接並匹配充滿才華的營銷人才。",
                "Rimix": "Rimix通過採購流程數字化、完備的項目數據存儲記錄及標準的供應商管理體系，幫助企業實現採購流程, 透明合規可追溯，大幅節約採購成本。Rimix已為包括聯合利華，滴滴出行等多家廣告主提供服務，平均可為客戶節約採購成本30%。",
                "Bid Agent": "Bid Agent是一款專為中國市場設計的搜索引擎優化管理工具，支持5大搜索引擎。Bid Agent平均為廣告主提升18%的投放效果，同時平均提升47%的管理效率。目前已為陽獅廣告、GroupM等知名4A公司提供服務與支持。",
            }
            return product_info.get(product_name, "没有找到这个产品")

        # def find_company_info(self, query: str) -> str:
        #     """模擬公司介紹文檔數據庫，讓llm根據抓取信息回答問題"""
        #     context = """
        #     關於產品："讓廣告技術美而溫暖"是復歌的產品理念。在努力為企業客戶創造價值的同時，也希望讓使用復歌產品的每個人都能感受到技術的溫度。
        #     我們關注用戶的體驗和建議，我們期待我們的產品能夠給每個使用者的工作和生活帶來正面的改變。
        #     我們崇尚技術，用科技的力量使工作變得簡單，使生活變得更加美好而優雅，是我們的願景。
        #     企業文化：復歌是一個非常年輕的團隊，公司大部分成員是90後。
        #     工作上，專業、注重細節、擁抱創新、快速試錯。
        #     協作中，開放、坦誠、包容、還有一點點舉重若輕的幽默感。
        #     以上這些都是復歌團隊的重要特質。
        #     在復歌，每個人可以平等地表達自己的觀點和意見，每個人的想法和意願都會被尊重。
        #     如果你有理想，並擁有被理想所驅使的自我驅動力，我們期待你的加入。
        #     """
        #     prompt = CONTEXT_QA_PROMPT.format( context=context, query=query )
        #     return self.llm(prompt = prompt)
    fuge_data_source = FugeDataSource(llm_chat) #初始化


    #### 分為兩段：
    # - 第一段：得到前k筆資料是有價值的(score大於某門檻)
    # - 第二段：讓retrievalQA去搜尋前k筆資料並依據其作出回答
    fubon_question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    Return any relevant text.
    {context}
    Question: {question}
    Relevant text, if any:"""
    FUBON_QUESTION_PROMPT = PromptTemplate(
        template=fubon_question_prompt_template, input_variables=["context", "question"]
    )

    fubon_combine_prompt_template = """你是個專業文檔師，你的任務是在你的回覆中，
    ，保留大部分我給定的資訊，並把段落結合在一起. 

    QUESTION: {question}
    =========
    {summaries}
    =========
    Answer in traditional Chinese:"""
    FUBON_COMBINE_PROMPT = PromptTemplate(
        template=fubon_combine_prompt_template, input_variables=["summaries", "question"]
    )

    class FubonDataSource:
        def __init__(self, llm:OpenAI(temperature=0)):
            self.llm = llm
        def find_doc_above_score(self, query: str) -> str:
            """讓chain知道前幾筆資料是有用的, pass到search.kwarg 因pinecone+langchain不支援同時取score"""
            model = SentenceModel('shibing624/text2vec-base-chinese')
            query_embedd = model.encode(query, convert_to_numpy=True).tolist()
            response = index.query(query_embedd, top_k=2, include_metadata=True)
            #print(response) 
            threshold = 0.60
            above_criterion_cnt = 0
            for data in response['matches']:
                if data['score'] < threshold:
                    break;
                #print(data)
                above_criterion_cnt += 1
            print(f"\nHow many docs match the criterion? {above_criterion_cnt} docs\n")
            return above_criterion_cnt
        def return_doc_summary(self, query: str) -> str:
            k = self.find_doc_above_score(query)
            if k == 0: return '沒有內部相符的文檔'
            data_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, 
                                            chain_type="map_reduce", 
                                            retriever= docsearch.as_retriever(search_kwargs={"k": k}),
                                            chain_type_kwargs = {"verbose": False,
                                                                "question_prompt": FUBON_QUESTION_PROMPT,
                                                                "combine_prompt": FUBON_COMBINE_PROMPT,
                                                                },
                                            return_source_documents=False)
            return data_retriever.run(query)

    retrieval_llm = ChatOpenAI(temperature=0)
    fubon_data_source = FubonDataSource(retrieval_llm) #初始化

    """#### 搜尋數篇網路文章並總結"""

    def sumWebAPI(input_query: str) : 
        '''依照關鍵字搜尋google前n個網址並總結'''
        num_news = 2 # 找前2篇網站
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=10, separators=[" ", ",", "\n", "\n\n", "\t", ""]
        )

        # refine方法的template
        prompt_template = """Write a concise summary about 100 words of the following:

        {text}

        CONCISE SUMMARY IN Chinese:
        """
        web_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        refine_template = (
            """Your job is to produce a final summary so that a reader will have a full understanding of what happened, and provide as much information as possible
            We have provided an existing summary up to a certain point: {existing_answer}
            We have the opportunity to refine the existing summary
            (only if needed) with some more context below.
            Context:
            ------------
            {text}
            ------------
            Given the new context, refine the original summary
            If the context isn't useful, return the original summary.
            The response should be in bullet points but not too short, traditional Chinese"""
        )
        refine_prompt = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )
        # Google search Api params
        params = {
            "q": f"{input_query}",
            "location": "Taiwan",
            "hl": "tw", #國家
            "gl": "us",
            "google_domain": "google.com",
            "api_key": SERP_API_KEY, # your api key 
            "num": f"{num_news}"
        }

        search = GoogleSearch(params)
        search_results = search.get_dict()
        #print('\nGoogleSearch Result: ', search_results)
        menu_items = search_results.get('search_information', {}).get('menu_items', [])
        link_news = [item.get('link') for item in menu_items]
        loader = WebBaseLoader(link_news[:num_news])
        documents = loader.load() #網站都合在一起變成document
        def extract_text(document):
            content = document.page_content.strip()
            content = re.sub(r'[\n\t]+', '', content)
            text = re.sub(r'[^\w\s.]', '', content)
            return text
        
        split_docs = text_splitter.split_documents(documents)
        # 取部分內容作總結就行
        if len(split_docs) >= 10:
            split_docs = split_docs[3:8]
        elif len(split_docs) > 5:
            split_docs = split_docs[1:5]

        for doc in split_docs:
            doc = extract_text(doc)
            print('here is a doc:', doc)
        web_sum_chain = load_summarize_chain(llm_chat, chain_type="refine", question_prompt=web_PROMPT, 
                                             refine_prompt=refine_prompt,verbose=False) #verbose可以看過程   
        result = web_sum_chain.run(split_docs)
        return result
    
    def summarizeYoutubeScript(input_url) : 
        loader = YoutubeLoader.from_youtube_url(input_url, add_video_info=False)
        document= loader.load()
        if not document:
            return "告訴使用這此部youtube影片沒有提供字幕"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=20, separators=[" ", ",", "\n", "\n\n", "\t", ""]
        )
        split_docs = text_splitter.split_documents(document)
        print("\nYour youtube scripts: \n", split_docs)

        map_prompt_template = """Write a concise summary of a long document,  Ignore the grammatical particles and focus only on the substance

        {text}

        CONCISE SUMMARY:"""
        MAP_PROMPT = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt_template = """ You're now a professional youtube watcher, 
        Given the following extracted parts of a youtube transcript, create a final summary around 300 words in Traditional Chinese. 

        =========
        {text}
        =========

        Answer in Traditional Chinese, bullet points: """

        COMBINE_PROMPT = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        yt_chain = load_summarize_chain(ChatOpenAI(temperature=0.2), chain_type="map_reduce",
                                    return_intermediate_steps=False, map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
        summary = yt_chain.run(split_docs[:6])
        return summary
    
    #search = SerpAPIWrapper(params = {'engine': 'google', 'gl': 'us', 'google_domain': 'google.com', 'hl': 'tw'})
    customize_tools = [
        Tool(
            name = '查詢富邦相關資訊',
            func=fubon_data_source.return_doc_summary,
            description="Useful for questions related to Fubon Bank topics to get more precise information,\
            if the user tell you to answer by Knowledge base, you MUST use this tool\
            your action input here must be a single sentence query that correspond to the question"
        ),
        Tool(
            name = "SummarizeWebInformation",
            func=sumWebAPI,
            description="Only use when user ask to search for web affairs, input should be key word"
        ),
        Tool(
            name = "SummarizeYoutubeTranscript",
            func=summarizeYoutubeScript,
            description="Only use when user provide a youtube url and want information about it. input should be exactly the full url"
        ),
        # Tool(
        #     name="查詢復歌科技公司產品名稱",
        #     func=fuge_data_source.find_product_description,
        #     description="通过产品名称找到复歌科技产品描述时用的工具，输入应该是产品名称",
        # ),
        # Tool(
        #     name="復歌科技公司相關信息",
        #     func=fuge_data_source.find_company_info,
        #     description="幫用戶詢問復歌科技公司相关的问题, 可以通过这个工具了解相关信息",
        # )
    ]   
            
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", 
                                            input_key="input", 
                                            output_key='output', return_messages=True)

    my_agent = initialize_agent(
        tools=customize_tools,
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        agent='conversational-react-description',
        verbose=True,
        memory=memory,
        max_iterations=3,
        early_stopping_method='generate',
        handle_parsing_errors="Check your output and make sure it conforms!",
    )

    #https://www.youtube.com/watch?v=q-HNphrWsDE
    agent_prompt_prefix = """
    Assistant is a large language model in 富邦銀行. Always answer question with traditional Chinese, By default, I use a Persuasive, Descriptive style, but if the user has a preferred tone or role, assistant always adjust accordingly to their preference. If a user has specific formatting needs,such as answer in bullet point, they should NEVER be ignored in your responses

    Assistant is designed to be able to assist with a wide range of tasks,It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to questions. 

    Additionally, Assistant is able to generate its own text based on the observation it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and in-depth explanations on a wide range of topics, like programming, summarizing text. 

    Unfortunately, assistant is terrible at current affairs and bank topic, no matter how simple, assistant always refers to it's trusty tools for help and NEVER try to answer the question itself.

    TOOLS:
    ------

    Assistant has access to the following tools:
    """


    agent_prompt_format_instructions = """To use a tool, use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you gathered all the observation and have final response to say to the Human,
    or you do not need to use a tool, YOU MUST follow the format(the prefix of "Thought: " and "{ai_prefix}: " are must be included):
    ```
    {ai_prefix}: [your response]
    ```"""

    agent_prompt_suffix = """Begin! 

    Previous conversation history:
    {chat_history}

    New user question: {input}
    {agent_scratchpad}
    """

    #自己填充的prompt Costco信用卡可以在哪裡繳款，給我詳細資訊
    new_sys_msg = my_agent.agent.create_prompt(
        tools = customize_tools,
        prefix = agent_prompt_prefix,
        format_instructions= agent_prompt_format_instructions,
        suffix = agent_prompt_suffix,
        ai_prefix = "AI",
        human_prefix = "Human"
    ) 
    from langchain.schema import AgentAction, AgentFinish, OutputParserException
    class MyConvoOutputParser(AgentOutputParser):
        ai_prefix: str = "AI"
        def get_format_instructions(self) -> str:
            return agent_prompt_format_instructions

        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            if f"{self.ai_prefix}:" in text:
                return AgentFinish(
                    {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
                )
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, text)
            if not match:
                raise OutputParserException(f"IIIII Could not parse LLM output: `{text}`")
            action = match.group(1)
            action_input = match.group(2)
            return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
        @property
        def _type(self) -> str:
            return "conversational"
        
    my_agent.agent.llm_chain.prompt = new_sys_msg
    #my_agent.agent.llm_chain.prompt.output_parser = MyConvoOutputParser() #沒有連到 改用預設的
    return my_agent

"""
my_agent.run('我叫吳花油')

my_agent.run('我的名字是什麼')

my_agent.run('幫我翻譯這句話成英文：您好，請問何時能夠洽談合作')

my_agent.run('''I have a dataframe, the three columns named 'played_duration', title_id, user_id, I want to know which title_id is the most popular. please add played_duration by title_id and return the title and their sum list''')

my_agent.run('''I want to sort them by their sum, the largest at front, return a list of title_id''')

my_agent.run('我昨天弄丟信用卡了，幫我搜尋補發方法')

"""


