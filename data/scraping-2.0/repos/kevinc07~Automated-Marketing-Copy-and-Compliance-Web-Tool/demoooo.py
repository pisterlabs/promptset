# Commented out IPython magic to ensure Python compatibility.
# always needed
import os
from config import OPEN_API_KEY, SERP_API_KEY
#from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# log and save
from pathlib import Path
# %matplotlib inline
import openai
import re

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import  TextLoader

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
import nltk
nltk.download('punkt')
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
import faiss
from typing import List
import re


from langchain.chat_models import ChatOpenAI
from pathlib import Path

def initialize():
    os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
    os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
    #OpenAI類默認對應 「text-davinci-003」版本：
    #ChatOpenAI類默認是 "gpt-3.5-turbo"版本
    #OpenAI是即將被棄用的方法，最好是用ChatOpenAI
    ## 檢查是否有可用的GPU
    #EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #llm_chat = ChatOpenAI(temperature=0) #GPT-3.5-turbo
    embeddings = OpenAIEmbeddings()
    return embeddings

def process_and_store_documents(file_paths: List[str]) -> None:
    embeddings = OpenAIEmbeddings()
    def init_txt(file_pth: str):
        loader = TextLoader(file_pth)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=50, separators=["。", "\n", "\n\n", "\t"]
        )
        split_docs_txt = text_splitter.split_documents(documents)
        return split_docs_txt
    
    def init_csv(file_pth: str):
        # my_csv_loader = CSVLoader(file_path=f'{file_pth}',encoding="utf-8", 
        #                           csv_args={'delimiter': ','
        # })
        loader = CSVLoader(f'{file_pth}')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
        )
        split_docs_csv = text_splitter.split_documents(documents)
        return split_docs_csv
    
    def init_xlsx(file_pth: str):
        loader = UnstructuredExcelLoader(file_pth,mode="elements")
        documents = loader.load() 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
        )
        split_docs_xlsx = text_splitter.split_documents(documents)
        return split_docs_xlsx
    
    def init_pdf(file_pth: str):
        loader = PyPDFLoader(file_pth)
        documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
        )
        split_docs_pdf = text_splitter.split_documents(documents)        
        return split_docs_pdf

    def init_word(file_pth: str):
        loader = Docx2txtLoader(file_pth)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
        )
        split_docs_word= text_splitter.split_documents(documents)  
        return split_docs_word
    
    def init_ustruc(file_pth: str):
        loader = UnstructuredFileLoader(file_pth)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
        )
        split_docs_ustruc= text_splitter.split_documents(documents)  
        return split_docs_ustruc
    
    doc_chunks = []
    for file_path in file_paths:
        extension = os.path.splitext(file_path)[-1].lower()  # Get the file extension
        if extension == '.txt':
            txt_docs = init_txt(file_path)
            doc_chunks.extend(txt_docs)
        elif extension == '.csv':
            csv_docs = init_csv(file_path)
            doc_chunks.extend(csv_docs)
        elif extension == '.xlsx':
            xlsx_docs = init_xlsx(file_path)
            doc_chunks.extend(xlsx_docs)
        elif extension == '.pdf':
            pdf_docs = init_pdf(file_path)
            doc_chunks.extend(pdf_docs)
        elif extension == '.docx':
            word_docs = init_word(file_path)
            doc_chunks.extend(word_docs)
        else:
            ustruc_docs = init_ustruc(file_path)
            doc_chunks.extend(ustruc_docs)

    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent
    # 在當前目錄下找 "faiss_index" 資料夾
    faiss_index_path = current_script_path / 'faiss_index'
    # 加載FAISS
    docsearch = FAISS.load_local(str(faiss_index_path), embeddings)
    new_metadatas = [doc.metadata for doc in doc_chunks]
    # 提取每個文檔的實際來源
    docsearch.add_texts([t.page_content for t in doc_chunks], metadatas=new_metadatas)
    docsearch.save_local(folder_path=str(faiss_index_path))
"""## 模板 （Agent, tool, chain)

## 定義Tools的集合
"""
def get_my_agent():
    embeddings = OpenAIEmbeddings()
    llm_chat = ChatOpenAI(temperature=0) #GPT-3.5-turbo 
    from langchain import PromptTemplate, OpenAI, LLMChain

    prompt_template = """
    Assistant is a large language model in 富邦銀行. Always answer question with "Traditional Chinese"
    {query}
    """

    llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    return llm_chain

"""
my_agent.run('我叫吳花油')

my_agent.run('我的名字是什麼')

my_agent.run('幫我翻譯這句話成英文：您好，請問何時能夠洽談合作')

my_agent.run('''I have a dataframe, the three columns named 'played_duration', title_id, user_id, I want to know which title_id is the most popular. please add played_duration by title_id and return the title and their sum list''')

my_agent.run('''I want to sort them by their sum, the largest at front, return a list of title_id''')

my_agent.run('我昨天弄丟信用卡了，幫我搜尋補發方法')

"""
# 廣宣

#讀csv檔
def read_csv_file(url):
    data = pd.read_csv(url)
    data["補充資料"] = data["補充資料"].fillna("無")

    check_list = data["自評項目"].values.tolist()
    support_data = data["補充資料"].values.tolist()
    class_list = data["類別"].values.tolist()
    QN_list = data[data.columns[1]].values.tolist()
    return check_list, support_data, class_list, QN_list


#計算不符&不適合數量
def retrieve_output_dic(class_list, output_list):
    retrieve_dic = {}

    for i in class_list:
        retrieve_dic[i] = {"否":0, "不適用":0}
        
    for Class, output in zip(class_list, output_list):
        match = re.search(r"規定[：:](.+)\n", output)
        result = match.group(1).strip()
        if result == "否":
            retrieve_dic[Class]['否'] += 1
        elif result == "不適用":
            retrieve_dic[Class]['不適用'] += 1

    return  retrieve_dic

def spilit_text(output_list):
    ptn_result = re.compile(r"規定[：:](.+)\n")
    ptn_word = re.compile(r"敘述[：:](.+)\n")
    ptn_reason = re.compile(r"原因[：:](.+)")

    result_list = [ptn_result.search(i).group(1) for i in output_list]
    word_list = [ptn_word.search(i).group(1) for i in output_list]
    reason_list = [ptn_reason.search(i).group(1) for i in output_list]

    return result_list, word_list, reason_list


def check_list_chain(prompt, check_list: list, support_data: list):
    openai.api_key = "api_key"
    openai.api_type = "api_type"
    openai.api_base = "api_base"
    openai.api_version = "api_version"

    output_list = []
    #print("1", prompt_text)
    for rule, additional_inf in zip(check_list, support_data):

        prompt_text = prompt.format(rule=rule, additional_inf=additional_inf)

        message=[{"role": "user", "content": prompt_text}]
        response = openai.ChatCompletion.create(
            engine="gpt-4-32k",
            messages = message,
            temperature=0.1,
            #max_tokens=1000,
            frequency_penalty=0.0
        )
        output = response["choices"][0]["message"]["content"]
        print("題目: " + rule + "\n" + output + "\n")
        output_list.append(output)

    result_list = [str(i) + "." + j for i ,j in enumerate(output_list)]  
    return result_list


def onepick_answer(prompt, rule: str, additional_inf: str):
    openai.api_key = "api_key"
    openai.api_type = "api_type"
    openai.api_base = "api_base"
    openai.api_version = "api_version"

    prompt_text = prompt.format(rule=rule, additional_inf=additional_inf)

    message=[{"role": "user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
        engine="gpt-4-32k",
        messages = message,
        temperature=0.1,
        frequency_penalty=0.0
    )
    output = response["choices"][0]["message"]["content"]
    return output




#獲取當前資料夾路徑
#current_script_path = os.getcwd()
#data_path = current_script_path + "\static" + "\data" + "\自評項目(直接回答)- 測試.csv"
#獲取資料庫資料
#check_list, support_data, class_list, QN_list = read_csv_file(r"C:\Fubon_intern\廣宣專案\frontend\static\data\自評項目(直接回答)- 測試.csv")
#output_list = check_list_chain_test(check_list, support_data)

#a = retrieve_output_dic(class_list, example)
#print(a)
#match = re.search(r"是否符合規定：(.*)\n", example[0])
#ptn = re.compile(r"是否符合規定：(.)")
#result = ptn.search("1.是否符合規定：是").group()
#for i in example:
    #result = re.search(r"是否符合規定：(.+)\n", i).group(1)
    #print(result)


#check_list, support_data, class_list, QN_list = read_csv_file(r"C:\Fubon_intern\廣宣專案\data\自評項目(直接回答)- 測試.csv")
#output_list = check_list_chain_test(check_list, support_data)
#print(output_list)

#result_list, word_list, reason_list = spilit_text(example)
#ptn_reason = re.compile(r"不符合的原因[：:](.+)")
#print(example[0])
#print(ptn_reason.search(example[0]).group(1))
#print(reason_list)

