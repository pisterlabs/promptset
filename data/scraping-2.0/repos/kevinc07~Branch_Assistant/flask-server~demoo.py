# Commented out IPython magic to ensure Python compatibility.
# always needed
import json
import os
import numpy as np
import faiss
import openai
#from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# log and save
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import  TextLoader

from langchain.agents import initialize_agent

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
import nltk
nltk.download('punkt')
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from typing import List, Union
import re
from langchain import LLMChain
from langchain.agents import Tool, AgentOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory ,ReadOnlySharedMemory

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from pathlib import Path
from sql import id_retrieval, max_id, id_QQQ, id_QA, source_search

max = max_id()

def initialize():
    
    #定義api key
    #os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
    #os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
    os.environ["OPENAI_API_TYPE"] = "OPENAI_API_TYPE"
    os.environ["OPENAI_API_VERSION"] = "OPENAI_API_VERSION"
    os.environ["OPENAI_API_BASE"] = "OPENAI_API_BASE"
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

    #定義llmAzure參數
    global llmAzure 
    
    llmAzure = AzureChatOpenAI(
        openai_api_version="openai_api_version",
        deployment_name="deployment_name",
        model_name="model_name"
    )
    #OpenAI類默認對應 「text-davinci-003」版本：
    #ChatOpenAI類默認是 "gpt-3.5-turbo"版本
    #OpenAI是即將被棄用的方法，最好是用ChatOpenAI
    # 檢查是否有可用的GPU
    #EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #llm_chat = ChatOpenAI(temperature=0) #GPT-3.5-turbo
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)

    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent

    # 在當前目錄下檢查 "faiss_index" 資料夾是否存在
    faiss_index_path = current_script_path / 'faiss_index'
    faiss_index_path.mkdir(parents=True, exist_ok=True)

    # 檢查 "index.faiss" 文件是否存在
    index_path = faiss_index_path / 'index.faiss'
    if not index_path.exists():
        texts = ['這是一個測試文本', '這是另一個測試文本']
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(folder_path=str(faiss_index_path))
    # 讀取已經創建的向量數據庫
    index = faiss.read_index(str(index_path))
    # 獲取向量數據庫中的向量數量
    num_vectors = index.ntotal
    print("向量數據庫中的向量數量：", num_vectors)
    return embeddings
def process_and_store_documents(file_paths: List[str]) -> None:
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)
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
    #過濾已存在的資料
    new_doc_chunk = [] 
    for i in doc_chunks:
        similarity = docsearch.similarity_search_with_score(i.page_content)[0][1]
        if similarity >= 0.001:
            new_doc_chunk.append(i)
    new_clean_metadatas = [doc.metadata for doc in new_doc_chunk]
    # 提取每個文檔的實際來源
    docsearch.add_texts([t.page_content for t in new_doc_chunk], metadatas=new_clean_metadatas)
    docsearch.save_local(folder_path=str(faiss_index_path))
"""## 模板 （Agent, tool, chain)

## 定義Tools的集合
"""



def get_my_agent(response_length=None):
    print(response_length)
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1) 

    global llmAzure 

    # 讓retrievalQA去搜尋前k筆資料並依據其作出回答
    fubon_question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    Return any relevant text.
    {context}
    Question: {question}
    Relevant text, if any:"""
    FUBON_QUESTION_PROMPT = PromptTemplate(
        template=fubon_question_prompt_template, input_variables=["context", "question"]
    )

    # 根據 response_length 參數調整 prompt
    if response_length == 'long':
        output_length_text = "將你取出的3個來源內的最相關的前80%，輸出總字數至少為300個字或以上"
    elif response_length == 'medium':
        output_length_text = "將你取出的3個來源內的最相關的前50%，輸出總字數至少為150個字或以上"
    elif response_length == 'short':
        output_length_text = "將你取出的3個來源內的最相關的前30%，輸出總字數至少為50個字或以上"
    else:
        # 預設值
        output_length_text = "將你取出的3個來源內的最相關的前50%，輸出總字數至少為150個字或以上"


    fubon_combine_prompt_template = f"""你是個專業知識歸納師，你的任務是詳細且完整的回覆客戶的問題，
    如果我給的資訊有高度相關職的準確資訊，未提及「外國或國外」，一律視為「本國或台灣」；
    回答結果先分類再以條列示方式顯示，{output_length_text}；
    若找不到高度相關職的準確資訊，但知識庫內有提供電話或[HTTP網址]放在[點我連結]內，一起提供；如果提取的重要內容有包含網址請一併列出，
    若都沒有，請回覆「無法在富邦文件庫內找到您的問題」。

    QUESTION: {{question}}
    =========
    {{summaries}}
    =========
    請忽略以上所有不相關的text，針對有用的資訊回答就好，{output_length_text}；
    Answer in traditional Chinese:
    """
    FUBON_COMBINE_PROMPT = PromptTemplate(
        template=fubon_combine_prompt_template, input_variables=["summaries", "question"]
    )
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)

    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent
    # 在當前目錄下找 "faiss_index" 資料夾
    faiss_index_path = current_script_path / 'faiss_index'
    class FubonQA:
        def __init__(self, llmAzure):
            self.llmAzure = llmAzure
        def return_doc_summary(self, query: str) -> str:
            k = 3
            # 加載FAISS
            docsearch = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_QA")
            similarity = docsearch.similarity_search(query, k=3)
            
            global similarity_QA
            global similarity_QA_source 
            similarity_QA = [i.page_content for i in similarity]
            similarity_QA_source = [i.metadata for i in similarity]
            
            data_retriever = RetrievalQA.from_chain_type(self.llmAzure, chain_type="map_reduce", memory=readonlymemory,
                                            retriever=docsearch.as_retriever(search_kwargs={"k": k}),
                                            chain_type_kwargs = {"verbose": True,
                                                                 "question_prompt": FUBON_QUESTION_PROMPT, #注意是question_prompt
                                                                 "combine_prompt": FUBON_COMBINE_PROMPT,
                                                })
            #return data_retriever({"question": query}, return_only_outputs=True)
            return data_retriever.run(query)
    retrieval_llm = llmAzure
    fubon_QAsource = FubonQA(retrieval_llm) #初始化



    def synonyms_qa(query: str):
        global llmAzure 
        embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)

        # 獲取當前 Python 腳本的絕對路徑
        current_script_path = Path(__file__).resolve().parent
        # 在當前目錄下找 "faiss_index" 資料夾
        faiss_index_path = current_script_path / 'faiss_index'

        prompt_template = """你現在是一個文本過濾器，你的工作不是回答問題，而是幫我們改善文本，
        請根據提供的文本幫我把問題裡出現的所有同義詞(或與其相似的詞)替換成與其相似的所有集合詞，並輸出替換後的問題(只需要原本的問題)
        ============
        問題:{question}
        文本:
        {context}
        =============
        只需要回傳問題本身
        """
        question_prompt = PromptTemplate(
                template=prompt_template, input_variables=["question","context"]
        )
        docsearch0 = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_Synonym")
        qa = RetrievalQA.from_chain_type(ChatOpenAI(engine="deployment_name"), chain_type="stuff", retriever=docsearch0.as_retriever(),
                                        chain_type_kwargs = {"verbose": True,
                                                            "prompt": question_prompt}) 
        result = qa.run(query)
        

        fubon_question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
        Return any relevant text. 
        {context}
        Question: {question}
        Relevant text, if any:
        """
        FUBON_QUESTION_PROMPT = PromptTemplate(
            template=fubon_question_prompt_template, input_variables=["context", "question"]
        )

        fubon_combine_prompt_template = f"""你是個專業知識歸納師，你的任務是詳細且完整的回覆客戶的問題，
        如果我給的資訊有高度相關職的準確資訊，未提及「外國或國外」，一律視為「本國或台灣」；
        回答結果先分類再以條列示方式顯示，{output_length_text}；
        若找不到高度相關職的準確資訊，但知識庫內有提供電話或[HTTP網址]放在[點我連結]內，一起提供；如果提取的重要內容有包含網址請一併列出，
        若都沒有，請回覆「無法在富邦文件庫內找到您的問題」。

        QUESTION: {{question}}
        =========
        {{summaries}}
        =========
        請忽略以上所有不相關的text，針對有用的資訊回答就好，{output_length_text}；
        Answer in traditional Chinese:
        """
        FUBON_COMBINE_PROMPT = PromptTemplate(
            template=fubon_combine_prompt_template, input_variables=["summaries", "question"]
        )
        docsearch = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_QA")
        similarity = docsearch.similarity_search(query, k=3)

        global similarity_QA
        global similarity_QA_source 
        similarity_QA = [i.page_content for i in similarity]
        similarity_QA_source = [i.metadata for i in similarity]

        data_retriever = RetrievalQA.from_chain_type(ChatOpenAI(engine="deployment_name"), chain_type="map_reduce",
                                                    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
                                                    chain_type_kwargs = {"verbose": True,
                                                                        "question_prompt": FUBON_QUESTION_PROMPT,
                                                                        "combine_prompt": FUBON_COMBINE_PROMPT})
        output = data_retriever.run(result)
        output = re.sub(r"\n+","", output)
        return output
    


        ######################
    def docsearch(query: str):
        global llmAzure
        embeddings = OpenAIEmbeddings(deployment_name="deployment_name",)

        # 獲取當前 Python 腳本的絕對路徑
        current_script_path = Path(__file__).resolve().parent
        # 在當前目錄下找 "faiss_index" 資料夾
        pd.set_option('display.max_colwidth', None)

        global similarity_QA
        global similarity_QA_source 
        similarity_QA = []
        similarity_QA_source = []
        id_list = []
        doc_result = embeddings.embed_documents([query])
        print(doc_result)
        doc_result_np = np.array(doc_result).astype('float32')
        # 加載您的 FAISS 索引
        index = faiss.read_index(r'C:\Users\ASUS\Downloads\Langchain_Faiss\Langchain_Faiss\Langchain_Faiss\flask-server\faiss_index\index.faiss')
        # 搜索最近鄰居的數量
        k = 3
        # 執行搜索
        D, I = index.search(doc_result_np, k)
        print("distance", D)
        print("index", I)
        
        for id_value in I[0]:
            # 将 numpy.int64 转换为 Python 的 int
            id = int(id_value)
            print(id)

            # similarity_QA_source.append(doc.metadata) #這邊改成從id抓資料庫的檔名
            before_row_content = id_retrieval(id-1) if id -1 >= 0 else None
            target_row_content = id_retrieval(id) 
            after_row_content = id_retrieval(id+1) if id +1< max  else None
            print(before_row_content)
            # 輸出結果
            #print(target_row_content)
            combined_content = "\n\n".join(filter(None, [before_row_content, target_row_content, after_row_content]))
            similarity_QA.append(combined_content)
            similarity_QA_source.append(source_search(id))

            id_list.extend([id, id - 1 if id -1 >= 0 else None, id + 1])
            id_list.sort()
            id_list_sorted = list(set(id_list))

        print(id_list_sorted)
        grouped = []
        previous = id_list_sorted[0]  # 初始化previous為列表的第一個值
        target_row_content = id_retrieval(previous)
        combined_content = "\n\n".join(filter(None, [target_row_content]))
        temp_group = [combined_content]

        for i in id_list_sorted[1:]:
            if i == previous + 1:  # 檢查當前值是否是前一個值加1
                print(i)
                id_retrieval(i)
                combined_content = "\n\n".join(filter(None, [target_row_content]))
                temp_group.append(combined_content)
            else:
                grouped.append(','.join(temp_group))
                id_retrieval(i)
                combined_content = "\n\n".join(filter(None, [target_row_content]))
                temp_group = [combined_content]
            previous = i  # 更新previous為當前值
                
        grouped.append(','.join(temp_group))

        answer = []
        for combined_content in grouped :
            #for doc, score in similarity:
            question_template ="""
            Use the following portion of a long document to see if any of the text is relevant to answer the question. 
            Return any relevant text.
            {content}
            Question: {title}
            Relevant text, if any:"""
            prompt_template = PromptTemplate(input_variables=["title", "content"], template=question_template)
            first_chain = LLMChain(llm=llmAzure, prompt=prompt_template, verbose = True)
            out = first_chain.run({"title": query ,"content": combined_content})
            #print("OUT:" + out)
            #print("length:" + str(len(out)))
            answer.append(out)
        #print("ANSWER:")
        summaries = ""
        for a in answer:
            summaries +=a
        #print(summaries)
        combine_prompt_template = f"""你是個專業知識歸納師，你的任務是詳細且完整的回覆客戶的問題，
        如果我給的資訊有高度相關職的準確資訊，未提及「外國或國外」，一律視為「本國或台灣」；
        回答結果先分類再以條列示方式顯示，{output_length_text}；
        若找不到高度相關職的準確資訊，但知識庫內有提供電話或[HTTP網址]放在[點我連結]內，一起提供；如果提取的重要內容有包含網址請一併列出，
        若都沒有，請回覆「無法在富邦文件庫內找到您的問題」。

        QUESTION: {{title}}
        =========
        {{summaries}}
        =========
        請忽略以上所有不相關的text，針對有用的資訊回答就好，{output_length_text}；
        Answer in traditional Chinese:
        """
        prompt_template = PromptTemplate(input_variables=["title", "summaries"], template=combine_prompt_template)
        second_chain = LLMChain(llm=llmAzure, prompt=prompt_template, verbose = True)
        out = second_chain.run({"title": query, "summaries": summaries})
        return out


    def synonyms_docsearch(query: str):
            global llmAzure
            embeddings = OpenAIEmbeddings(deployment_name="deployment_name",)

            # 獲取當前 Python 腳本的絕對路徑
            current_script_path = Path(__file__).resolve().parent
            # 在當前目錄下找 "faiss_index" 資料夾
            faiss_index_path = current_script_path / 'faiss_index'

            prompt_template = """你現在是一個文本過濾器，你的工作不是回答問題，而是幫我們改善文本，
            請根據提供的文本幫我把問題裡出現的所有同義詞(或與其相似的詞)替換成與其相似的所有集合詞，並輸出替換後的問題(只需要原本的問題)
            ============
            問題:{question}
            文本:
            {context}
            =============
            只需要回傳問題本身
            """
            question_prompt = PromptTemplate(
                    template=prompt_template, input_variables=["question","context"]
            )
            docsearch0 = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_Synonym")
            qa = RetrievalQA.from_chain_type(ChatOpenAI(engine="deployment_name"), chain_type="stuff", retriever=docsearch0.as_retriever(),
                                            chain_type_kwargs = {"verbose": True,
                                                                "prompt": question_prompt}) 
            result = qa.run(query)

            #####################

            embeddings = OpenAIEmbeddings(deployment_name = 'deployment_name', chunk_size=1)
            pd.set_option('display.max_colwidth', None)
            global similarity_QA
            global similarity_QA_source
            similarity_QA = []
            similarity_QA_source = []
            id_list = []
            doc_result = embeddings.embed_documents([query])
            print(doc_result)
            doc_result_np = np.array(doc_result).astype('float32')
            # 加載您的 FAISS 索引
            index = faiss.read_index(r'C:\Users\ASUS\Downloads\Langchain_Faiss\Langchain_Faiss\Langchain_Faiss\flask-server\faiss_index\index.faiss')
            # 搜索最近鄰居的數量
            k = 3
            # 執行搜索
            D, I = index.search(doc_result_np, k)

            for id_value in I[0]:
                # 将 numpy.int64 转换为 Python 的 int
                id = int(id_value)
                print(id)

                #similarity_QA_source.append(doc.metadata) #這邊改成從id抓資料庫的檔名
                before_row_content = id_retrieval(id-1) if id -1 >= 0 else None
                target_row_content = id_retrieval(id) 
                after_row_content = id_retrieval(id+1) if id +1< max  else None
                print(before_row_content)
                # 輸出結果
                #print(target_row_content)
                combined_content = "\n\n".join(filter(None, [before_row_content, target_row_content, after_row_content]))
                similarity_QA.append(combined_content)
                similarity_QA_source.append(source_search(id))

                id_list.extend([id, id - 1 if id -1 >= 0 else None, id + 1])
                id_list.sort()
                id_list_sorted = list(set(id_list))
                
                print(id_list_sorted)
                grouped = []
                previous = id_list_sorted[0]  # 初始化previous為列表的第一個值
                target_row_content = id_retrieval(previous)
                combined_content = "\n\n".join(filter(None, [target_row_content]))
                temp_group = [combined_content]
                    
                for i in id_list_sorted[1:]:
                    if i == previous + 1:  # 檢查當前值是否是前一個值加1
                        print(i)
                        id_retrieval(i)
                        combined_content = "\n\n".join(filter(None, [target_row_content]))
                        temp_group.append(combined_content)
                    else:
                        grouped.append(','.join(temp_group))
                        id_retrieval(i)
                        combined_content = "\n\n".join(filter(None, [target_row_content]))
                        temp_group = [combined_content]
                    previous = i  # 更新previous為當前值
                    
                grouped.append(','.join(temp_group))
                answer = []
                    
                for combined_content in grouped :
                    #for doc, score in similarity:
                    question_template ="""
                    Use the following portion of a long document to see if any of the text is relevant to answer the question. 
                    Return any relevant text.
                    {content}
                    Question: {title}
                    Relevant text, if any:"""
                    prompt_template = PromptTemplate(input_variables=["title", "content"], template=question_template)
                    first_chain = LLMChain(llm=llmAzure, prompt=prompt_template, verbose = True)
                    out = first_chain.run({"title": result, "content": combined_content})
                    #print("OUT:" + out)
                    # #print("length:" + str(len(out)))
                answer.append(out)
                #print("ANSWER:")
                summaries = ""
                for a in answer:
                    summaries +=a
                    #print(summaries)
                    combine_prompt_template = f"""你是個專業知識歸納師，你的任務是詳細且完整的回覆客戶的問題，
                    如果我給的資訊有高度相關職的準確資訊，未提及「外國或國外」，一律視為「本國或台灣」；
                    回答結果先分類再以條列示方式顯示，{output_length_text}；
                    若找不到高度相關職的準確資訊，但知識庫內有提供電話或[HTTP網址]放在[點我連結]內，一起提供；如果提取的重要內容有包含網址請一併列出，
                    若都沒有，請回覆「無法在富邦文件庫內找到您的問題」。

                    QUESTION: {{title}}
                    =========
                    {{summaries}}
                    =========
                    請忽略以上所有不相關的text，針對有用的資訊回答就好，{output_length_text}；
                    Answer in traditional Chinese:
                    """
                    prompt_template = PromptTemplate(input_variables=["title", "summaries"], template=combine_prompt_template)
                    second_chain = LLMChain(llm=llmAzure, prompt=prompt_template, verbose = True)
                    out = second_chain.run({"title": result, "summaries": summaries})
                    return out
    


    customize_tools = [
        Tool(
            name = 'QAsearch_First',
            func=fubon_QAsource.return_doc_summary,
            description= """
            Useful for questions related to FUBON BANK topics that you upload to get more precise information,\
            if the user tell you to answer by Knowledge base, you MUST use this tool\
            your action input 是完整的 user input question\
            Only use when user told you to use QAsearch_First.\
            """
        ),
        Tool(
            name="QAsearch_second",
            func=synonyms_qa,
            description="ONLY use when you did not found any useful observation from QAsearch_First then use this tool for more information.",
        ),
        Tool(
            name="DOCsearch_only",
            func=docsearch,
            description="if you did not found any useful observation from QAsearch_second then use this tool for more information.\
            Or when user told you to do DOCsearch_only you MUST USE THIS TOOL."
        ),
        Tool(
            name="DOCsearch_synonym",
            func=synonyms_docsearch,
            description="if the observation from DOCsearch_only is insufficient then use this tool for more information."
        )
    ]   
   
    memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", 
                                            input_key="input", 
                                            output_key='output', return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    def _handle_error(error) -> str:
       return str(error)[:150]
    

    llmAzure = AzureChatOpenAI(
        openai_api_version="openai_api_version",
        deployment_name="deployment_name",
        model_name="model_name"
    )

    my_agent = initialize_agent(
        tools=customize_tools,
        llm =llmAzure,
        agent='conversational-react-description',
        verbose=True,
        memory=memory,
        max_iterations=4,
        early_stopping_method='generate',
        handle_parsing_errors=_handle_error
    )
  
    #https://www.youtube.com/watch?v=q-HNphrWsDE
    agent_prompt_prefix = """
    Assistant is a large language model in 富邦銀行. Always answer question with "Traditional Chinese", By default, I use a Persuasive, Descriptive style, but if the user has a preferred tone or role, assistant always adjust accordingly to their preference. If a user has specific requirements, (such as formatting needs, answer in bullet point) they should NEVER be ignored in your responses

    Assistant is designed to be able to assist with a wide range of tasks,It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to questions. 
    Additionally, Assistant is able to generate its own text based on the observation it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and in-depth explanations on a wide range of topics, like programming, summarizing. 

    Unfortunately, assistant is terrible at current affairs and bank topic, no matter how simple, assistant always refers to it's trusty tools for help and NEVER try to answer the question itself.
    Re-emphasizing, you must respond in "Traditional Chinese" and Don't forget to provide 來源路徑 if the observation from the tool includes one.
    TOOLS:
    ------

    Assistant has access to the following tools:
    """


    agent_prompt_format_instructions = """To use a tool, use the following format:

    ```
    Thought: Do I need to use a tool(DO NOT USE THE SAME TOOL that has been used)? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you gathered all the observation and have final response to say to the Human,
    or you do not need to use a tool, YOU MUST follow the format(the prefix of "Thought: " and "{ai_prefix}: " are must be included):
    ```
    Thought: Do I need to use a tool? No
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
    my_agent.agent.llm_chain.prompt.output_parser = MyConvoOutputParser() #沒有連到 改用預設的
    return my_agent


def flowchart(result: str):
    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent
    # 在當前目錄下找 "faiss_index" 資料夾
    faiss_index_path = current_script_path / 'faiss_index'
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)
    #流程圖向量檢索
    docsearch_flowchart = FAISS.load_local(faiss_index_path, embeddings, index_name = "flowchart")
    flowchart = docsearch_flowchart.similarity_search(result, k=1)[0].page_content
    ptn = re.compile(r'https:\/\/drive\.google\.com\/file\/d\/[a-zA-Z0-9_-]+\/view\?usp=sharing')
    flowchart_url = ptn.search(flowchart).group()
    return flowchart_url


def QQQ(input: str):
    embeddings = OpenAIEmbeddings(deployment_name='deployment_name', chunk_size=1)

    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent
    # 在當前目錄下找 "faiss_index" 資料夾
    faiss_index_path = current_script_path / 'faiss_index'
    docsearch_Q = FAISS.load_local(faiss_index_path, embeddings, index_name = "Q")
    
    similarity = docsearch_Q.similarity_search_with_score(input, k =3)
    pd.set_option('display.max_colwidth', None)

    

    embeddings = OpenAIEmbeddings(deployment_name="deployment_name",)
    doc_result = embeddings.embed_documents([input])
    doc_result_np = np.array(doc_result).astype('float32')
    # 加載您的 FAISS 索引
    index = faiss.read_index(r'C:\Users\ASUS\Downloads\Langchain_Faiss\Langchain_Faiss\Langchain_Faiss\flask-server\faiss_index\Q.faiss')
    # 搜索最近鄰居的數量
    k = 3
    # 執行搜索
    D, I = index.search(doc_result_np, k)
    
    # 建立一個空的字串，用來儲存最終的結果
    result_str = ""

    for id_value in I[0]:
        id = int(id_value)
        row = id_QQQ(id)

        if row:
            values_list = list(row.values())
            idFromQ = values_list[0]  # 獲取第二个字段的值

            searchQA = id_QA(idFromQ)
            if searchQA:
                # 特殊處理 page_content 和 metadata 字段
                if 'page_content' in searchQA:
                    result_str += f"內容：{searchQA['page_content']}\n"

                if 'metadata' in searchQA:
                    try:
                        metadata = json.loads(searchQA['metadata'])
                        if 'source' in metadata:
                            result_str += f"來源：{metadata['source']}\n"
                    except json.JSONDecodeError:
                        print("metadata 格式错误")

                result_str += "--------------------------------\n\n"

            else:
                print(f"在 indexqa 中未找到 ID {idFromQ} 的紀錄。")
        else:
            print(f"ID {id} 沒有找到對應的紀錄。")

    print(result_str)


    # for item in similarity:
    #     doc, score = item
    #     source = doc.metadata['source']
    #     new_path =  source.replace("_Q", "")
    #     row = doc.metadata['row']
    #     df = pd.read_csv(new_path)

    #     row_index = row
    #     target_row_content = df.iloc[row_index]  

    #     # 使用 iteritems() 來遍歷該行的每一列
    #     for col_name, value in target_row_content.items():
    #         # 將列名和值組合成一個字串，然後追加到結果字串中
    #         result_str += f"{col_name}:{value}\t\n"
            
    #     result_str += "--------------------------------\n\n"

    return result_str




'''
initialize()
model = get_my_agent()
result = model.run("受理開戶及更換負責人應注意事項 firstly, try DOCsearch_only if there are not engough information, DOCsearch_synonym")
source_text = similarity_QA
source_doc = similarity_QA_source
source = [i + "\n\n" + json.dumps(j) for i, j in zip(source_text, source_doc)]
print(source)
'''
"視障人士開戶需要見證人嗎? firstly, try DOCsearch_only if there are not engough information, DOCsearch_synonym"
"""
my_agent.run('我叫吳花油')

my_agent.run('我的名字是什麼')

my_agent.run('幫我翻譯這句話成英文：您好，請問何時能夠洽談合作')

my_agent.run('''I have a dataframe, the three columns named 'played_duration', title_id, user_id, I want to know which title_id is the most popular. please add played_duration by title_id and return the title and their sum list''')

my_agent.run('''I want to sort them by their sum, the largest at front, return a list of title_id''')

my_agent.run('我昨天弄丟信用卡了，幫我搜尋補發方法')

"""
"""
llm_chat, embeddings = initialize()
process_and_store_documents(['/Users/kevin/Desktop/python_code/測試文件/測試.txt'])
current_script_path = Path(__file__).resolve().parent
faiss_index_path = current_script_path / 'faiss_index'
index_path = faiss_index_path / 'index.faiss'
index = faiss.read_index(str(index_path))
# 獲取向量數據庫中的向量數量
num_vectors = index.ntotal
print("向量數據庫中的向量數量：", num_vectors)
my_agent = get_my_agent()
my_agent.run('我叫吳花油')
"""
''''
llm_chat, embeddings = initialize()
my_agent = get_my_agent()
my_agent.run("""
:閱讀以下文本後，根據我上傳的文檔，判斷該文本製在作設計時，是否觸犯「銀行辦理財富管理及金融商品銷售業務自律規範」，如有不符合法規之處請列出，並列出觸犯的法條是哪一條。 輸出格式如下： 文本所有不符合之處： -->觸犯的法條：（"章節＿：第＿條"）
---
投資警語
奈米投注意事項
1. 信託財產之管理運用並非絕無風險，本行以往之經理績效不保證指定信託運用信託財產之最低收益；本行除盡善良管理人之注意義務外，不負責指定運用信託財產之盈虧，亦不保證最低之收益，委託人簽約前應詳閱「奈米投指定營運範圍或方法單獨管理運用金錢信託投資國內外有價證券信託契約條款」(下稱「奈米投約定條款」)。')
""")
'''
# embeddings = initialize()
# def synonyms_docsearch(query: str):
#         embeddings = OpenAIEmbeddings(deployment_name='gpt-4-32k', chunk_size=1)

#         # 獲取當前 Python 腳本的絕對路徑
#         current_script_path = Path(__file__).resolve().parent
#         # 在當前目錄下找 "faiss_index" 資料夾
#         faiss_index_path = current_script_path / 'faiss_index'

#         prompt_template = """你現在是一個文本過濾器，你的工作不是回答問題，而是幫我們改善文本，
#         請根據提供的文本幫我把問題裡出現的所有同義詞(或與其相似的詞)替換成與其相似的所有集合詞，並輸出替換後的問題(只需要原本的問題)
#         ============
#         問題:{question}
#         文本:
#         {context}
#         =============
#         只需要回傳問題本身
#         """
#         question_prompt = PromptTemplate(
#                 template=prompt_template, input_variables=["question","context"]
#         )
#         docsearch0 = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_Synonym")
#         qa = RetrievalQA.from_chain_type(ChatOpenAI(model="gpt-4"), chain_type="stuff", retriever=docsearch0.as_retriever(),
#                                         chain_type_kwargs = {"verbose": True,
#                                                             "prompt": question_prompt}) 
#         result = qa.run(query)

#         #####################

#         embeddings = OpenAIEmbeddings()
#         docsearch = FAISS.load_local(faiss_index_path, embeddings)
#         similarity = docsearch.similarity_search_with_score(query, k =3)
#         pd.set_option('display.max_colwidth', None)
#         global similarity_QA
#         global similarity_QA_source 
#         similarity_QA = []
#         similarity_QA_source = []
#         id_list = []
#         openai.api_key = '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'
#         os.environ["OPENAI_API_KEY"] = openai.api_key
#         response = openai.Embedding.create(
#         input=query,
#         model="text-embedding-ada-002"
#         )
#         embeddings = response['data'][0]['embedding']

#         # 將列表轉換為 NumPy 陣列並重塑為 (1, 1536)
#         embeddings_array = np.array([embeddings]).astype('float32')
#         # 加載您的 FAISS 索引
#         index = faiss.read_index(r'C:\Users\sa\Desktop\files\Langchain_Faiss\Langchain_Faiss\flask-server\faiss_index\index.faiss')
#         # 搜索最近鄰居的數量
#         k = 3
#         # 執行搜索
#         D, I = index.search(embeddings_array, k)
#         print("distance", D)
#         print("index", I)

#         for id_value in I[0]:
#             # 将 numpy.int64 转换为 Python 的 int
#             id = int(id_value)
#             print(id)

#             # similarity_QA_source.append(doc.metadata) #這邊改成從id抓資料庫的檔名
#             before_row_content = id_retrieval(id-1) if id -1 >= 0 else None
#             target_row_content = id_retrieval(id) 
#             after_row_content = id_retrieval(id+1) if id +1< max  else None
#             print(before_row_content)
#             # 輸出結果
#             #print(target_row_content)
#             combined_content = "\n\n".join(filter(None, [before_row_content, target_row_content, after_row_content]))
#             similarity_QA.append(combined_content)

#             id_list.extend([id, id - 1 if id -1 >= 0 else None, id + 1])
#             id_list.sort()
#             id_list_sorted = list(set(id_list))

#         print(id_list_sorted)
#         grouped = []
#         previous = id_list_sorted[0]  # 初始化previous為列表的第一個值
#         target_row_content = id_retrieval(previous)
#         combined_content = "\n\n".join(filter(None, [target_row_content]))
#         temp_group = [combined_content]

#         for i in id_list_sorted[1:]:
#             if i == previous + 1:  # 檢查當前值是否是前一個值加1
#                 print(i)
#                 id_retrieval(i)
#                 combined_content = "\n\n".join(filter(None, [target_row_content]))
#                 temp_group.append(combined_content)
#             else:
#                 grouped.append(','.join(temp_group))
#                 id_retrieval(i)
#                 combined_content = "\n\n".join(filter(None, [target_row_content]))
#                 temp_group = [combined_content]
#             previous = i  # 更新previous為當前值
                
#         grouped.append(','.join(temp_group))

#         answer = []
#         for combined_content in grouped :
#             #for doc, score in similarity:
#             question_template ="""
#             Use the following portion of a long document to see if any of the text is relevant to answer the question. 
#             Return any relevant text.
#             {content}
#             Question: {title}
#             Relevant text, if any:"""
#             prompt_template = PromptTemplate(input_variables=["title", "content"], template=question_template)
#             first_chain = LLMChain(llm=llmAzure, prompt=prompt_template, verbose = True)
#             out = first_chain.run({"title": result, "content": combined_content})
#             #print("OUT:" + out)
#             #print("length:" + str(len(out)))
#             answer.append(out)
#         #print("ANSWER:")
#         summaries = ""
#         for a in answer:
#             summaries +=a
#         #print(summaries)
#         combine_prompt_template = """你是個專業知識歸納師，你的任務是在你的回覆中，
#         如果我給的資訊有高度相關職的準確資訊，未提及「外國或國外」，一律視為「本國或台灣」；
#         回答結果先分類再以條列示方式顯示，但一定要保留重要訊息或注意事項；
#         若找不到高度相關職的準確資訊，但知識庫內有提供電話或[HTTP網址]放在[點我連結]內，一起提供；
#         若都沒有，請回覆「無法在富邦文件庫內找到您的問題」。

#         QUESTION: {title}
#         =========
#         {summaries}
#         =========
#         請忽略以上所有不相關的text，針對有用的資訊回答就好
#         Answer in traditional Chinese:"""
#         prompt_template = PromptTemplate(input_variables=["title", "summaries"], template=combine_prompt_template)
#         second_chain = LLMChain(llm=llmAzure, prompt=prompt_template, verbose = True)
#         out = second_chain.run({"title": result, "summaries": summaries})
#         return out
