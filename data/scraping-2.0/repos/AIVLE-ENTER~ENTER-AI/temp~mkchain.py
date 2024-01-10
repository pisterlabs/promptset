import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import format_document
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel
from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os.path
import pickle
import requests
# from bs4 import BeautifulSoup
import pandas as pd
import datetime
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import os
api_key = os.getenv('OPEN_API_KEY')
# class Chain_f:
#     def __init__(self,keyword,ak):
#         self.vectorstore = FAISS.load_local(keyword, embeddings=OpenAIEmbeddings(openai_api_key=ak))
#         self.memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
def make_chain(keyword,history):
    ak = api_key
    #stream_it = AsyncIteratorCallbackHandler()
    
    
    llm = ChatOpenAI(openai_api_key=ak, temperature=0)
    vectorstore = FAISS.load_local('/home/wsl_han/aivle_project/remote/ENTER-AI/'+keyword, embeddings=OpenAIEmbeddings(openai_api_key=ak))
    retriever = vectorstore.as_retriever()#(search_kwargs={"k": 50})

    # llm = LlamaCpp(
    #     model_path="../llachat.bin",
    #     temperature=0,
    #     max_tokens=2000,
    #     top_p=1,
    #     #callback_manager=callback_manager,
    #     #verbose=True,  # Verbose is required to pass to the callback manager
    # )

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm,
    )

    # chat_history와 현재 question을 이용해 질문 생성하는 템플릿
    _template = """Given the following conversation and a follow up Input, rephrase the follow up Input to be a standalone Input, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


    # context를 참조해 한국어로 질문에 답변하는 템플릿
    template = """Guess the answer in korean about the question by referring the following context,:{context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)


    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


    def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    if os.path.isfile(history):
        with open('m.txt','rb') as f:
            memory = pickle.load(f)
    else:
        memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
 
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(openai_api_key=ak,temperature=0)
        | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever_from_llm,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(openai_api_key=ak, )#streaming=True,callbacks=[stream_it]),    #streaming?
        #"docs": itemgetter("docs"),
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    
    return final_chain,memory

# def crawl(topic):
#     ak = "**********************"
#     d = {'comment':[],'title':[],'date':[]}
#     for i in range(1,101):
#         url = f"https://www.fmkorea.com/search.php?act=IS&is_keyword={topic}&mid=home&where=comment&page={i}"
#         response = requests.get(url)
#         dom=BeautifulSoup(response.text,"html.parser")
#     #     if datetime.datetime.strptime(dom.select_one(".time").text,'%Y-%m-%d %H:%M') < datetime.datetime(2023, 11, 20, 9, 41):
#     #         break
#         elements = dom.select("dt>a")
#         dates = dom.select(".time")
#         for j in range(len(elements)):
#             title  = str(elements[j]).split("<br/>")[0].split('] ')[1]
#             comment = str(elements[j]).split("<br/>")[1][:-4].replace('<b class="searchContextDoc">','').replace('</b>','')
#             date = dates[j].text
#             d['title'].append(title)
#             d['comment'].append(comment)
#             d['date'].append(date)

#     df = pd.DataFrame(d)
#     loader = DataFrameLoader(df,page_content_column='comment')
#     docs = loader.load()
#     vectorstore = FAISS.from_documents(docs,embedding=OpenAIEmbeddings(openai_api_key=ak))