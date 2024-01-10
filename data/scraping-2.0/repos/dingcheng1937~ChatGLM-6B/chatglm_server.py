from fastapi import FastAPI, Request
import uvicorn, json, datetime
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import \
    HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import \
    UnstructuredFileLoader
import sentence_transformers
import torch
from chatglm_llm import ChatGLM
import requests
import re
from plugins import settings
from langchain.schema import Document

app = FastAPI()
# Global Parameters



@app.on_event("startup")
def init_cfg():
    global chatglm, embeddings, vector_store,\
        VECTOR_SEARCH_TOP_K
    EMBEDDING_MODEL = "text2vec"
    VECTOR_SEARCH_TOP_K = 3
    LLM_HISTORY_LEN = 3
    LLM_MODEL = "chatglm-6b"
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    embedding_model_dict = {
        "text2vec": "GanymedeNil/text2vec-large-chinese",
    }

    llm_model_dict = {
        "chatglm-6b": "THUDM/chatglm-6b",
    }
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[EMBEDDING_MODEL], )
    embeddings.client = sentence_transformers.SentenceTransformer(
        embeddings.model_name, device=DEVICE)

    chatglm = ChatGLM()
    chatglm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL])
    chatglm.history_len = LLM_HISTORY_LEN

    loader = UnstructuredFileLoader("output2.txt",
                                    mode="elements")
    docs = []
    docs += loader.load()
    vector_store = FAISS.from_documents(docs, embeddings)




session = requests.Session()
# 正则提取摘要和链接
title_pattern = re.compile(
    '<a.target=..blank..target..(.*?)</a>')
brief_pattern = re.compile('K=.SERP(.*?)</p>')
link_pattern = re.compile(
    '(?<=(a.target=._blank..target=._blank..href=.))(.*?)(?=(..h=))')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
proxies = {"http": None, "https": None, }


def find(search_query, headers=headers, proxies=proxies):
    url = 'https://cn.bing.com/search?q={}'.format(
        search_query)
    res = session.get(url, headers=headers, proxies=proxies)
    r = res.text

    title = title_pattern.findall(r)
    brief = brief_pattern.findall(r)
    link = link_pattern.findall(r)

    # 数据清洗
    clear_brief = []
    for i in brief:
        tmp = re.sub('<[^<]+?>', '', i).replace('\n',
                                                '').strip()
        tmp1 = re.sub('^.*&ensp;', '', tmp).replace('\n',
                                                    '').strip()
        tmp2 = re.sub('^.*>', '', tmp1).replace('\n',
                                                '').strip()
        clear_brief.append(tmp2)

    clear_title = []
    for i in title:
        tmp = re.sub('^.*?>', '', i).replace('\n',
                                             '').strip()
        tmp2 = re.sub('<[^<]+?>', '', tmp).replace('\n',
                                                   '').strip()
        clear_title.append(tmp2)
    return [{'title': "[" + clear_title[i] + "]", 'content': clear_brief[i]}
            for i in
            range(min(settings.chunk_count, len(brief)))]


def AskLLM(query, chat_history=[]):
    global chatglm, embeddings, vector_store

    # Prompt template for knowledge chain
    prompt_template = """你基于以下已知信息，以"啵嘤冰"的身份进行聊天，用中文进行回答。如果你不能理解我说的话，也不要说不知道，而是幽默地插科打诨。

                        已知内容:
                        {context}

                        问题:
                        {question}"""
    # Instantiate the prompt template
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context",
                                             "question"])
    chatglm.history = chat_history
    # Fetch the searchtool result
    search_result = find(query)
    print(f"[debug] search result: {search_result}\n")
    if len(search_result) > 0:
        documents = [Document(page_content=result['content'])
                     for result in search_result]
        vector_store_tmp = vector_store
        vector_store_tmp.add_documents(documents)
    else:
        vector_store_tmp = vector_store

    # Update the retriever in knowledge_chain with the new vector store

    # Instantiate the knowledge chain
    knowledge_chain = RetrievalQA.from_llm(
        llm=chatglm,
        retriever=vector_store_tmp.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt,
    )
    # Set the document prompt for the combine documents chain
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Enable returning source documents
    knowledge_chain.return_source_documents = True

    # Call the knowledge chain with a query
    result = knowledge_chain({"query": query})
    #chatglm.history[-1][0] = query
    return result, chatglm.history

@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = []
    log = "[" + "] " + '", prompt:"' + prompt + '", history:"' + repr(history) + '"'
    print(log)
    result, history = AskLLM(prompt, history)
    response = result['result']
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    uvicorn.run('chatglm_server:app', host='0.0.0.0',
                port=8000, workers=1)