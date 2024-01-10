import uvicorn
import datetime
import time
import os
import json
import base64
import hashlib

from fastapi import FastAPI, Request
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from create_knowledge import create_base
from file_utils import file_loader, gradioFile_loader
import config
import torch

app = FastAPI()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda"
DEVICE_ID = [0]


def torch_gc():
    for cuda_id in DEVICE_ID:
        CUDA_DEVICE = f"{DEVICE}:{cuda_id}" if DEVICE_ID else DEVICE
        print(CUDA_DEVICE)
        if torch.cuda.is_available():
            with torch.cuda.device(CUDA_DEVICE):
                print("in")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def get_file_hash(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


@app.get("/")
def read_root():
    return {"Hello": "World chat glm2"}


@app.post("/create_knowledge")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    kb_name = json_post_list.get("kb_name")
    file_name = json_post_list.get("file_name")
    encode_file = json_post_list.get("encode_file")
    file_type = json_post_list.get("file_type")

    # 保存文件至encode_file/conversation_id
    file_path = os.path.join(config.config["file_path"], kb_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, f"{file_name}.pdf")
    print("file path")
    print(file_path)
    # 写入base64文件
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(encode_file))

    # 创建知识库
    return create_base(file_path, kb_name)


@app.post("/chat_article")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    print("json_post_list")
    print(json_post_list)
    kb_name = json_post_list.get("kb_name")
    messages = json_post_list.get("messages")
    file_hash = json_post_list.get("file_hash")
    query = messages[-1]["content"]

    kb_name = str(file_hash) + str(kb_name)
    print("kb_name")
    print(kb_name)
    persist_directory = os.path.join(config.config["knowledge_path"], kb_name)

    # 联网搜索
    # try:
    #     from search_engine_parser.core.engines.bing import Search as BingSearch
    #     bsearch = BingSearch()
    #     search_args = (query, 1)
    #     results = await bsearch.async_search(*search_args)
    #     web_content = results["description"][:5]
    #     logger.info("Web_Search - {}".format(web_content))
    # except Exception as e:
    #     logger.error("Web_Search - {}".format(e))
    #     web_content = ""
    web_content = ""

    # 从目录加载向量
    print(
        "Start load vector database... %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    #

    embeddings = HuggingFaceEmbeddings(model_name=config.config["text2vec"])
    print("xxxx")
    print(os.path.exists(persist_directory))
    if os.path.exists(persist_directory):
        print("找到了缓存的索引文件，加载中……")
        vectordb = FAISS.load_local(persist_directory, embeddings)
    docs = vectordb.similarity_search(query, k=5)
    # page = list(set([docs.metadata["page"] for docs in docs]))
    # page.sort()
    context = [docs.page_content for docs in docs]
    #
    #
    prompt = f"已知PDF内容：\n{context}\n根据已知信息回答问题：\n{query}\n网络检索内容：\n{web_content}"
    print("contenxt:", context)
    print("prompt:", prompt)
    query = f"内容为：{context}\n根据已知信息回答问题：{query}"
    history = []
    for i in range(max(-11, -len(messages) + 2), -1, 2):
        history.append((messages[i]["content"], messages[i + 1]["content"]))
    print("history", history)

    response, history = model.chat(tokenizer, query, history=history)
    print("asadada")
    print(history)
    now = datetime.datetime.now()
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(answer)
    torch_gc()
    return answer


@app.post("/chat_knowledge")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    kb_name = json_post_list.get("kb_name")
    file_name = json_post_list.get("file_name")
    encode_file = json_post_list.get("encode_file")
    file_type = json_post_list.get("file_type")

    # 保存文件至encode_file/conversation_id
    file_path = os.path.join(config.config["file_path"], kb_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, f"{str(file_name)+str(file_type)}")
    print("file path")
    print(file_path)
    # 写入base64文件
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(encode_file))

    query = "这篇文章讲了什么"

    file_hash = get_file_hash(file_path)
    kb_name = str(file_hash) + str(kb_name)
    persist_directory = os.path.join(config.config["knowledge_path"], kb_name)

    # 联网搜索
    # try:
    #     from search_engine_parser.core.engines.bing import Search as BingSearch
    #     bsearch = BingSearch()
    #     search_args = (query, 1)
    #     results = await bsearch.async_search(*search_args)
    #     web_content = results["description"][:5]
    #     logger.info("Web_Search - {}".format(web_content))
    # except Exception as e:
    #     logger.error("Web_Search - {}".format(e))
    #     web_content = ""
    web_content = ""

    # 从目录加载向量
    print(
        "Start load vector database... %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    #

    embeddings = HuggingFaceEmbeddings(model_name=config.config["text2vec"])
    print("xxxx")
    print(os.path.exists(persist_directory))
    if os.path.exists(persist_directory):
        print("找到了缓存的索引文件，加载中……")
        vectordb = FAISS.load_local(persist_directory, embeddings)
    else:
        # loader = PyPDFLoader(file_path)
        print("loader...")
        # docs_load = loader.load()
        #         docs_load = file_loader(file_path, file_type)

        #         print(f"docs: {docs_load}")
        #         documents = []
        #         text_splitter = RecursiveCharacterTextSplitter(
        #             chunk_size=500, chunk_overlap=200
        #         )
        #         docs_load = text_splitter.split_documents(docs_load)
        #         documents.extend(docs_load)
        documents = file_loader(file_path, file_type)
        vectordb = FAISS.from_documents(documents, embeddings)
        vectordb.save_local(persist_directory)

    docs = vectordb.similarity_search(query, k=5)
    # page = list(set([docs.metadata["page"] for docs in docs]))
    # page.sort()
    context = [docs.page_content for docs in docs]
    #
    #
    prompt = f"已知PDF内容：\n{context}\n根据已知信息回答问题：\n{query}\n网络检索内容：\n{web_content}"
    print("contenxt:", context)
    print("prompt:", prompt)
    query = f"内容为：{context}\n根据已知信息回答问题：{query}"

    response, history = model.chat(tokenizer, query, history=[])

    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_hash": file_hash,
    }
    print(answer)
    torch_gc()
    return answer

# gradio



@app.post("/gradio_chat")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    print("json_post_list")
    print(json_post_list)
    kb_name = json_post_list.get("kb_name")
    messages = json_post_list.get("messages")
    query = messages[-1]["content"]

    print(kb_name)
    persist_directory = os.path.join(config.config["knowledge_path"], kb_name)

    # 联网搜索
    # try:
    #     from search_engine_parser.core.engines.bing import Search as BingSearch
    #     bsearch = BingSearch()
    #     search_args = (query, 1)
    #     results = await bsearch.async_search(*search_args)
    #     web_content = results["description"][:5]
    #     logger.info("Web_Search - {}".format(web_content))
    # except Exception as e:
    #     logger.error("Web_Search - {}".format(e))
    #     web_content = ""
    web_content = ""

    # 从目录加载向量
    print(
        "Start load vector database... %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    #

    embeddings = HuggingFaceEmbeddings(model_name=config.config["text2vec"])
    print("xxxx")
    print(os.path.exists(persist_directory))
    if os.path.exists(persist_directory):
        print("找到了缓存的索引文件，加载中……")
        vectordb = FAISS.load_local(persist_directory, embeddings)
    docs = vectordb.similarity_search(query, k=5)
    # page = list(set([docs.metadata["page"] for docs in docs]))
    # page.sort()
    context = [docs.page_content for docs in docs]
    #
    #
    prompt = f"已知PDF内容：\n{context}\n根据已知信息回答问题：\n{query}\n网络检索内容：\n{web_content}"
    print("contenxt:", context)
    print("prompt:", prompt)
    query = f"内容为：{context}\n根据已知信息回答问题：{query}"
    history = []
    for i in range(max(-11, -len(messages) + 2), -1, 2):
        history.append((messages[i]["content"], messages[i + 1]["content"]))
    print("history", history)

    response, history = model.chat(tokenizer, query, history=history)
    print("asadada")
    print(history)
    now = datetime.datetime.now()
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(answer)
    torch_gc()
    return answer


@app.post("/gradio_knowledge")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    kb_name = json_post_list.get("kb_name")
    file_dict = json_post_list.get("file_dict")

    # 保存文件至encode_file/conversation_id
    file_path = os.path.join(config.config["file_path"], kb_name)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 文件放到服务器
    for file_key in file_dict:
        file_save = os.path.join(file_path, f"{str(file_key)}")
        print("file path")
        print(file_save)
        # 写入base64文件
        with open(file_save, "wb") as f:
            f.write(base64.b64decode(file_dict[file_key]['fileContent']))

    persist_directory = os.path.join(config.config["knowledge_path"], kb_name)

    # 联网搜索
    # try:
    #     from search_engine_parser.core.engines.bing import Search as BingSearch
    #     bsearch = BingSearch()
    #     search_args = (query, 1)
    #     results = await bsearch.async_search(*search_args)
    #     web_content = results["description"][:5]
    #     logger.info("Web_Search - {}".format(web_content))
    # except Exception as e:
    #     logger.error("Web_Search - {}".format(e))
    #     web_content = ""
    web_content = ""

    # 从目录加载向量
    print(
        "Start load vector database... %s",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    #
    try:
        embeddings = HuggingFaceEmbeddings(model_name=config.config["text2vec"])
        print("xxxx")
        print(os.path.exists(persist_directory))
        if os.path.exists(persist_directory):
            print("找到了缓存的索引文件，加载中……")
            vectordb = FAISS.load_local(persist_directory, embeddings)
        else:
            print("loader...")
            documents = gradioFile_loader(file_dict, file_path)
            vectordb = FAISS.from_documents(documents, embeddings)
            vectordb.save_local(persist_directory)
        vector_info = "知识库创建成功"
        status_code = 200
    except:
        vector_info = "知识库创建失败"
        status_code = 500

    answer = {
        "response": vector_info,
        "history": "",
        "status": status_code,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_hash": "",
    }
    torch_gc()
    return answer




if __name__ == "__main__":
    print("load path..")
    model_path = "/root/autodl-tmp/chatglm2-6b-int4"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(model_path, trust_remote_code=True)
        .quantize(4)
        .half()
        .cuda(0)
    )
    print("load ok...")

    model.eval()

    uvicorn.run(app, host="127.0.0.1", port=8002, workers=1)
