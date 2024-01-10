# ！/usr/bin env python3
# -*- coding: utf-8 -*-
# author: yangyunlong time:2023/12/1

import gradio as gr
import uvicorn
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
from FlagEmbedding import FlagModel
from langchain.document_loaders import TextLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema.embeddings import Embeddings
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from SparkGPT import SparkGPT
from prompt_utils import generate_prompt

BGE_MODEL_PATH = "D:\\codes\\bge-large-zh"
FILE_PATH="D:\\codes\\zsxq"

class BaaiEmbedding(Embeddings):

    def __init__(self, max_length=512, batch_size=256):
        self.model = FlagModel(BGE_MODEL_PATH,
                               query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode_corpus(texts, self.batch_size,
                                        self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode_queries(text, self.batch_size,
                                         self.max_length).tolist()


def extract_file_dirs(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):

        for file in files:
            fp = os.path.join(root, file)
            file_paths.append(fp)
    return file_paths


app = FastAPI()

files = extract_file_dirs(FILE_PATH)
print(files)
loaders = [TextLoader(f, encoding='utf-8') for f in files]

docs = []
for l in loaders:
    docs.extend(l.load())

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

embeddings = BaaiEmbedding()
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="split_parents",
                     embedding_function=embeddings,
                     persist_directory="./zsxq_index")

# vectorstore.persist()
# The storage layer for the parent documents

store = InMemoryStore()
search_kwargs = {
    "k": 3
}
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs=search_kwargs
)
retriever.add_documents(docs)


def spark_api(prompt):
    spark_gpt = SparkGPT(model="Spark2.0")
    spark_gpt.initialize_message()
    spark_gpt.user_message(prompt)
    response = spark_gpt.get_response()
    return response


def predict(mesasge):
    retrieved_docs = retriever.get_relevant_documents(mesasge)
    docs = []
    for x in retrieved_docs:
        docs.append(x.page_content)

    prompt = generate_prompt(mesasge, docs)
    response = spark_api(prompt)
    return response


def rag_predict(mesasge, label):
    if label:
        prompt_result = predict(mesasge)
    else:
        prompt_result = spark_api(mesasge)
    return prompt_result


# 部署到次级目录，会出现 "GET /queue/join HTTP/1.1" 404 Not Found
def stream_chat(chat_history, message, label):
    bot_message = rag_predict(message, label)

    chat_history[-1][1]= ""
    for char in bot_message:
        chat_history[-1][1] += char
        yield chat_history


def chat(chat_history, message, label):
    bot_message = rag_predict(message, label)
    chat_history.append((message, bot_message))
    return "", chat_history

def add_text(history, text):
    history = history + [(text, None)]
    return history

with gr.Blocks() as demo:
    with gr.Tab("AI Assistant"):
        with gr.Row():
            # 参考官网链接：gradio.app/docs/chatbot
            # 创建一个聊天界面
            with gr.Column(scale=0.6):
                chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=300,
                                     avatar_images=(None, (
                                         os.path.join(os.path.dirname(__file__),
                                                      "./GPT4.png"))))

            with gr.Column(scale=0.4):
                with gr.Row():
                    txt = gr.Textbox(show_label=False,
                                         placeholder="请输入文本",
                                         container=False)

                    use_kg = gr.Checkbox(label="使用", value=False,
                                             info="是否使用检索增强")

                with gr.Row():
                        submit_button = gr.Button("提交")
                        clear_button = gr.ClearButton()

        txt_msg = submit_button.click(add_text, [chatbot, txt], [chatbot],
                                       queue=False)
        txt_chat = txt_msg.then(stream_chat, inputs=[chatbot, txt, use_kg],
                               outputs=[chatbot],
                               api_name="bot_response")
        txt_clear = txt_chat.then(fn=lambda x: "", inputs=[txt],
                           outputs=[txt])
        clear_button.click(fn=lambda: [None,None], inputs=None,
                           outputs=[chatbot,txt])


root = "/"
app = gr.mount_gradio_app(app, demo, path=root)
demo.queue()
if __name__ == "__main__":
    uvicorn.run(app)
