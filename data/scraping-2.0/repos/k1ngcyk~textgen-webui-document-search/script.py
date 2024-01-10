import os
import shutil

import gradio as gr
from modules import chat

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader


KNOWLEDGE_PATH = "/tmp/knowledge"
VECTORDB_PATH = "/tmp/vectordb"
EMBEDDING_MODEL = "/mnt/ceph-share/models/SentenceTransformers/text2vec-base-chinese"

params = {
    'topk': 2,
    'enabled': True
}

store = None

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(collection_name="vector_store", persist_directory=VECTORDB_PATH, embedding_function=embeddings)


def load_current_vector():
    global store
    store = load_vector_store()


def create_embedding(folder):
    if os.path.exists(VECTORDB_PATH):
        shutil.rmtree(VECTORDB_PATH)
    if os.path.exists(KNOWLEDGE_PATH):
        shutil.rmtree(KNOWLEDGE_PATH)
    os.makedirs(VECTORDB_PATH)
    os.makedirs(KNOWLEDGE_PATH)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(collection_name="vector_store", persist_directory=VECTORDB_PATH, embedding_function=embeddings)

    for file in folder:
        filename = os.path.split(file.name)[-1]
        shutil.move(file.name, os.path.join(KNOWLEDGE_PATH, filename))
        loader = UnstructuredMarkdownLoader(os.path.join(KNOWLEDGE_PATH, filename))
        docs = loader.load()
        vector_store.add_documents(documents=docs, embedding_function=embeddings)
    vector_store.persist()


def similar(query, topk=2):
    global store
    return store.similarity_search(query, topk)


def custom_generate_chat_prompt(user_input, state, **kwargs):
    if not params['enabled']:
        print('skip ext')
        return chat.generate_chat_prompt(user_input, state, **kwargs)

    global store
    results = similar(user_input, topk=2)
    results = [x.page_content.replace("\n", " ") for x in results]
    if state['mode'] == 'chat-instruct':
        results = similar(user_input, topk=3)
        results = [x.page_content for x in results]
        state['chat-instruct_command'] = '已知内容:\n' + '\n'.join(results) + '\n\n请尽量基于上面的已知内容完成下面的任务. \n' + state['chat-instruct_command']
    else:
        additional_context = '. 请基于以下已知的内容, 专业地回答我的问题, 如果提供的内容不是非常相关, 请抛弃这些内容并以友善的语气进行回答。已知内容: ' + '。'.join(results)
        user_input += additional_context

    return chat.generate_chat_prompt(user_input, state, **kwargs)


def input_modifier(string):
    return string


def ext(check):
    params['enabled'] = check


def ui():
    with gr.Row():
        ext_enabled = gr.Checkbox(value=params['enabled'], label="插件开启")
        ext_enabled.change(ext, inputs=[ext_enabled])
        with gr.Column(min_width=600):
            with gr.Tab("File input"):
                folder_files = gr.File(file_count="directory",
                                       show_label=False)
                load_folder_button = gr.Button("为文件夹创建知识库")
            load_current = gr.Button("加载现有知识库")

    load_folder_button.click(fn=create_embedding,
                             show_progress=True,
                             inputs=[folder_files])
    load_current.click(fn=load_current_vector)
    load_current_vector()