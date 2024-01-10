# 使用langchain导入本地文件,存入向量数据库

from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerationConfig

from langchain.document_loaders import TextLoader
from langchain.embeddings import MiniMaxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import torch
import streamlit as st
import json
from io import StringIO
import os


@st.cache_resource #缓存model，多次用相同参数调用function时，直接从cache返回结果，不重复执行function    
def load_model():

    ckpt_path = "/root/llm/baichuan-13B/Baichuan-13B-Chat"
    # from_pretrained()函数中，device_map参数的意思是把模型权重按指定策略load到多张GPU中
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,trust_remote_code=True,device_map="auto",torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,trust_remote_code=True)

    return model,tokenizer


def init_chat_history():

    with st.chat_message("assistant"):
        st.markdown("你好，我是火山引擎文档助手")

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    else:
        for msg in st.session_state.msgs:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    return st.session_state.msgs


def clear_history():
    del st.session_state.msgs


def similarity_search_from_vectordb(query,db_name):
    serarch_results = db_name.similarity_search(query)

    prompt = []
    for result in serarch_results:
        prompt.append(result.page_content)               
        prompt.append("\n"+query)
        prompts = "".join(prompt)

    return prompts


def main():
    uploaded_file = st.file_uploader("上传文件")

    if uploaded_file is not None:
        # streamlit上传成功的文件在内存中，为了便于langchain使用，需要将文件持久化存储到硬盘。
        with open(os.path.join("/root/llm/baichuan-13B/docs/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("文件保存，成功！")
    
        # 把上传的文件导入到langchain的textloader
        loader = TextLoader(os.path.join("/root/llm/baichuan-13B/docs/",uploaded_file.name))
        doc = loader.load()
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=20)
        documents = text_splitter.split_documents(doc)
        # 把分割后到文本embedding成向量，嵌入到向量数据库
        db = FAISS.from_documents(documents,MiniMaxEmbeddings())
        st.success("文档转换为向量，成功！")
        # 加载模型
        model,tokenizer = load_model()
        st.success("模型加载，成功！")

        msgs = init_chat_history()

        if query := st.chat_input("输入你的问题"):
            msgs.append({'role':'user','content':query})
            with st.chat_message("user"):
                st.markdown(query)

            prompts = similarity_search_from_vectordb(query,db)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown(prompts)

            msgs.append({'role':'assistant','content':prompts})        

            with st.chat_message("assistant"):
                placeholder = st.empty()
                for responce in model.chat(tokenizer,msgs,stream=True):
                    placeholder.markdown(responce)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            msgs.append({'role':'assistant','content':responce})

            st.button("清空对话",on_click=clear_history)
            print(json.dumps(st.session_state.msgs, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


