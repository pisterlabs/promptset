# 导入csv
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerationConfig
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import MiniMaxEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
import torch
import json


st.set_page_config(page_title="会PDI术语的Baichuan-13B")
st.title("会PDI术语的Baichuan-13B")

file_path = "/root/datasets/PDI术语库.csv"

@st.cache_resource 
def csv_loader(file_path):

    loader = CSVLoader(file_path=file_path,csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['中CN', '英EN', '缩写Acronym','描述Description']
    })
    data=loader.load()

    embbedings = MiniMaxEmbeddings()
    vedb = FAISS.from_documents(data,embedding=embbedings)
    return vedb

# while True:
#     query = input("prompt:")
#     serarch_results = vedb.similarity_search(query)
#     print(serarch_results[0].page_content)


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
        st.markdown("你好，尝试问我关于PDI术语库的问题吧")

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    else:
        for msg in st.session_state.msgs:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    return st.session_state.msgs


def clear_history():
    del st.session_state.msgs


def similarity_search_from_vectordb(query,vedb):
    serarch_results = vedb.similarity_search(query)
    prompt = serarch_results[0].page_content
    return prompt


def main():
    
    vedb = csv_loader(file_path)
    model,tokenizer=load_model()
    msgs = init_chat_history()

    if query := st.chat_input("请输入问题"):
        msgs.append({'role':'user','content':query})
        with st.chat_message("user"):
            st.markdown(query)

        prompts = similarity_search_from_vectordb(query,vedb)

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

