# langchain prompt templates
# langchain documentloader

from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerationConfig
from langchain import PromptTemplate
import torch
import streamlit as st
import json

st.set_page_config(page_title="langchain翻译官")
st.title("langchain+Baichuan-13B翻译官")


def prompt_format(chat_input:str):
    template = "翻译成英文：{input}"

    prompt_template = PromptTemplate.from_template(template)
    prompt = prompt_template.format(input=chat_input)
    return prompt


# messages:List[dict]
# messages = [{"role":"user","content":prompt}]
# for resp in model.chat(tokenizer,messages=messages,stream=True):
#     print(resp)


def chat_history():

    # 初始化聊天记录
    welcome = "您好，我是您的翻译助手"

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

        with st.chat_message('assistant'):
            st.markdown(welcome)
        st.session_state.messages.append({'role':"assistant",'content':welcome})

    # 展示聊天消息
    else:
        for message in st.session_state["messages"]:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    return st.session_state["messages"]


def clear_history():
    del st.session_state.messages


@st.cache_resource #缓存model，多次用相同参数调用function时，直接从cache返回结果，不重复执行function    
def load_model():

    ckpt_path = "/root/llm/baichuan-13B/Baichuan-13B-Chat"
    # from_pretrained()函数中，device_map参数的意思是把模型权重按指定策略load到多张GPU中
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,trust_remote_code=True,device_map="auto",torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,trust_remote_code=True)

    return model,tokenizer


def main():
    
    msg = chat_history()
    model,tokenizer = load_model()
    
    # 接收用户输入
    if chat_input := st.chat_input("在这里输入您想翻译的文字"):
        prompt = prompt_format(chat_input)
        msg.append({'role':'user','content':prompt})
        # 展示用户输入
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 展示LLM的回答
        with st.chat_message("assistant"):

            placeholder = st.empty()
            for responce in model.chat(tokenizer,msg,stream=True):
                placeholder.markdown(responce)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        msg.append({'role':'assistant','content':responce})
        print(json.dumps(msg, ensure_ascii=False), flush=True)

        st.button("清空历史",on_click=clear_history)


if __name__ == "__main__":
    main()