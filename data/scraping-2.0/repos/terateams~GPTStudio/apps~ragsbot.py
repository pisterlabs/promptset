import json
import logging
import streamlit as st
import sys
import os
from dotenv import load_dotenv
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import ChatMessage
from libs.llama_utils import get_llama_memary_index, get_llama_store_index, create_document_index_by_texts, \
    create_document_index_by_files, query_knowledge_data
from libs.session import PageSessionState
from llama_index.tools import QueryEngineTool, ToolMetadata, RetrieverTool
from llama_index.agent import OpenAIAgent
from libs.prompts import get_content_from

#
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

sys.path.append(os.path.abspath('..'))
load_dotenv()


def get_ragsbot_page(botname):
    page_state = PageSessionState(botname)
    # st.sidebar.markdown("# 💡Python 编程导师")

    # 用于存储对话记录, 第一条为欢迎消息
    page_state.initn_attr("messages", [])
    # 用于标记上一条用户消息是否已经处理
    page_state.initn_attr("last_user_msg_processed", True)
    # 用于标记流式输出是否结束
    page_state.initn_attr("streaming_end", True)

    resp = query_knowledge_data("Radius", "radiusrfc.chroma.db", "radiusrfc")
    if resp:
        for r in resp:
            st.write(r.get_text())
            st.write(r.get_embedding())


    def end_chat_streaming():
        """当停止按钮被点击时执行，用于修改处理标志"""
        page_state.streaming_end = True
        page_state.last_user_msg_processed = True

    def start_chat_streaming():
        """当开始按钮被点击时执行，用于修改处理标志"""
        page_state.streaming_end = False
        page_state.last_user_msg_processed = False

    def on_input_prompt(iprompt: str):
        if iprompt.strip() == "":
            return
        page_state.chat_prompt = iprompt
        start_chat_streaming()
        page_state.add_chat_msg("messages", {"role": "user", "content": page_state.chat_prompt})
        with st.chat_message("user"):
            st.write(page_state.chat_prompt)

    # 文本类文件上传
    files = st.sidebar.file_uploader("上传文件", accept_multiple_files=True)

    if st.sidebar.button("索引文件"):
        if files:
            create_document_index_by_files(files, "ragsbot.chroma.db", "default_collection")
            page_state.rags_load = True

    for msg in page_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not page_state.rags_load:
        st.warning("请上传文件")
        st.stop()

    rindex = get_llama_store_index("ragsbot.chroma.db", "default_collection")
    agent = rindex.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        context_prompt="""
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about radius tech"
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        """,
        verbose=True
    )
    # 用户输入
    if not page_state.last_user_msg_processed:
        st.chat_input("请等待上一条消息处理完毕", disabled=True)
    else:
        if prompt := st.chat_input("输入你的问题"):
            on_input_prompt(prompt)

    stop_action = st.sidebar.empty()
    if not page_state.streaming_end:
        stop_action.button('停止输出', on_click=end_chat_streaming, help="点击此按钮停止流式输出")

    # 用户输入响应，如果上一条消息不是助手的消息，且上一条用户消息还没有处理完毕
    if (page_state.messages
            and page_state.messages[-1]["role"] != "assistant"
            and not page_state.last_user_msg_processed):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages = [ChatMessage(role=m["role"], content=m["content"]) for m in page_state.messages[-10:-1]]
                response = agent.stream_chat(
                    page_state.chat_prompt,
                    chat_history=messages,
                )
                # 流式输出
                placeholder = st.empty()
                full_response = ''
                page_state.add_chat_msg("messages", {"role": "assistant", "content": ""})
                for token in response.response_gen:
                    # # 如果用户手动停止了流式输出，就退出循环
                    if page_state.streaming_end:
                        break
                    if token is not None:
                        full_response += token
                        placeholder.markdown(full_response)
                        page_state.update_last_msg("messages", {"role": "assistant", "content": full_response})
                placeholder.markdown(full_response)

        stop_action.empty()
        end_chat_streaming()

    st.sidebar.download_button('导出对话历史',
                               data=json.dumps(page_state.messages, ensure_ascii=False),
                               file_name="chat_history.json", mime="application/json")

    if st.sidebar.button('清除对话历史'):
        page_state.messages = []
        st.rerun()
