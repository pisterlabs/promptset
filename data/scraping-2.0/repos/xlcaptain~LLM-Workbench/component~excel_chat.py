import os

import pandas as pd
import streamlit as st
from PIL import Image
from loguru import logger
import openai

from .pipelines.excel_utils import CodeKernel, extract_code, execute


EXAMPLES = [
    "",
    "未成年乘客占比多少？",
    "男性乘客的平均年龄是多少？",
    "有多少人有 3 个以上的兄弟姐妹？",
    "各个登船港口的人数有多少？",
    "各等级客舱的平均船票价格是多少？",
    "查询年龄最大的乘客的登船信息",
]


@st.cache_resource
def save_file(df: pd.DataFrame, filename):
    df.to_excel(filename, index=False)
    return filename


@st.cache_resource
def get_kernel():
    return CodeKernel()


def get_system_messages(filename):
    PRESET_CODE = f"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

# # 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 18

# read data
df = pd.read_excel("{filename}")
"""
    code = PRESET_CODE + "df.info()"
    _, res = execute(code, get_kernel())

    SYSTEM_MESSAGE = [
        {
            "role": "system",
            "content": """你是一位智能AI助手，你叫"审元大模型"，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。
    你主要基于 pandas 库来执行代码并输出结果，以此来回答用户的相关问题。现在已经通过 pandas 正确加载了数据文件并创建了初始数据框 pd.DataFrame，其名称为 df。"""
        },
        {
            "role": "user",
            "content": "查看一下数据框 df 的具体信息"
        },
        {
            "role": "assistant",
            "content": f""" interpreter\n要查看数据框 df 的具体信息，您可以使用 `info()` 函数。

    ```python
    {code}
    ```
    """
        },
        {
            "role": "function",
            "content": res,
        },
        {
            "role": "assistant",
            "content": f"根据查询结果，该数据框的具体信息为：\n{res}"
        },
    ]

    code = PRESET_CODE + "df.head(2)"
    _, res = execute(code, get_kernel())
    SYSTEM_MESSAGE.extend(
        [
            {
                "role": "user",
                "content": "查看数据的前两行"
            },
            {
                "role": "assistant",
                "content": f""" interpreter\n要查看数据框 df 的前两行，您可以使用 `head()` 函数。

            ```python
            {code}
            ```
            """
            },
            {
                "role": "function",
                "content": res,
            },
            {
                "role": "assistant",
                "content": f"根据查询结果，该数据框的前两行为：\n{res}"
            },
        ]
    )
    return SYSTEM_MESSAGE, PRESET_CODE


def chat_once(preset_code: str, system_messages: list, query: str):
    params = dict(
        model="chatglm3",
        messages=system_messages + [{"role": "user", "content": query}],
        temperature=0,
    )
    openai.api_key = 'xxxx'
    openai.api_base = "http://192.168.20.59:7891/v1"
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content
    if "interpreter" in content:
        logger.info(f"Interpreter Response: {content}")

        try:
            code = extract_code(content)
            logger.info(f"Interpreter Code: {code}")
            code = preset_code + code
            res_type, res = execute(code, get_kernel())
            logger.info(f"Observation Response: {res}")

            params["messages"].append(
                {
                    "role": "assistant",
                    "content": content
                }
            )

            if res_type == "image":
                return res
            else:
                params["messages"].append(
                    {
                        "role": "function",
                        "content": res,
                    }
                )
                return openai.ChatCompletion.create(**params, stream=True)

        except:
            return "抱歉，我暂时无法回答该问题，请您尝试别的问题！"
    else:
        return content


@st.cache_resource
def load_excel(file_path):
    return pd.read_excel(file_path)


import time


def excel_chat():
    st.title("💬 表格问答")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        uploaded_file = st.file_uploader("请在此处上传excel文件",
                                         accept_multiple_files=False,
                                         type=['xlsx', 'xls'],
                                         )

        if uploaded_file is not None:
            with open(os.path.join('static/excel', uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            load_excel(os.path.join('static/excel', uploaded_file.name))
            st.success("Uploaded file: '{}'".format(uploaded_file.name))

        # 选择文件
        filename = st.selectbox('请选择一个文件1：', sorted(os.listdir('static/excel'), reverse=True), )
        st.write('You selected: ', filename)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('🗑️删除选中文件', use_container_width=True, ):
                try:
                    os.remove(f'static/excel/{filename}')
                    st.success(f'文件 {filename} 已被删除。')
                    st.rerun()
                except Exception as e:
                    st.error(f'删除文件时出错: {e}')

        with col2:
            if st.button("🗑️ 清空历史对话", use_container_width=True, ):
                st.session_state.messages = []

    file_path = os.path.join('static/excel', filename)
    df = load_excel(file_path)
    time.sleep(0.1)
    # 加载文件
    if len(df) != 0:
        with st.expander("DataFrame", False):
            st.dataframe(df)

        system_messages, preset_code = get_system_messages(file_path)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                else:
                    st.image(message["content"])

        if system_messages:
            if prompt := st.chat_input("示例：该数据集一共有多少个样本？", key="prompt"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt
                    }
                )

                with st.chat_message("assistant"):
                    with st.spinner('Wait...'):
                        response = chat_once(preset_code, system_messages, prompt)

                    if isinstance(response, str):
                        full_response = response
                        st.markdown(full_response)
                    elif isinstance(response, Image.Image):
                        full_response = response
                        st.image(full_response)
                    else:
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in response:
                            full_response += chunk.choices[0].delta.get('content', "")

                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response
                        }
                    )