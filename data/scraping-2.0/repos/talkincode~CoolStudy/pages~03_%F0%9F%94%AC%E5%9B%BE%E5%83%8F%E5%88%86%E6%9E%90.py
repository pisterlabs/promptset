import streamlit as st
from libs.llms import openai_analyze_image, openai_streaming
from libs.msal import msal_auth
from libs.session import PageSessionState
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath('..'))
load_dotenv()

st.set_page_config(page_title="图像分析", page_icon="🔬")

page_state = PageSessionState("image_analysis")


# 用于存储对话记录
page_state.initn_attr("messages", [])

# 用于标记上一条用户消息是否已经处理
page_state.initn_attr("last_user_msg_processed", True)

# 用于存储图像分析结果
page_state.initn_attr("analysis_result", "")

page_state.initn_attr("input_type", "camera")

st.sidebar.markdown("# 🔬图像分析")

# 图像分析提示输入
prompt = st.sidebar.text_area("图像分析提示", "识别分析图片内容", height=30)


def on_image_change():
    page_state.analysis_result = None
    page_state.last_user_msg_processed = True
    page_state.messages = []


st.sidebar.button("清除结果", on_click=on_image_change)

# 摄像头输入获取图片
if st.sidebar.selectbox("选择图片输入方式", ["摄像头", "上传图片"]) == "摄像头":
    page_state.input_type = "camera"
    image = st.camera_input("点击按钮截图", on_change=on_image_change, key="image_analysis_camera_image")
else:
    page_state.input_type = "upload"
    image = st.sidebar.file_uploader("上传图片", type=["png", "jpg", "jpeg"],
                                     on_change=on_image_change, key="image_analysis_camera_image")

c1, c2 = st.columns(2)
if page_state.camera_image is not None:
    st.image(page_state.camera_image, caption="", use_column_width=True)
    if page_state.analysis_result is None:
        with st.spinner("分析中..."):
            page_state.analysis_result = openai_analyze_image(prompt, page_state.camera_image)
            page_state.add_chat_msg("messages", {"role": "assistant", "content": page_state.analysis_result})

# 设置对话记录
for msg in page_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 输入用户消息
if uprompt := st.chat_input("输入你的问题"):
    # 用于标记用户消息还没有处理
    page_state.last_user_msg_processed = False
    page_state.add_chat_msg("messages", {"role": "user", "content": uprompt})
    with st.chat_message("user"):
        st.write(uprompt)

# 用户输入响应，如果上一条消息不是助手的消息，且上一条用户消息还没有处理完毕
if ((page_state.messages
     and page_state.messages[-1]["role"] != "assistant"
     and not page_state.last_user_msg_processed)
        and page_state.analysis_result not in [""]):
    # 处理响应
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sysmsg = f""""
            以下是来自一图片识别获取的内容结果：
            '''
            {page_state.analysis_result}
            '''
            我们将围绕这个内容进行深入讨论。
            """
            response = openai_streaming(sysmsg, page_state.messages[-10:])
            # 流式输出
            placeholder = st.empty()
            full_response = ''
            for item in response:
                text = item.content
                if text is not None:
                    full_response += text
                    placeholder.markdown(full_response)
                    page_state.update_last_msg("messages", {"role": "assistant", "content": full_response})
            placeholder.markdown(full_response)

    # 用于标记上一条用户消息已经处理完毕
    page_state.last_user_msg_processed = True
