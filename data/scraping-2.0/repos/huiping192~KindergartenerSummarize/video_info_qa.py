import os
import subprocess
import tempfile

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import json


def get_video_info(video_path, original_filename):
    # 构建 ffprobe 命令
    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]

    # 执行命令并捕获输出
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 将输出从字节转换为字符串
    output = result.stdout.decode('utf-8')

    # 将字符串解析为字典
    video_info = json.loads(output)

    # 添加或修改文件名字段
    video_info['file_name'] = original_filename

    return json.dumps(video_info, indent=4)


def display_chat_history(chain):
    # 检查是否有聊天历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'initial_message_displayed' not in st.session_state:
        # 显示初始消息
        initial_question = "50文字以内で、動画の情報を要約してください。"
        answer = chain.run(initial_question)
        # 添加到聊天历史
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # 标记初始消息已显示
        st.session_state.initial_message_displayed = True

    # 显示聊天历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    # 用户输入
    user_query = st.chat_input("質問を入力してください")

    if user_query:
        # 显示用户消息
        st.chat_message("user").markdown(user_query)
        # 添加到聊天历史
        st.session_state.messages.append({"role": "user", "content": user_query})

        # 获取助手回答
        answer = chain.run(user_query)
        # 显示助手消息
        st.chat_message("assistant").markdown(answer)
        # 添加到聊天历史
        st.session_state.messages.append({"role": "assistant", "content": answer})


load_dotenv()

st.title("Video info ChatBot")
# Initialize Streamlit
st.sidebar.title("Video Processing")
uploaded_files = st.sidebar.file_uploader("Upload video", accept_multiple_files=True)

if uploaded_files:
    text = ""
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        # check file_extension is video type
        if file_extension not in [".mp4", ".avi", ".mkv"]:
            # 显示错误信息
            st.error("Please provide a video file.")
            continue

        video_info = get_video_info(temp_file_path, file.name)

        print(video_info)
        text += video_info
        os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]

    model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    retriever = vector_store.as_retriever()
    prompt_template_qa = """あなたは親切で優しいアシスタントです。丁寧に、日本語でお答えください！
    もし以下の情報が探している情報に関連していない場合は、わかりませんと答えてください。

    {context}

    質問: {question}
    回答（日本語）:"""

    prompt_qa = PromptTemplate(
        template=prompt_template_qa,
        input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt_qa}
    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=vector_store.as_retriever(),
                                        chain_type_kwargs=chain_type_kwargs)

    display_chat_history(chain)
    # answer = chain.run("50文字以内で、動画の情報を要約してください。")
    # st.chat_message("assistant").markdown(answer)
    # st.session_state.messages.append({"role": "assistant", "content": answer})
    #
    # user_query = st.chat_input("質問を入力してください")
    #
    # if user_query:
    #     st.chat_message("user").markdown(user_query)
    #     st.session_state.messages.append({"role": "user", "content": user_query})
    #     answer = chain.run(user_query)
    #     st.chat_message("assistant").markdown(answer)
    #     st.session_state.messages.append({"role": "assistant", "content": answer})
