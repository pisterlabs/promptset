import streamlit as st
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# APIキーの設定
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
google_gemini_key = os.environ["GOOGLE_GEMINI_KEY"]
genai.configure(api_key=google_gemini_key)

# Streamlitのセッションステートの初期化
if "chat_log_gemini" not in st.session_state:
    st.session_state.chat_log_gemini = []

if "messages_gemini" not in st.session_state:
    # st.session_state["messages_gemini"] = [ChatMessage(role="assistant", content="なんでも聞いてね")]
    st.session_state["messages_gemini"] = []

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Streamlitのインターフェース設定
st.title('😱Generative AI with Google API')

# 既存のメッセージを表示
for msg in st.session_state.messages_gemini:
    st.chat_message(msg.role).write(msg.content)

# ユーザー入力の処理
if prompt := st.chat_input():
    # ユーザーのメッセージを追加
    st.session_state.messages_gemini.append(ChatMessage(role="user", content=prompt))

    # AIモデルで応答を生成
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        formatted_messages = ["[{}]: {}".format(message.role, message.content) for message in st.session_state.messages_gemini]
        joined_string = "\n".join(formatted_messages)
        response = llm.invoke(joined_string)
        
        # AIの応答をメッセージリストに追加
        st.session_state.messages_gemini.append(ChatMessage(role="assistant", content=response.content))
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        st.error(f"エラーが発生しました: {e}")

# メッセージの表示を更新
for msg in st.session_state.messages_gemini:
    st.chat_message(msg.role).write(msg.content)

