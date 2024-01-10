import streamlit as st
from streamlit_authenticator import Authenticate
from streamlit_chat import message
import streamlit_authenticator as stauth
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import plugin_list as pl



def run():
    init_page()
    
    # 認証
    authenticator = stauth.Authenticate(
        credentials={
            'usernames': {
                USER_NAME : {
                    'email': EMAIL,
                    'name': USER_NAME,
                    'password': PASSWORD
                }
            },
            'cookie': {
                'expiry_days': 90,
                'key': 'some_signature_key',
                'name': 'some_cookie_name'
            }
        }
    )
    
    authenticator.login("ログイン", "main")
    
    # 判定
    if st.session_state["authentication_status"]:
        # メイン画面
        main_view()
        
        authenticator.logout('ログアウト', 'sidebar')
        
    elif st.session_state["authentication_status"] is False:
        st.error('ユーザ名 or パスワードが間違っています')
        
    elif st.session_state["authentication_status"] is None:
        st.warning('ユーザ名とパスワードを入力してください')


def main_view():
    # モデルの選択
    llm = select_model()

    # プラグインの選択
    plugin = select_plugin(llm)

    if plugin == "なし":
        # メッセージの初期化
        init_messages()
    
        # ユーザーの入力を監視
        if user_input := st.chat_input("聞きたいことを入力してね！"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT が考えています ..."):
                response = llm(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")
            

def init_page():
    st.set_page_config(
        page_title="My ChatGPT",
        page_icon="⚙️"
    )
    st.header("My ChatGPT")
    st.sidebar.title("ChatGPT")
                
        
def init_messages():
    clear_button = st.sidebar.button("Clear chat history", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="何かお役に立てることはありますか？")
        ]
        st.session_state.costs = []
        
        
def select_model():
    # サイドバーにモデル選択のラジオボタンを追加
    model = st.sidebar.radio("モデルを選択", ["GPT-3.5", "GPT-4"])
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo-0613"
    else:
        model_name = "gpt-4"
        
    # サイドバーにスライダーを追加、temperatureの値を選択可能にする
    # 初期値は0.0、最小値は0.0、最大値は2.0、ステップは0.1
    temperature = st.sidebar.slider("サンプリング温度", 0.0, 2.0, 0.0, 0.1)
    
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown("**Total cost**")
    # st.sidebar.markdown(cb.total_cost)

    return ChatOpenAI(temperature=temperature, model_name=model_name)        


def select_plugin(llm):
    # サイドバーにプラグイン選択のセレクトボックスを追加
    previous_plugin = st.session_state.get('plugin', None)
    plugin = st.sidebar.selectbox("プラグイン", ["なし", "WEBサイト要約", "Youtube動画要約", "PDF質問"], key='plugin')
    
    # 選択されたプラグインが変更された場合、セッションをクリア
    if previous_plugin is not None and previous_plugin != plugin:
        st.session_state.clear()
        st.session_state['plugin'] = plugin
    
    if plugin == "WEBサイト要約":
        pl.web_summarize(llm)
    elif plugin == "Youtube動画要約":
        pl.youtube_summarize(llm)
    elif plugin == "PDF質問":
        pl.pdf_question(llm)
    
    return plugin
    
    
if __name__ == '__main__':
    run()