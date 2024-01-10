import streamlit as st
import openai
import os
import streamlit.components.v1 as components
from openai import OpenAI
from dotenv import load_dotenv
from helpers import create_auto_article

load_dotenv()
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]


#ChatGPTによる文章生成


# Streamlitのタイトルを設定
st.title('問題自動生成')

# ユーザー入力用のテキストボックスを設置
user_input__SEO_keyword = st.text_area("科目")
user_input_target = st.text_area("例題")
user_input_target_ploblem= st.text_area("どのような問題を作ってほしいか")

#カスタムインプットを作りたいなら事前に何個か用意するのが良さそう
if st.button('生成開始'):
            
    result = create_auto_article(user_input_target,user_input_target_ploblem,user_input__SEO_keyword)
    
    st.write("生成記事:")
    st.markdown(result, unsafe_allow_html=True)
    st.session_state["result"]=result



    
    
    



