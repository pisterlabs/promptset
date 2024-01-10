import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")

# ChatGPTによる文章生成
def create_auto_article(inputs):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "あなたはブログ記事を生成する人です すべてHTML形式で出力してください。デザインを施してください "},
            {"role": "user", "content": f"あなたは以下のデータを基に記事を作成します。また、すべてHTML形式で出力してください\
             ターゲットは{inputs['target']}\
            ターゲットの問題は{inputs['target_problem']}\
            悩みから生まれる感情は{inputs['emotion']}\
            悩みのの原因は{inputs['problem_reason']}\
            権威は{inputs['authority']}\
            商品名は{inputs['goods']}\
            アピール,訴求は{inputs['appeal']}\
            レビューは{inputs['review']}\
            価格は{inputs['price']}\
            特別価格は{inputs['special_price']}\
            参考にする記事の見出しは{inputs['example_title']}\
            参考にする記事の本文は{inputs['example_text']}"}
        ]
    )
    return response.choices[0].message.content

# Streamlitのタイトルを設定
st.title('記事自動生成')

# ユーザー入力用のテキストボックスを設置
input_names = [
    "target", "target_problem", "emotion", "problem_reason",
    "authority", "goods", "appeal", "review", "price",
    "special_price", "example_title", "example_text"
]

user_inputs = {}
for input_name in input_names:
    user_inputs[input_name] = st.text_area(input_name.capitalize())

if st.button('生成開始'):
    result = create_auto_article(user_inputs)
    st.write("生成記事:")
    st.markdown(result, unsafe_allow_html=True)
