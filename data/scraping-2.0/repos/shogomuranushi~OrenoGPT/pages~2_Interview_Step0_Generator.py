import streamlit as st
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# 定義
title = "カジュアル面談スカウト文章ジェネレーター"
system_prompt = """あなたは優秀な面接担当者です。採用候補者に「あなたのここが良いと思ったのでぜひお会いしたいです。」という文章を送るので、次のキーワードを添えて文章を校正してください。ステップバイステップで考えましょう。"""
user_prompt_template = """キーワードを参考にビジネス文章として校正してください。

【XXX社が提供できる価値】
xxx
xxx

【条件】
ChatGPTが生成した文章らしい感じは減らしてください。
持ち上げ過ぎず、無礼すぎない、候補者に興味を持っている、共感している、ほんの少しカジュアルな文章にしてください。
キーワードとXXX社が提供できる価値の類似度が高い場合は、XXX社が提供できる価値を参考に「XXX社だとこういう環境が提供できるよ」「こういうカルチャーだからフィットするよ」「こういう業務任せたいと思っているから、経験が合うよ」という文章を生成してください。

【キーワード】
{keyword}
【生成文】
"""

# チャットボットとやりとりする関数
def api_call(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    return response

def create_exercise(keyword):
    user_prompt = user_prompt_template.format(keyword=keyword)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = api_call(messages)
    return response

# UIの構築
st.title(title)

tab1, tab2= st.tabs(["生成", "プロンプト"])

with tab1:
    keyword = st.text_area("キーワードを入れると、その内容を踏まえて生成されます。一つの問いずつ入れると精度が上がりやすいです。")
    if st.button("生成"):
        response = create_exercise(keyword)
        with st.empty():
            exercise = ""
            for chunk in response:
                tmp_exercise = chunk["choices"][0]["delta"].get("content", "")
                exercise += tmp_exercise
                st.write("💻: " + exercise)

with tab2:
    st.text_area("システムプロンプト", system_prompt)
    st.text_area("ユーザプロンプト", user_prompt_template)
