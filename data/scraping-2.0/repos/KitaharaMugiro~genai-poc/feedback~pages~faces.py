import openai
import streamlit as st
from streamlit_feedback import streamlit_feedback 
import requests
import json
from streamlit_feedback import streamlit_feedback

openai.api_base = "https://oai.langcore.org/v1"

def on_submit(feedback, request_body, response_body, openai_api_key):
    feedback_type = feedback["type"]
    score = feedback["score"]    
    if score == "😞":
        score = 0
    elif score == "🙁":
        score = 1
    elif score == "😐":
        score = 2
    elif score == "🙂":
        score = 3
    elif score == "😀":
        score = 4
    

    optional_text_label = feedback["text"]

    url = "http://langcore.org/api/feedback"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_api_key
    }
    data = {
        "request_body": request_body,
        "response_body": response_body,
        "feedback_type": feedback_type,
        "score": score,
        "optional_text_label": optional_text_label
    }

    requests.post(url, headers=headers, data=json.dumps(data))
    st.toast("フィードバックを送信しました。")
    
    # URLを表示する
    st.write("フィードバックはこちらに記録されます: https://langcore.org/feedback")

def set_userInput(userInput: str):
    st.session_state["userInput"] = userInput
    st.session_state["result"] = None

def main():
    st.title("キャッチコピー生成AI")
    st.write("お題からキャッチコピーを生成します。")
    if "userInput" not in st.session_state:
        st.session_state["userInput"] = None
    if "result" not in st.session_state:
        st.session_state["result"] = None

    # User input
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    input_text = st.text_input("お題を入力してください。")

    if not openai_api_key:
        st.warning("OpenAI API Keyを入力してください。")
        return

    openai.api_key = openai_api_key
    result = None
    request_body = None
    response_body = None
    st.button("キャッチコピー生成", on_click=set_userInput, args=[input_text])
    if st.session_state["userInput"] != None and st.session_state["result"] == None:
        with st.spinner("AIが考え中..."):
            request_body = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": """#お願い
あなたは一流の企画担当です。独創的で、まだ誰も思いついていないような、新しいキャッチコピーを1つ出してください。"""
                    },
                    {
                        "role": "user",
                        "content": st.session_state["userInput"]
                    }
                ],
                "user": "山田太郎",
            }
            response_body = openai.ChatCompletion.create(**request_body)
            result = response_body.choices[0].message.content
            st.session_state["result"] = result
            st.session_state["request_body"] = request_body
            st.session_state["response_body"] = response_body
    
    if st.session_state["result"] != None:
        st.subheader("結果:")
        st.write(st.session_state["result"])


    if st.session_state["result"]:
        feedback = streamlit_feedback(feedback_type="faces", optional_text_label="フィードバックをお願いします")
        if feedback:
            on_submit(feedback, request_body=st.session_state["request_body"] , response_body=st.session_state["response_body"] , openai_api_key=openai_api_key)

if __name__ == "__main__":
    main()

    