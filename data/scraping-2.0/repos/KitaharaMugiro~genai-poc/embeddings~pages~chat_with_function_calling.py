import openai
import streamlit as st
import json

st.title("LangCore Chatbot")
# APIキーの入力
api_key = st.text_input("Enter your Langcore API Key:", type="password")
# グループ名の入力
group_name = st.text_input("Enter a group name:")

openai.api_base = "https://oai.langcore.org/v1"
openai.api_key = api_key


def function_calling(messages, functions, function_name):
    function_call = "auto"
    if function_name: 
        function_call = {"name": function_name}
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call=function_call
    )

    assert "choices" in response, response
    res = response["choices"][0]["message"] # type: ignore
    if "function_call" in res:
        return json.loads(res["function_call"]["arguments"]), True
    return res, False

with st.expander("Click to expand and enter system prompt"):
    system_prompt = st.text_area("Enter system prompt", value="""ユーザの質問に対して、以下の情報を使って答えてください。

{{EMBEDDINGS_CONTEXT}}

上記の情報のみを利用し、確信があることだけ書いて(もし上記に情報がなければ回答しないで)
分からない時は必要な情報をわたしに質問して
情報に自身が無いことは回答しないで
""")
    
    match_threshold = st.text_input("Embeddings-Match-Threshhold", value="0.5")
    match_count = st.text_input("Embeddings-Match-Count", value="3")


if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0 or st.sidebar.button("Reset chat history"):
    st.session_state.messages.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("LangCoreについて教えて")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 一旦Function Callingでクエリを考える
        args, is_function_called = function_calling(
            messages=[{"role": "system", "content": system_prompt}, *st.session_state.messages],
            functions=[
    {
            "name": "query", 
            "description": "文章からユーザが求めている情報の検索ワードを作成する", 
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "ユーザの会話からユーザの求めているものを検索するためのクエリを作成してください。", 
                    }
                },
                "required": [  "query"], 
            },
        }
    ],
            function_name="query")

        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            query=args["query"] ,
            groupName = group_name,
            headers = {
                "Content-Type": "application/json",
                "LangCore-Embeddings": "on",
                "LangCore-Embeddings-Match-Threshold": match_threshold,
                "LangCore-Embeddings-Match-Count": match_count,
            },
            messages= [
                {
                    "role": "system",
                    "content": system_prompt
                },
                *st.session_state.messages
                ],
                stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "") # type: ignore
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})