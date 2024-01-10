import openai
import streamlit as st
import json

openai.api_base = "https://oai.langcore.org/v1"

def write_spreadsheet(res: dict):
    st.success("以下の情報をスプレッドシートに保存します")
    st.json(res)

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
## functions
def system_prompt(question): 
    return f"""ロール:
あなたは社会人向けのキャリアコーチです。ユーザの深層心理を引き出してください。

行動:
1.まず、[質問]のあとに書かれている質問をユーザにしてください。
2.ユーザの回答が不十分だと感じられる場合は、深掘りをする質問をしてください。
3.[質問]に対する回答を引き出せたと感じたら、end_question関数を呼び出してください。
4.しつこい深堀はしないでください。また[質問]から逸脱しないでください。

[質問] 
{question} 
    """

def functions(question):
    return [
    {
            "name": "end_question", 
            "description": "深掘りを完了した時に呼び出す関数", 
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": f"[{question}]に対するユーザの回答会話履歴を全部見てを100文字程度でまとめてください。", 
                    },
                    "insight": {
                        "type": "string",
                        "description": "ユーザの会話履歴を踏まえて、「あなたの人生の軸はこれですね、こういう仕事が向いているかもしれませんね」というアドバイスを100文字程度で書いてください"
                    }
                },
                "required": [ "answer", "insight"], 
            },
        }
    ]


## end functions


st.title("キャリアコンサルタント PoC")
st.text("対話を通して深層心理を導き、スプレッドシートに保存します")
with st.expander("Click to expand and enter system prompt"):
    question = st.text_input("聞きたい質問", value="あなたが決断するときに，大事にしていることは何ですか？")
    sym_prompt = st.text_area("Enter system prompt", value=system_prompt(question))
    system_prompt_structure = {
        "role": "system",
        "content": sym_prompt
    }


if "messages" not in st.session_state:
    st.session_state.messages = []
if "attempts" not in st.session_state:
    st.session_state.attempts = 0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("あなたの回答を入力してください")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.attempts > 2:
            system_prompt_structure = {
                "role": "system",
                "content": "コーチングが終了したので、お礼を言って会話を終了させてください"
            }
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [
                system_prompt_structure,
                *st.session_state.messages
                ],
                stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "") # type: ignore
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.attempts += 1
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    if st.session_state.attempts > 3: 
        with st.spinner("スプレッドシートへの書き込み中"): 
            res, is_end = function_calling([ system_prompt_structure, *st.session_state.messages], functions(question), "end_question")
            res["question"] = question
            write_spreadsheet(res)