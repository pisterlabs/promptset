#-*- coding: utf-8 -*-
import json
import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
from pathlib import Path

import chromadb
import openai
# import markdown_to_json
import json
import uuid

import os

global vectordb_ids, vectordb_docs, collection
openai.api_key = os.environ.get("LLM_LECTURE_KEY")

# response에 CSV 형식이 있는지 확인하고 있으면 저장하기
def load_kakao_channel_info ():
    file_path = Path("datas/project_data_kakao_channel.txt")
    if file_path.is_file() :
        return file_path.open("r", encoding="utf-8").read()
    else :
        raise f"file {file_path.absolute()} erorr"

def parse_markdown_manually (data_in_line:list) :
    
    data = {}
    tmp_data_name = ""
    tmp_data_context = ""
    
    for i, v in enumerate(data_in_line) :
        if len(v) == 0 : continue
        if i == 0 :
            tmp_data_name = "doc_title"
            tmp_data_context = v
            continue
        if v[0] == "#" :
            data[tmp_data_name] = tmp_data_context.strip()
            tmp_data_context = ""
            tmp_data_name = v[1:].strip()
            continue
        tmp_data_context += "\n"
        tmp_data_context += v
        continue
    
    return data

def data_vectorization_using_gpt (collection) :
    
    temperature = 0.0
    max_tokens = 4096
    
    data = load_kakao_channel_info()
    data = "\n".join([v.strip() for v in data.split("\n")])
    
    # jsonified = markdown_to_json.jsonify(data)
    
    parsed_data = parse_markdown_manually(data.split("\n"))
    # print("data", parsed_data)
    
    ids = []
    doc_meta = []
    documents = {}
    embeddings = []

    doc_title = ""
    for i, key in enumerate(parsed_data.keys()) :
        if key == "doc_title" :
            doc_title = parsed_data[key]
            continue
        id = str(uuid.uuid4())[:8]

        document_to_embed = f"{key}: {parsed_data[key]}"

        meta = {
        }
        embedding = get_vector_from_openai(document_to_embed)

        ids.append(id)
        doc_meta.append(meta)
        documents[id] = {
            "Doc" : doc_title,
            "Title" : key,
            "Contents" : parsed_data[key],
        }
        embeddings.append(embedding)
    
    # DB 저장
    collection.add(
        # documents=documents,
        embeddings=embeddings,
        # metadatas=doc_meta,
        ids=ids
    )
    print("vector db update completed")
    
    return ids, documents

def get_vector_from_openai (text) :
    response = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return response.data[0].embedding

def query_kakao_channel_info (query_text) :
    global vectordb_ids, vectordb_docs, collection
    
    result = collection.query(
        query_embeddings=get_vector_from_openai(query_text),
        n_results=2,
    )
    results = []
    for id in result["ids"][0] :
        results.append(vectordb_docs[id])
    return results


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )
    
    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "query_kakao_channel_info": query_kakao_channel_info,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response),
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def main():
    message_log = [
            {
                "role": "system",
                "content": '''
                You are a very kind kakao developers helper BOT.
                You can instruct all of the api in kakao API.
                Your user will be Korean, so communicate in Korean.
                    
                '''
            }
        ]

    functions = [
        {
            "name": "query_kakao_channel_info",
            "description": "Get a document which describes introduction of kakaotalk channel API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "a keyword for search information from kakao api database",
                    },
                },
                "required": ["query_text"],
            },
        }
    ]
    
    # create vectordb client
    client = chromadb.EphemeralClient()
    # client = chromadb.PersistentClient()

    global collection
    collection = client.get_or_create_collection(
        name="kakao_api",
        metadata={"hnsw:space": "l2"}
    )

    global vectordb_ids, vectordb_docs
    vectordb_ids, vectordb_docs = data_vectorization_using_gpt(collection)


    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"AI assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    def init_message () :
        conversation.insert(tk.END, f"카카오 Assistant: 안녕하세요, 저는 카카오 서비스 챗봇입니다.\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("카카오 서비스 챗봇")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)
    init_message()

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    main()

