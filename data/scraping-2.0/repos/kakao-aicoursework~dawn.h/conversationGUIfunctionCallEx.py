import json
import openai
import chromadb
import tkinter as tk
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
import os
openai.api_key = os.environ["GPT_KEY"]



def generate_db(name, space="cosine"):

    client = chromadb.PersistentClient()

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": space}  # l2 is the default
    )

    return collection


def insert_text_to_vectordb(text_dict, collection):
    # DB 저장
    collection.add(
        documents=text_dict["text"],
        ids=text_dict["id"],
    )

def query_to_vectordb(query, collection, k=5):
    # DB에서 불러오기
    results = collection.query(
        query_texts=query,
        n_results=k,
    )

    result_list = results['documents'][0]
    result_string = " ".join(result_list)
    return result_string

def load_text_file(text_file_path: str):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(text)

    text_dict= {"id": [], "text": texts}
    text_dict["id"] = [f"{i}" for i in range(len(texts))]

    return text_dict


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1, collection=None):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]
    print(response_message)
    if response_message.get("function_call"):
        available_functions = {
            "query_to_vectordb": query_to_vectordb,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_args["collection"] = collection
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
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
            당신은 카카오톡 서비스의 백과사전입니다. 사용자의 질문에 대해 데이터베이스에서 검색을 해서 이와 관련된 문장을 전문적으로 답변합니다.
            '''
        }
    ]

    functions = [
        {
            "name": "query_to_vectordb",
            "description": "카카오톡과 관련된 문장을 DB에서 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "카카오톡과 관련해서 검색하고자 하는 키워드 또는 문장",
                    },
                },
                "required": ["query"],
            },
        }
    ]

    collection = generate_db("kakaotalk")
    insert_text_to_vectordb(load_text_file("data/project_data_카카오톡채널.txt"), collection)

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
        response = send_message(message_log, functions, collection=collection)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")
    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#292828")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#424242")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

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