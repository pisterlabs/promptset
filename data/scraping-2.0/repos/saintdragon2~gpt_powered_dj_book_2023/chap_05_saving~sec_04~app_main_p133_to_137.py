import json
import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
from download_youtube_audio import download_songs_in_csv

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

# response에 CSV 형식이 있는지 확인하고 있으면 저장하기
def save_to_csv(df):
    file_path=filedialog.asksaveasfilename(defaultextension='.csv')
    if file_path:
        df.to_csv(file_path, sep=';', index=False, lineterminator='\n')
        return f'파일을 저장했습니다. 저장 경로는 다음과 같습니다. \n {file_path}\n이 플레이리스트의 음원을 내려받겠습니까?', file_path # file_path를 리턴하는 부분
    return '저장을 취소했습니다', None

def save_playlist_as_csv(playlist_csv):
    if ";" in playlist_csv:
        lines=playlist_csv.strip().split("\n")
        csv_data=[]
        
        for line in lines:
            if ";" in line:
                csv_data.append(line.split(";"))
            
        if len(csv_data) > 0:
            df=pd.DataFrame(csv_data[1:], columns=csv_data[0])
            return save_to_csv(df)
 
    return f'저장에 실패했습니다. \n저장에 실패한 내용은 다음과 같습니다. \n{playlist_csv}', None

def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response=openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message=response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions={
            "save_playlist_as_csv": save_playlist_as_csv,
            "download_songs_in_csv": download_songs_in_csv
        }
        function_name=response_message["function_call"]["name"]
        fuction_to_call=available_functions[function_name]
        function_args=json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response=fuction_to_call(**function_args)

        if function_name == 'save_playlist_as_csv':
            function_response, csv_file_path=function_response
        
        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message) # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": f'{function_response}',
            }
        ) # 함수 실행 결과도 GPT messages에 추가하기
        response=openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        ) # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content

def main():
    message_log=[
        {
            "role": "system", 
            "content": '''
            You are a DJ assistant who creates playlists. Your user will be Korean, so communicate in Korean, but you must not translate artists' names and song titles into Korean.
                - At first, suggest songs to make a playlist based on users' request. The playlist must contains the title, artist, and release year of each song in a list format. After you show a playlist to the user, You must ask users if they want to save the playlist as follow: "이 플레이리스트를 CSV로 저장하시겠습니까?"
                - After saving the playlist as a CSV file, you must show the CSV file path and ask the users if they would like to download the MP3 files of the songs in the playlist.
            '''
        }
    ]

    functions=[
        {
            "name": "save_playlist_as_csv",
            "description": "Saves the given playlist data into a CSV file when the user confirms the playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_csv": {
                        "type": "string",
                        "description": "A playlist in CSV format separated by ';'. It must contains a header and the release year should follow the 'YYYY' format. The CSV content must starts with a new line. The header of the CSV file must be in English and it should be formatted as follows: 'Title;Artist;Released'.",
                    },
                },
                "required": ["playlist_csv"],
            },
        },
        {
            "name": "download_songs_in_csv",
            "description": "Download mp3 of songs in the recent CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_file": {
                        "type": "string",
                        "description": "The recent csv file path",
                    },
                },
                "required": ["csv_file"],
            },
        },
    ]

    def show_popup_message(window, message):
        popup=tk.Toplevel(window)
        popup.title("")
        
        # 팝업 창의 내용
        label=tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)
        
        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width=label.winfo_reqwidth() + 20
        popup_height=label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")
        
        # 팝업 창의 중앙에 위치하기
        window_x=window.winfo_x()
        window_y=window.winfo_y()
        window_width=window.winfo_width()
        window_height=window.winfo_height()

        popup_x=window_x + window_width // 2 - popup_width // 2
        popup_y=window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")
        
        popup.transient(window)
        popup.attributes('-topmost', True)
        
        popup.update()
        return popup

    def on_send():
        user_input=user_entry.get()
        user_entry.delete(0, tk.END)
        
        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL) # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup=show_popup_message(window, "생각 중...")
        window.update_idletasks() 
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response=send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})
        
        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"AI assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED) 
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window=tk.Tk()
    window.title("GPT Powered DJ")

    font=("맑은 고딕", 10)
    
    conversation=scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame=tk.Frame(window) # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10) # 창의 크기에 맞추어 조절하기(5)
    
    user_entry=tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)
    
    send_button=tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)
    
    window.bind('<Return>', lambda event: on_send())
    window.mainloop()

if __name__ == "__main__":
    main()