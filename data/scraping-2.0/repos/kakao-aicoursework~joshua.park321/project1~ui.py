import tkinter as tk
from tkinter import scrolledtext

from project1.assets import data_카카오톡채널
from project1.openai_helper import OpenAiHelper

FONT = ("맑은 고딕", 12)
openai_helper = OpenAiHelper()
data = data_카카오톡채널()
openai_helper.set_system_prompt(f'''
    너는 카카오톡 채널 문서를 읽고 유저의 질문에 답변하는 챗봇이다
    성실하게 데이터를 검토하고 적절한 답변을 해야 한다
    답변은 자세한 답변을 요구하기 전까지는 두세 문장의 짧은 답변을 하라
    가능한 관련 link 를 첨부해서 유저가 문서에 직접 접근할 수 있도록 하라
    
    참조할 데이터는 아래와 같다
    ===================
    {data}
    ''')


def setup_ui():
    window = tk.Tk()
    window.title("GPT AI")

    ui_conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, fg='black', bg='#f0f0f0', font=FONT)
    # width, height를 없애고 배경색 지정하기(2)
    ui_conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    ui_conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    ui_conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    ui_input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    ui_input_frame.pack(fill=tk.X, padx=10, pady=10, expand=True)  # 창의 크기에 맞추어 조절하기(5)

    ui_user_entry = tk.Entry(ui_input_frame)
    ui_user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    on_send = lambda: _on_click_send(ui_conversation, ui_user_entry)

    ui_send_button = tk.Button(ui_input_frame, text="Send", command=on_send)
    ui_send_button.pack(side=tk.RIGHT)
    window.bind('<Return>', lambda event: on_send())

    return window


def _on_click_send(ui_conversation, ui_user_entry):
    user_input = ui_user_entry.get()
    if user_input is None or user_input == "":
        return

    _clear_user_entry(ui_user_entry)

    if user_input.lower() == "quit":
        window.destroy()
        return

    _write_message_on_ui(user_input, 'user', ui_conversation)
    response = _send_message_with_popup(user_input)
    _write_message_on_ui(response, 'assistant', ui_conversation)

def _clear_user_entry(ui_user_entry):
    ui_user_entry.delete(0, tk.END)



def _write_message_on_ui(message, role, ui_conversation):
    ui_conversation.config(state=tk.NORMAL)  # 이동
    ui_conversation.insert(tk.END, f"{role}: {message}\n", role)  # 이동
    ui_conversation.config(state=tk.DISABLED)
    ui_conversation.see(tk.END)


def _send_message_with_popup(user_input):
    thinking_popup = _show_popup_message(window, "처리중...")
    window.update_idletasks()
    # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
    response = openai_helper.send_user_message_and_get_content(user_input)
    thinking_popup.destroy()

    return response


def _show_popup_message(window, message):
    popup = tk.Toplevel(window)
    popup.title("")

    # 팝업 창의 내용
    label = tk.Label(popup, text=message, font=FONT)
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


if __name__ == "__main__":
    window = setup_ui()
    window.mainloop()
