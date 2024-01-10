import tkinter as tk
from tkinter import messagebox
import openai

default_api_key = "sk-***"

user_name = "用户"
chatgpt_name = "ChatGPT"

def modify_info():
    def save_changes():
        global user_name, chatgpt_name
        user_name = user_entry.get()
        chatgpt_name = chatgpt_entry.get()
        new_api_key = api_key_entry.get()
        
        if new_api_key:  # 只有在输入框不为空时才更新 API 密钥
            try:
                openai.api_key = new_api_key
                openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "测试 API 密钥"}]
                )
                user_entry.delete(0, tk.END)
                user_entry.insert(0, user_name)
                chatgpt_entry.delete(0, tk.END)
                chatgpt_entry.insert(0, chatgpt_name)
                api_key_entry.delete(0, tk.END)
                api_key_entry.insert(0, new_api_key)
            except openai.error.AuthenticationError as e:
                messagebox.showerror('API 密钥错误', '无效的 API 密钥。请重新输入。')
                openai.api_key = default_api_key
        else:  # 如果输入框为空，只更新用户名称和 ChatGPT 名称
            user_entry.delete(0, tk.END)
            user_entry.insert(0, user_name)
            chatgpt_entry.delete(0, tk.END)
            chatgpt_entry.insert(0, chatgpt_name)

        popup.destroy()

    popup = tk.Toplevel()
    popup.title("修改信息")
    popup.iconbitmap('chatgpt.ico')

    user_label = tk.Label(popup, text="用户名称:")
    user_label.pack()

    user_entry = tk.Entry(popup)
    user_entry.insert(0, user_name)
    user_entry.pack()

    chatgpt_label = tk.Label(popup, text="ChatGPT 名称:")
    chatgpt_label.pack()

    chatgpt_entry = tk.Entry(popup)
    chatgpt_entry.insert(0, chatgpt_name)
    chatgpt_entry.pack()

    api_key_label = tk.Label(popup, text="API 密钥:")
    api_key_label.pack()

    api_key_entry = tk.Entry(popup)
    api_key_entry.pack()

    save_button = tk.Button(popup, text="保存", command=save_changes)
    save_button.pack()

    w = 400
    h = 300
    ws = popup.winfo_screenwidth()
    hs = popup.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    popup.geometry('%dx%d+%d+%d' % (w, h, x, y))
    popup.resizable(width=False, height=False)

def Chat():
    if input_tk.get() != "":
        user_message = input_tk.get()
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"{user_name}: " + user_message + "\n")
        chat_log.config(state=tk.DISABLED)

        loading_label.pack()
        chat_window.update()
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
        )
        reply_content = completion.choices[0].message.content

        input_tk.delete(0, tk.END)
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"{chatgpt_name}: " + reply_content + "\n")
        chat_log.config(state=tk.DISABLED)
        
        loading_label.pack_forget()

    else:
        Error()

def Error():
    messagebox.showerror('错误', '请在提交前输入内容')

chat_window = tk.Tk()
chat_window.title("ChatGPT")
chat_window.iconbitmap('chatgpt.ico')

window_width = 650
window_height = 350

screen_width = chat_window.winfo_screenwidth()
screen_height = chat_window.winfo_screenheight()

x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2

chat_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
chat_window.resizable(width=False, height=False)

modify_button = tk.Button(chat_window, text="修改", command=modify_info)
modify_button.pack(side=tk.LEFT, anchor='sw')

input_tk = tk.Entry(chat_window)
input_tk.pack()

chat_log = tk.Text(chat_window, height=20, width=70)
chat_log.pack()
chat_log.config(state=tk.DISABLED) 

loading_label = tk.Label(chat_window, text="加载中...", font=("Arial", 12))

submit_button = tk.Button(chat_window, text="提交", command=Chat)
submit_button.pack()

chat_window.mainloop()
