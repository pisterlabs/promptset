import tkinter as tk
from windows import window, menubar
from config import font_style, font_size
import openai
from Box_Message import model_message_box

def login():
    # 读取API_KEY文件
    filename = "../API_KEY/API_KEY_3.5"
    # 打开文件
    with open(filename, "r", encoding="utf-8") as f:
        # 读取第一行内容
        API_KEY = f.readline().strip("\n")
    # 登录
    openai.api_key = API_KEY
    model_message_box.config(state=tk.NORMAL)
    model_message_box.delete("1.0", "end")
    model_message_box.insert("insert", "登录成功")
    model_message_box.config(font=(font_style, font_size + 4))
    model_message_box.config(width=50, height=4)
    model_message_box.config(state=tk.DISABLED)

# -----------------------------------------------------------------------------------#