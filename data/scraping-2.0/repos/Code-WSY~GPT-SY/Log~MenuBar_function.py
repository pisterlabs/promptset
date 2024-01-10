import openai
import tkinter as tk
from tkinter import filedialog
from config import font_style, font_size
from Menu_mode import selected_mode

from Box_Dialog import Dialog_box
from Box_Input import Input_box
from Box_Message import model_message_box
from Cbox_Model import model_list
from Cbox_Promot import func_list
from Bottom_Submit import messages_list

from UI import chat_UI,load_UI,foget_all


"""
实现菜单栏功能的函数：
    1. 打开文件：open_file()
    2. 保存文件: save_file()
    3. 登录: login()
"""
## 打开文件
def open_file():
    file_path = tk.filedialog.askopenfilename(
        title="选择文件", filetypes=[("All Files", "*")]
    )
    with open(file_path, "r", encoding="utf-8") as f:
        Input_box.delete("1.0", "end")
        Input_box.insert("insert", f.read())


def save_file():
    # 保存messages:
    save_file_path ="../Chat_history/" + model_list.get() + "_" + func_list.get() + ".txt"
    #查看是否有重复文件
    try:
        i=1
        while True:
            #打开文件
            f = open(save_file_path, "r", encoding="utf-8")
            f.close()
            #上面的语句没有报错，说明文件存在
            save_file_path = "../Chat_history/" + model_list.get() + "_" + func_list.get() + "_" + str(i) + ".txt"
            i += 1
    except:
        # 逐个写入
        with open(save_file_path, "w", encoding="utf-8") as f:
            for message in eval(messages_list.get()):
                f.write(str(message) + "\n")

    # 输出保存成功到message_box
    model_message_box.config(state=tk.NORMAL)
    model_message_box.delete("1.0", "end")
    model_message_box.insert("insert", "保存成功")
    model_message_box.config(font=(font_style, font_size + 4))
    model_message_box.config(width=50, height=4)
    # 关闭文件
    f.close()


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

def clear_display():

    Dialog_box.config(state=tk.NORMAL)
    Dialog_box.delete(0.0, tk.END)
    Dialog_box.config(state=tk.DISABLED)
def clear_messages_list():

    messages_list.set("[]")
    model_message_box.config(state=tk.NORMAL)
    model_message_box.delete(0.0, tk.END)
    model_message_box.insert("insert", "成功清空对话记录")
    model_message_box.config(state=tk.DISABLED)

def change_UI():
    if selected_mode.get() == "Prompt-based":
        foget_all()
        chat_UI()
    elif selected_mode.get() == "Fine-tuning":
        foget_all()
        load_UI()

