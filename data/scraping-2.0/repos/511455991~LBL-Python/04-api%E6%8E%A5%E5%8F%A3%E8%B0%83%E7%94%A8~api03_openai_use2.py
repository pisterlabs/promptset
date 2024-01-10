# encoding=utf-8

"""
openai的chatgpt接口调用，使用tkinter制作界面，pyinstaller打包成exe
author：LBL
date:2023-2-24
"""
import os
import openai
from tkinter import *

class chatgpt(object):
    def __init__(self):
        OPENAI_API_KEY = "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"
        openai.api_key = os.getenv("OPENAI_API_KEY",OPENAI_API_KEY)


    def text(self):
        """ 使用chatgpt处理文本信息 """
        # 设置要提的问题
        # prompt = "模仿鲁迅写法编写五百字描写饥荒时，一个小男孩被救下的场景"
        prompt = entry.get()

        # 设置参数，发送请求，获取响应
        response = openai.Completion.create(
            model = "text-davinci-003",     # 设置模型model = "text-davinci-003"是 chatGPT 文本处理模型。
            prompt = prompt,                # 设置提问内容
            temperature = 1,                # 设置生成文本的多样性和创意度在0-1
            max_tokens = 2048,              # 设置生成文本的最大长度，最大为2048
            n = 1,                          # 设置要生成的文本数量
            stop = None                     # 设置文本停止条件，当生成文本中包含这些条件之一停止生成。为一个字符串或列表
        )
        # 提取响应内容中文本答案插入到text框末尾
        result = response.choices[0].text
        text.insert(END, "chatgpt:{}".format(result))
        # 文本框滚动到末尾
        text.see(END)
        # 更新文本框内容
        text.update()


# 程序主入口
if __name__ == "__main__":

    chat = chatgpt()
    # chat.text()
    # 主窗口
    root = Tk()
    # 设置窗口大小
    root.geometry('900x600+0+0')
    # 窗口标题
    root.title('chatgpt聊天机器人')
    # 添加label标签，row行号，column列号
    label = Label(root, text = '请输入问题：', font = ('微软雅黑', 15))
    label.grid(row=0,column=0)

    # 添加entry输入框
    entry = Entry(root,width=60, font=('微软雅黑',15))
    entry.grid(row=0,column=1)

    # 添加text文本框
    text = Text(root,height=40,width=120)
    text.grid(row=1,columnspan=2)

    # 添加button按钮,第三行，第一列，左对齐
    button_s = Button(root, text='发送问题', width='10', command=chat.text)
    button_s.grid(row=2, column=0, sticky=W)
    button_q = Button(root, text='退出程序', width='10', command=root.quit)
    button_q.grid(row=2, column=1, sticky=W)


    # 主窗口不断循环
    root.mainloop()

    # 打包exe命令： pyinstaller -F api03_openai_use2 -w







