import tkinter as tk
from threading import Thread
import pygame
from Api.openai_api import OpenAIChatbot
from Api.baidu_api_text import BaiduTranslator
from Api.vits_api import voice_vits

class IntegratedChatbot:
    def __init__(self, openai_api_key, baidu_appid, baidu_secret_key):
        self.chatbot = OpenAIChatbot(openai_api_key)
        self.translator = BaiduTranslator(baidu_appid, baidu_secret_key)

    def play_audio(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def get_chat_response(self, user_input):
        
        openai_response = self.chatbot.get_chat_response(user_input)
 
        translated_response = self.translator.translate(openai_response, 'zh', 'jp')

        audio_file_path = voice_vits(translated_response)
        if audio_file_path:
            self.play_audio(audio_file_path)

        return openai_response, translated_response


class ChatApplication:
    def __init__(self, master, chatbot):
        self.master = master
        self.master.title("ATRI")
        self.chatbot = chatbot

        # 主窗口网格布局配置
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # 创建聊天框架
        chat_frame = tk.Frame(master)
        chat_frame.grid(row=0, column=0, sticky="nsew")

        # 聊天框架网格配置
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=5)
        chat_frame.rowconfigure(1, weight=1)

        # 创建用于显示聊天记录的文本框，初始化为空
        self.text_widget = tk.Text(chat_frame, state='disabled', font=("Microsoft YaHei", 10))
        self.text_widget.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)

        # 创建滚动条
        scrollbar = tk.Scrollbar(chat_frame, width=10, command=self.text_widget.yview)  # 将width设置为较小的值
        scrollbar.grid(row=0, column=1, sticky='nsew')
        self.text_widget['yscrollcommand'] = scrollbar.set

        # 创建消息输入框和发送按钮
        self.msg_entry = tk.Entry(chat_frame, width=50)
        self.msg_entry.grid(row=1, column=0, padx=15, pady=15, sticky="ew")

        self.send_button = tk.Button(chat_frame, text="发送", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=15, pady=5, sticky="ew")

        # 绑定Enter键到发送消息函数
        self.msg_entry.bind("<Return>", self.send_message_on_enter)

    def send_message(self):
        user_input = self.msg_entry.get()
        if user_input:
            self._insert_message(user_input, "You")
            self.master.update_idletasks()
            Thread(target=self.handle_response, args=(user_input,)).start()

    def handle_response(self, user_input):
        openai_response, _ = self.chatbot.get_chat_response(user_input)
        self._insert_message(openai_response, "Bot")

    def _create_message_bubble(self, canvas, message, sender):
        # 定义气泡颜色和文本颜色
        sender_color = "#345B63"
        text_color = "black"
        bubble_color = "#DCF8C6" if sender == "You" else "#ECECEC"
        
        # 设置发送者标签和消息文本的字体
        sender_font = ("Helvetica", 10, "bold")
        message_font = ("Microsoft YaHei", 12)

        # 创建发送者名字标签
        sender_label = "User:" if sender == "You" else "ATRI:"
        sender_text_id = canvas.create_text(5, 5, anchor="nw", text=sender_label, fill=sender_color, font=sender_font)
        
        # 获取发送者标签的包围盒，以计算消息文本的起始位置
        sender_bbox = canvas.bbox(sender_text_id)
        sender_width = sender_bbox[2] - sender_bbox[0]
        
        # 创建文本气泡
        padding_x = 20
        padding_y = 10
        message_x = sender_width + 30  # 留出空间放置发送者名字
        text_id = canvas.create_text(message_x, padding_y, anchor="nw", text=message, fill=text_color,
                                    width=280, font=message_font)
        bbox = canvas.bbox(text_id)
        
        # 扩展包围盒以为文本四周添加一些额外的空间
        expanded_bbox = (bbox[0] - padding_x, bbox[1] - padding_y, bbox[2] + padding_x, bbox[3] + padding_y)

        # 创建矩形气泡
        canvas.create_rectangle(expanded_bbox, fill=bubble_color, outline=bubble_color)
        canvas.tag_raise(text_id)  # 将文本移至矩形上方

        # 根据调整后的包围盒设置Canvas的大小
        canvas.config(width=expanded_bbox[2] + 5, height=expanded_bbox[3] + 10)  # 留出空间放置发送者名字




    def _insert_message(self, message, sender):
        self.text_widget.config(state='normal')

        # 创建Canvas并且添加气泡
        canvas = tk.Canvas(self.text_widget, bg="white", highlightthickness=0)
        self._create_message_bubble(canvas, message, sender)

        # 将Canvas插入到Text组件中，并为每个气泡之间添加额外的空间
        self.text_widget.window_create('end', window=canvas)
        self.text_widget.insert('end', '\n\n')  # 添加两个空行作为气泡间隔


        # 自动滚动到文本区域的底部
        self.text_widget.see('end')
        
        # 禁用文本区域的编辑
        self.text_widget.config(state='disabled')
        
        # 清空输入框
        self.msg_entry.delete(0, 'end')
        
        # 更新UI
        self.master.update_idletasks()


    def send_message_on_enter(self, event):
        self.send_message()

if __name__ == "__main__":
    root = tk.Tk()
    # OpenAI API Key
    openai_api_key = ''
    # 百度翻译ID
    baidu_appid = ''
    # 百度翻译Key
    baidu_secret_key = ''

    chatbot = IntegratedChatbot(openai_api_key, baidu_appid, baidu_secret_key)
    app = ChatApplication(root, chatbot)
    root.mainloop()
