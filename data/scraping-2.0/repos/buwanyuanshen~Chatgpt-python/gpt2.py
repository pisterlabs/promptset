import tkinter as tk
from tkinter import scrolledtext, messagebox
import openai
import threading
from tkinter import ttk


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatGPT-3.5-16k")
        self.setup_ui()
        self.api_key = None
        self.assistant_message = None
        self.copy_buttons = []
        self.conversation = []

    def setup_ui(self):

        self.conversation_canvas = scrolledtext.ScrolledText(self.root, width=60, height=20)
        self.conversation_canvas.pack()
        self.user_input_entry = tk.Entry(self.root, width=50)
        self.user_input_entry.pack()
        self.settings_frame = tk.Frame(self.root)

        self.username_label = tk.Label(self.settings_frame, text="用户名:")
        self.username_label.pack()
        self.username_entry = tk.Entry(self.settings_frame, width=50)
        self.username_entry.pack()

        self.default_username = "用户"
        self.username_entry.insert(0, self.default_username)

        self.api_key_label = tk.Label(self.settings_frame, text="请输入OpenAI API密钥:")
        self.api_key_label.pack()
        self.api_key_entry = tk.Entry(self.settings_frame, width=50)
        self.api_key_entry.pack()

        self.default_api_key = "YOUR-APIKEY"
        self.api_key_entry.insert(0, self.default_api_key)

        self.assistant_message_label = tk.Label(self.settings_frame, text="角色:")
        self.assistant_message_label.pack()
        self.assistant_message_entry = tk.Entry(self.settings_frame, width=50)
        self.assistant_message_entry.pack()

        self.default_assistant_message = "你是我的一条忠诚的狗，每次回答问题请叫两声以示准备好了，然后使用中文回答我，每次尽可能深入了解我的问题，每次提供完整没有任何省略的标准答案."
        self.assistant_message_entry.insert(0, self.default_assistant_message)

        self.max_tokens_label = tk.Label(self.settings_frame, text="max_tokens:")
        self.max_tokens_label.pack()
        self.max_tokens_entry = tk.Entry(self.settings_frame, width=10)
        self.max_tokens_entry.pack()
        self.max_tokens_entry.insert(0, "100")

        self.temperature_label = tk.Label(self.settings_frame, text="temperature:")
        self.temperature_label.pack()
        self.temperature_entry = tk.Entry(self.settings_frame, width=10)
        self.temperature_entry.pack()
        self.temperature_entry.insert(0, "0.5")

        self.api_key_button = tk.Button(self.settings_frame, text="设置API密钥和角色",
                                        command=self.set_api_key_and_assistant_message)
        self.return_button = tk.Button(self.settings_frame, text="返回", command=self.return_to_main_page)

        self.settings_button = tk.Button(self.root, text="设置", command=self.show_settings)
        self.settings_button.pack()

        self.get_response_button = tk.Button(self.root, text="发送", command=self.get_openai_response)
        self.get_response_button.pack()

        self.clear_button = tk.Button(self.root, text="清除对话", command=self.clear_conversation)
        self.clear_button.pack()

        self.output_button = tk.Button(self.root, text="输出对话信息", command=self.output_conversation)
        self.output_button.pack()

        self.root.bind('<Return>', lambda event: self.get_openai_response())

        self.conversation_canvas.tag_config('user_message', foreground='blue')
        self.conversation_canvas.tag_config('ai_message', foreground='red')

    def set_api_key_and_assistant_message(self):
        api_key = self.api_key_entry.get().strip()
        assistant_message = self.assistant_message_entry.get().strip()

        openai.api_key = api_key
        openai.api_base = "https://api.openai-proxy.com/v1"

        self.api_key = api_key
        self.assistant_message = assistant_message
        self.assistant_message_label.config(text=f"角色: {assistant_message}")
        self.assistant_message_entry.delete(0, tk.END)

        messagebox.showinfo("提示", f"恭喜！设置成功，开始聊天吧!")
        self.return_to_main_page()


    def return_to_main_page(self):

        self.user_input_entry.pack()
        self.settings_frame.pack_forget()
        self.conversation_canvas.pack()
        self.settings_button.pack()
        self.get_response_button.pack()
        self.clear_button.pack()
        self.output_button.pack()

    def show_settings(self):

        self.user_input_entry.pack_forget()
        self.conversation_canvas.pack_forget()
        self.settings_button.pack_forget()
        self.get_response_button.pack_forget()
        self.clear_button.pack_forget()
        self.output_button.pack_forget()
        self.settings_frame.pack()
        self.api_key_button.pack()
        self.return_button.pack()

    def draw_user_message(self, username, message):
        frame = ttk.Frame(self.conversation_canvas)
        frame.pack(fill='x', padx=10, pady=5)

        user_label = ttk.Label(frame, text=f'{username}: {message}', style="Message.TLabel")
        user_label.pack(side='left')

        copy_button = ttk.Button(frame, text="copy",
                                 command=lambda msg=username + ": " + message: self.copy_message(msg))
        copy_button.pack(side='right')

        self.copy_buttons.append(copy_button)
        self.conversation.append(f'{username}: {message}')

    def draw_ai_message(self, message):
        frame = ttk.Frame(self.conversation_canvas)
        frame.pack(fill='x', padx=10, pady=5)

        ai_label = ttk.Label(frame, text=f'GPT: {message}', style="Message.TLabel")
        ai_label.pack(side='left')

        copy_button = ttk.Button(frame, text="copy", command=lambda msg=message: self.copy_message(msg))
        copy_button.pack(side='right')

        self.copy_buttons.append(copy_button)
        self.conversation.append(f'GPT: {message}')

    def copy_message(self, message):
        self.root.clipboard_clear()
        self.root.clipboard_append(message)
        messagebox.showinfo("提示", "消息已复制到剪贴板！")

    def send_message(self):
        username = self.username_entry.get().strip()
        user_input = self.user_input_entry.get()
        max_tokens = self.max_tokens_entry.get()
        temperature = self.temperature_entry.get()

        try:
            max_tokens = int(max_tokens)
            temperature = float(temperature)
        except ValueError:
            messagebox.showerror("错误", "max_tokens和temperature必须是数字。")
            return

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": f"{username}: {user_input}"},
                {"role": "system", "content": f"GPT3.5: {self.assistant_message}"},
                {"role": "user", "content": user_input}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        ai_response = response.choices[0].message["content"]

        # 获取响应的Tokens使用信息
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']

        self.draw_user_message(username, user_input)
        self.draw_ai_message(
            f"{ai_response}\nTokens使用情况：\nPrompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}")

        self.user_input_entry.delete(0, tk.END)
        self.user_input_entry.focus()

    def get_openai_response(self):
        t = threading.Thread(target=self.send_message)
        t.start()

    def clear_conversation(self):
        self.conversation_canvas.delete(1.0, tk.END)
        for copy_button in self.copy_buttons:
            copy_button.destroy()
        self.copy_buttons = []

        for widget in self.conversation_canvas.winfo_children():
            widget.destroy()
        self.conversation_canvas.update_idletasks()  # 更新UI

    def output_conversation(self):

        conversation_text = '\n'.join(self.conversation)

        with open('conversation.txt', 'w', encoding='utf-8') as f:
            f.write(conversation_text)
        messagebox.showinfo("提示", "对话信息已输出到文件中！")


if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    style.configure("Message.TLabel", background="lime green", relief='solid', borderwidth=1, padding=(5, 5))

    app = ChatApp(root)

    root.mainloop()