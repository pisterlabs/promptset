# gpt_api_thread.py

import datetime
import time
import openai
from openai import OpenAI
from PyQt6.QtCore import QThread, pyqtSignal
from settings_manager import SettingsManager

import logging

logger = logging.getLogger(__name__)


class GptApiThread(QThread):
    response_signal = pyqtSignal(str)
    new_user_message_signal = pyqtSignal(str)
    update_message_signal = pyqtSignal(str)

    def __init__(self, user_name):
        super().__init__()
        self.settings_manager = SettingsManager()
        try:
            openai_api_key = self.settings_manager.get_setting("openai_api_key")
            logger.info("Successfully loaded minimax API keys from settings")
        except Exception as e:
            logger.exception("❗️Error occurred")

        self.client = OpenAI(api_key=openai_api_key)

        self.assistant_id = "asst_40vLVijSiJ0cRONnIFPOaeas"  # asst_40vLVijSiJ0cRONnIFPOaeas | asst_38y14wqWGJsKn8XvlTgSqAzx``
        self.thread_id = self.create_thread()
        print("thread_id:", self.thread_id)
        self.user_name = user_name
        self.user_message = None
        # self.start_processing_signal.connect(self.chat_with_assistant)
        self.new_user_message_signal.connect(self.handle_new_user_message)

    def run(self):
        self.initialize_session()

    # Create a thread
    def create_thread(self):
        try:
            thread = self.client.beta.threads.create()
            return thread.id
        except openai.APIConnectionError:
            logger.exception("❗️Error occurred")
            print("无法连接到服务器，请检查网络连接。")
        except openai.RateLimitError:
            logger.exception("❗️Error occurred")
            print("已达到速率限制，请稍后再试。")
        except openai.APIStatusError as e:
            logger.exception("❗️Error occurred")
            print("接口返回非成功状态码:", e.status_code)
        except openai.APIError:
            logger.exception("❗️Error occurred")
            print("发生了一个 OpenAI API 错误。")
        return None

    # Creata a message
    def create_message(self, prompt, thread_id):
        try:
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt,
            )
            return message
        except openai.APIConnectionError:
            logger.exception("❗️Error occurred")
            print("无法连接到服务器，请检查网络连接。")
        except openai.RateLimitError:
            logger.exception("❗️Error occurred")
            print("已达到速率限制，请稍后再试。")
        except openai.APIStatusError as e:
            logger.exception("❗️Error occurred")
            print("接口返回非成功状态码:", e.status_code)
        except openai.APIError:
            logger.exception("❗️Error occurred")
            print("发生了一个 OpenAI API 错误。")
        return None

    # Run the assistant
    def run_thread(self, thread_id, assistant_id):
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
            )
            return run.id
        except openai.APIConnectionError:
            logger.exception("❗️Error occurred")
            print("无法连接到服务器，请检查网络连接。")
        except openai.RateLimitError:
            logger.exception("❗️Error occurred")
            print("已达到速率限制，请稍后再试。")
        except openai.APIStatusError as e:
            logger.exception("❗️Error occurred")
            print("接口返回非成功状态码:", e.status_code)
        except openai.APIError:
            logger.exception("❗️Error occurred")
            print("发生了一个 OpenAI API 错误。")
        return None

    # Check the status of the run
    def check_run_status(self, thread_id, run_id):
        try:
            run_list = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id,
            )
            return run_list.status
        except openai.APIConnectionError:
            logger.exception("❗️Error occurred")
            print("无法连接到服务器，请检查网络连接。")
        except openai.RateLimitError:
            logger.exception("❗️Error occurred")
            print("已达到速率限制，请稍后再试。")
        except openai.APIStatusError as e:
            logger.exception("❗️Error occurred")
            print("接口返回非成功状态码:", e.status_code)
        except openai.APIError:
            logger.exception("❗️Error occurred")
            print("发生了一个 OpenAI API 错误。")
        return None

    def retrieve_message_list(self, thread_id):
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id,
            )
            return messages.data
        except openai.APIConnectionError:
            logger.exception("❗️Error occurred")
            print("无法连接到服务器，请检查网络连接。")
        except openai.RateLimitError:
            logger.exception("❗️Error occurred")
            print("已达到速率限制，请稍后再试。")
        except openai.APIStatusError as e:
            logger.exception("❗️Error occurred")
            print("接口返回非成功状态码:", e.status_code)
        except openai.APIError:
            logger.exception("❗️Error occurred")
            print("发生了一个 OpenAI API 错误。")
        return None

    def generate_text_from_oai(self, system_prompt, user_message):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",  # gpt-3.5-turbo-1106 | gpt-4-1106-preview
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content
        except openai.APIConnectionError:
            logger.exception("❗️Error occurred")
            print("无法连接到服务器，请检查网络连接。")
        except openai.RateLimitError:
            logger.exception("❗️Error occurred")
            print("已达到速率限制，请稍后再试。")
        except openai.APIStatusError as e:
            logger.exception("❗️Error occurred")
            print("接口返回非成功状态码:", e.status_code)
        except openai.APIError:
            logger.exception("❗️Error occurred")
            print("发生了一个 OpenAI API 错误。")
        return None

    def initialize_session(self):
        # 初始化会话
        user_message = f"用户称呼：{self.user_name}"
        self.chat_with_assistant(user_message)

    def handle_new_user_message(self, user_message):
        # 处理新的用户输入
        self.chat_with_assistant(user_message)

    def chat_with_assistant(self, user_message):
        # 与助手对话的代码
        print("⭕️ 正在调用OpenAI API...")
        self.user_message = user_message
        self.create_message(self.user_message, self.thread_id)
        self.run_id = self.run_thread(self.thread_id, self.assistant_id)
        self.status = self.check_run_status(self.thread_id, self.run_id)

        while self.status != "completed":
            time.sleep(0.5)
            self.status = self.check_run_status(self.thread_id, self.run_id)
            if self.status == "failed":
                break

        # 如果线程完成，则获取消息
        if self.status == "completed":
            self.messages = self.retrieve_message_list(self.thread_id)
            response = self.messages[0].content[0].text.value

            # 发出带有API响应的信号
            self.response_signal.emit(response)

    def generate_file_title(self, file_type):
        self.save_location = self.settings_manager.get_setting("save_location")
        today = datetime.datetime.now()
        self.year_number = today.isocalendar()[0]
        self.week_number = today.isocalendar()[1]
        self.timestamp = today.strftime("%Y%m%d%H%M%S")
        self.title = f"{self.save_location}/{self.timestamp}-{self.year_number}w{self.week_number}-{file_type}.md"
        return self.title, self.year_number, self.week_number

    def save_chatlog(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chatlog_path, year, week = self.generate_file_title("chatlog")
        messages = self.retrieve_message_list(self.thread_id)[:-1]
        messages = reversed(messages)
        with open(chatlog_path, "a") as f:
            f.write(f"# {year}w{week} Weekly Review 对话记录\n\n")
            for i in messages:
                role = i.role
                text = i.content[0].text.value
                if role == "assistant":
                    f.write(f"**Echo**: {text}\n\n")
                else:
                    f.write(f"**{self.user_name}**: {text}\n\n")
            f.write("---\n\n")
            f.write("## Chagnelog\n\n")
            f.write(f"- {timestamp} 生成对话记录\n\n")
        return chatlog_path

    def generate_report(self, chatlog_path):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path, year, week = self.generate_file_title("report")

        system_prompt = """
            你是一个复盘报告写作专家。你需要根据一份对话记录，撰写一份周复盘报告。在这份报告中，你需要包含以下内容：
            1. 本周发生的重要事件
            2. 本周获得的最大成就
            3. 本周遇到的最大挑战
            4. 对话中十大高频词
            5. 本周的情绪状态
            6. 下周计划
            不需要写一级标题，只需要写二级标题即可。
        """

        with open(chatlog_path, "r") as f:
            user_message = f.read()

        response = self.generate_text_from_oai(system_prompt, user_message)

        with open(report_path, "w") as f:
            f.write(f"# {year}w{week} Weekly Review 复盘报告\n\n")
            f.write(response)
            f.write("---\n\n")
            f.write("## Changelog\n\n")
            f.write(f"- {timestamp} GPT-4 基于对话记录生成复盘报告\n\n")
        self.update_message_signal.emit("✅ 本周复盘报告已生成")
