import os
import threading
import wave
from datetime import datetime

import chatglm_cpp
import openai
from openai import OpenAI
import pyaudio
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QTimer
import time

# 管理主应用程序窗口，处理与GPT模型的消息交换
class ChatTab(QtWidgets.QWidget):
    update_chat_log_signal = Signal(str, str)    # 传递聊天信息更新的信号，包括内容和发送者
    set_button_state_signal = Signal(bool)       # 传递按钮状态的信号
    set_api_button_state_signal = Signal(bool)   # 传递api状态的信号
    recording_state_signal = Signal(bool)        # 传递录制状态的信号

    def __init__(self, api_key):
        super().__init__()
        self.model_path = ""
        self.conversation_history = []
        self.record_thread = None
        self.selected_api = "gpt-3.5-turbo"
        self.api_key = api_key
        self.sender_button = 1

        # 创建一个计时器
        self.recording_timer = QTimer(self)  # 创建一个计时器
        self.recording_start_time = None
        self.recording_timer.timeout.connect(self.update_recording_time)  # 连接信号

        # 用于显示聊天记录的文本框
        self.chat_log = QtWidgets.QTextEdit(self)
        self.chat_log.setReadOnly(True)
        normal_height_log = self.chat_log.sizeHint().height()
        self.chat_log.setFixedHeight(normal_height_log * 1.6)
        self.chat_log.setTextInteractionFlags(Qt.NoTextInteraction)  # 禁止文本交互

        # 用户输入信息的文本框
        self.chat_input = QtWidgets.QTextEdit(self)
        self.chat_input.setPlaceholderText("Send a message")

        # 图形阴影效果，用于美化外观
        shadow_input = QtWidgets.QGraphicsDropShadowEffect(self.chat_input)
        shadow_input.setBlurRadius(15)
        shadow_input.setOffset(0, 0)
        shadow_input.setColor(QtGui.QColor("grey"))
        self.chat_input.setGraphicsEffect(shadow_input)

        normal_height_input = self.chat_input.sizeHint().height()
        self.chat_input.setFixedHeight(normal_height_input * 0.7)

        # 选择api和本地模型的按钮
        self.config_layout = QtWidgets.QHBoxLayout()
        self.api_group_box = QtWidgets.QGroupBox("Model:")
        self.api_group_box_layout = QtWidgets.QVBoxLayout(self.api_group_box)
        self.api_gpt35_radio_button = QtWidgets.QRadioButton("GPT-3.5")
        self.api_gpt4_radio_button = QtWidgets.QRadioButton("GPT-4")
        self.api_local_model_radio_button = QtWidgets.QRadioButton("Local Model")
        self.api_group_box_layout.addWidget(self.api_gpt35_radio_button)
        self.api_group_box_layout.addWidget(self.api_gpt4_radio_button)
        self.api_group_box_layout.addWidget(self.api_local_model_radio_button)
        self.api_gpt35_radio_button.toggled.connect(self.api_radio_button_toggled)
        self.api_gpt4_radio_button.toggled.connect(self.api_radio_button_toggled)
        self.api_local_model_radio_button.toggled.connect(self.api_radio_button_toggled)
        self.api_gpt35_radio_button.setChecked(True)

        # 设置参数的按钮
        self.par_group_box = QtWidgets.QGroupBox("Parameter:")
        self.par_layout = QtWidgets.QVBoxLayout(self.par_group_box)
        self.temperature_label = QtWidgets.QLabel("Temperature:")
        self.temperature_input = QtWidgets.QLineEdit("0.5", self)
        shadow_temp = QtWidgets.QGraphicsDropShadowEffect(self.temperature_input)
        shadow_temp.setBlurRadius(15)
        shadow_temp.setOffset(0, 0)
        shadow_temp.setColor(QtGui.QColor("grey"))
        self.temperature_input.setGraphicsEffect(shadow_temp)
        self.max_tokens_label = QtWidgets.QLabel("Max Tokens:")
        self.max_tokens_input = QtWidgets.QLineEdit("4000", self)
        shadow_max = QtWidgets.QGraphicsDropShadowEffect(self.max_tokens_input)
        shadow_max.setBlurRadius(15)
        shadow_max.setOffset(0, 0)
        shadow_max.setColor(QtGui.QColor("grey"))
        self.max_tokens_input.setGraphicsEffect(shadow_max)
        self.par_layout.addWidget(self.temperature_label)
        self.par_layout.addWidget(self.temperature_input)
        self.par_layout.addWidget(self.max_tokens_label)
        self.par_layout.addWidget(self.max_tokens_input)

        # 选择语言的下拉框
        self.trans_group_box = QtWidgets.QGroupBox("Translation Settings:")
        self.trans_layout = QtWidgets.QVBoxLayout(self.trans_group_box)

        self.language_label = QtWidgets.QLabel("Targating Language:")
        self.language_combobox = QtWidgets.QComboBox()
        languages = ["Chinese", "English", "German", "French", "Japanese"]
        flags = ["icon/China.png", "icon/America.png", "icon/Germany.jpg", "icon/France.jpg", "icon/Japan.png"]
        for lang, flag in zip(languages, flags):
            self.language_combobox.addItem(QIcon(flag), lang)

        # Targating Style
        self.style_label = QtWidgets.QLabel("Targating Style:")
        self.style_combobox = QtWidgets.QComboBox()
        styles = ["Normal", "Simple", "Academic", "interesting"]
        for style in styles:
            self.style_combobox.addItem(style)

        language_layout = QtWidgets.QVBoxLayout()
        language_layout.addWidget(self.language_label)
        language_layout.addStretch(1)
        language_layout.addWidget(self.language_combobox)
        language_layout.addStretch(1)
        
        style_layout = QtWidgets.QVBoxLayout()
        style_layout.addWidget(self.style_label)
        style_layout.addStretch(1)
        style_layout.addWidget(self.style_combobox)
        style_layout.addStretch(1)
        
        self.trans_layout.addLayout(language_layout)
        self.trans_layout.addLayout(style_layout)

        self.config_layout.addWidget(self.api_group_box)
        self.config_layout.addWidget(self.par_group_box)
        self.config_layout.addWidget(self.trans_group_box)
        self.config_layout.setStretchFactor(self.api_group_box, 3)
        self.config_layout.setStretchFactor(self.par_group_box, 7)
        self.config_layout.setStretchFactor(self.trans_group_box, 4)

        # 功能按钮
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.translate_button = QtWidgets.QPushButton("Translate", self)
        self.export_button = QtWidgets.QPushButton("Export", self)
        self.record_translate_button = QtWidgets.QPushButton("Record to Translate", self)
        self.record_send_button = QtWidgets.QPushButton("Record to Transcriptions", self)
        self.clear_button = QtWidgets.QPushButton("Clear", self)

        # 整体GUI布局
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.chat_log)
        self.layout.addLayout(self.config_layout)
        self.layout.addWidget(self.chat_input)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.send_button)
        self.button_layout.addWidget(self.translate_button)
        self.button_layout.addWidget(self.export_button)
        self.layout.addLayout(self.button_layout)

        self.record_layout = QtWidgets.QHBoxLayout()
        self.record_layout.addWidget(self.record_translate_button)
        self.record_layout.addWidget(self.record_send_button)
        self.record_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.record_layout)

        # 设置按钮外观
        self.demo_ui()

        self.send_button.clicked.connect(self.send)
        self.translate_button.clicked.connect(self.translate)
        self.update_chat_log_signal.connect(self.update_chat_log)
        self.set_button_state_signal.connect(self.set_button_state)
        self.set_api_button_state_signal.connect(self.set_api_button_state)
        # self.recording_state_signal.connect(self.update_button_text)
        self.export_button.clicked.connect(self.export_chat)
        self.record_send_button.clicked.connect(self.start_recording)
        self.record_translate_button.clicked.connect(self.start_recording)
        self.clear_button.clicked.connect(self.clear)

        self.recording = threading.Event()

    # 禁用功能按钮
    @Slot()
    def set_button_disabled(self, bool):
        self.send_button.setDisabled(bool)
        self.clear_button.setDisabled(bool)
        self.export_button.setDisabled(bool)
        self.translate_button.setDisabled(bool)
        self.record_send_button.setDisabled(bool)
        self.record_translate_button.setDisabled(bool)

    # 禁用api选择按钮
    @Slot()
    def set_api_button_disabled(self, bool):
        self.api_gpt35_radio_button.setDisabled(bool)
        self.api_gpt4_radio_button.setDisabled(bool)
        self.api_local_model_radio_button.setDisabled(bool)

    # 更新聊天记录并设置外观颜色
    @Slot(str, str)
    def update_chat_log(self, message, message_type):
        # This slot function updates the chat log with the message
        # message_type is either 'user' or 'gpt' to differentiate the source of the message
        response_cursor = self.chat_log.textCursor()
        if message_type == "user":
            response_cursor.insertHtml("<span style='color: black; font-style: italic;'>You: </span>")
            response_cursor.insertText(f"{message}\n\n")
        elif message_type == "gpt-start":
            response_cursor.insertHtml("<span style='color: green;'>GPT: </span>")
            response_cursor.insertText(f"{message}")
        elif message_type == "gpt":
            response_cursor.insertHtml("<span style='color: green;'> </span>")
            response_cursor.insertText(f"{message}")
        elif message_type == "gpt-end":
            response_cursor.insertHtml("<span style='color: green;'> </span>")
            response_cursor.insertText(f"{message}\n\n")
        elif message_type == "gpt-start-translation":
            response_cursor.insertHtml("<span style='color: blue;'>GPT: </span>")
            response_cursor.insertText(f"{message}")
        elif message_type == "gpt-translation":
            response_cursor.insertHtml("<span style='color: blue;'> </span>")
            response_cursor.insertText(f"{message}")
        elif message_type == "gpt-end-translation":
            response_cursor.insertHtml("<span style='color: blue;'> </span>")
            response_cursor.insertText(f"{message}\n\n")
        elif message_type == "error":
            response_cursor.insertHtml("<span style='color: red;'>ERROR: </span>")
            response_cursor.insertText(f"{message}\n\n")

    @Slot(bool)
    def set_button_state(self, state):
        self.set_button_disabled(state)

    @Slot(bool)
    def set_api_button_state(self, state):
        self.set_api_button_disabled(state)

    # 发送信息的功能
    @Slot()
    def send(self):
        # Disable the send button to prevent multiple clicks
        self.set_button_state_signal.emit(True)
        self.set_api_button_state_signal.emit(True)

        message = self.chat_input.toPlainText()  # Get the user input
        if not message:
            self.set_button_state_signal.emit(False)
            return  # If there is no input, return

        self.chat_input.clear()  # Clear the input box
        self.update_chat_log_signal.emit(message, "user")  # Update the chat log with the user message

        if self.selected_api == "local model":
            message_thread = threading.Thread(target=self.local_process_message, args=(message,))
        else:
            message_thread = threading.Thread(target=self.process_message, args=(message,))
        message_thread.start()

    # 通过api和gpt通信
    def process_message(self, message):
        try:
            user_message = {"role": "user", "content": message}
            self.conversation_history.append(user_message)
            client = OpenAI(api_key=self.api_key) 
            response = client.chat.completions.create(
                model=self.selected_api,
                messages=self.conversation_history,  # Use the conversation history
                temperature=float(self.temperature_input.text()),
                stream=True
            )

            collected_messages = ""
            self.update_chat_log_signal.emit("", "gpt-start")
            for chunk in response:  # 遍历数据流的事件
                chunk_message = chunk.choices[0].delta.content # 提取消息
                if chunk_message is not None:
                    collected_messages += chunk_message  # 保存消息
                    self.update_chat_log_signal.emit(chunk_message, "gpt")
            self.update_chat_log_signal.emit("", "gpt-end")
            self.conversation_history.append({"role": "assistant", "content": collected_messages})
            # Re-enable the send button once message processing is complete
            self.set_button_state_signal.emit(False)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")
            self.set_button_state_signal.emit(False)

    # 和部署在本地的ChatGLM-3 模型通信
    def local_process_message(self, message, max_length=2048, max_context_length=512, top_k=0, top_p=0.7, temp=0.95,
                              repeat_penalty=1.0):
        try:
            pipeline = chatglm_cpp.Pipeline(self.model_path)
            self.conversation_history.append(message)
            print(float(self.temperature_input.text()))
            # 2. 定义生成参数
            generation_kwargs = dict(
                max_length=max_length,
                max_context_length=max_context_length,
                do_sample=float(self.temperature_input.text()) > 0,
                top_k=top_k,
                top_p=top_p,
                temperature=float(self.temperature_input.text()),
                repetition_penalty=repeat_penalty,
                stream=True,
            )

            collected_messages = ""
            self.update_chat_log_signal.emit("", "gpt-start")
            for response_text in pipeline.chat(self.conversation_history, **generation_kwargs):
                collected_messages += response_text
                self.update_chat_log_signal.emit(response_text, "gpt")
            self.update_chat_log_signal.emit("", "gpt-end")
            self.conversation_history.append(collected_messages)
            self.set_button_state_signal.emit(False)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")
            self.set_button_state_signal.emit(False)

    # 翻译功能
    @Slot()
    def translate(self):
        # Disable the send button to prevent multiple clicks
        self.set_button_state_signal.emit(True)
        self.set_api_button_state_signal.emit(True)

        message = self.chat_input.toPlainText()  # Get the user input
        if not message:
            self.set_button_state_signal.emit(False)
            return  # If there is no input, return
        self.chat_input.clear()  # Clear the input box

        selected_language = self.language_combobox.currentText()
        selected_style = self.style_combobox.currentText()
        if selected_style == "interesting":
            style_text = "Use a relaxed, playful and cute translation style that needs to be distinguished from normal translation"
        elif selected_style == "academic":
            style_text = "Use a rigorous and academic translation style that needs to be distinguished from normal translation"
        elif selected_style == "simple":
            style_text = "Use a simple and concise translation style, only translate the general meaning, and need to be different from normal translation."
        else:
            style_text = " "
        request = f"Please translate the following sentence to {selected_language}，{style_text}, and give me translation outcome without anything else: {message}"
        self.update_chat_log_signal.emit(message, "user")

        if self.selected_api == "local model":
            message_thread = threading.Thread(target=self.local_translate_message, args=(request,))
        else:
            message_thread = threading.Thread(target=self.translate_message, args=(request,))
        message_thread.start()

    # 调用api让gpt翻译
    def translate_message(self, message):
        try:
            user_message = {"role": "user", "content": message}
            # openai.api_key = self.api_key  # Set the OpenAI API key
            client = OpenAI(api_key = self.api_key)
            response = client.chat.completions.create(
                model=self.selected_api,
                messages=[user_message],  # Use the conversation history
                stream=True
            )

            self.update_chat_log_signal.emit("", "gpt-start-translation")
            for chunk in response:  # 遍历数据流的事件
                chunk_message = chunk.choices[0].delta.content # 提取消息
                if chunk_message is not None:
                    self.update_chat_log_signal.emit(chunk_message, "gpt-translation")
            # response_text = response.choices[0].message.content
            # self.update_chat_log_signal.emit(response_text, "gpt-translation")
            self.update_chat_log_signal.emit("", "gpt-end-translation")

            # Re-enable the send button once message processing is complete
            self.set_button_state_signal.emit(False)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")
            self.set_button_state_signal.emit(False)

    # 让部署在本地的ChatGLM-3 模型翻译
    def local_translate_message(self, message, max_length=2048, max_context_length=512, top_k=0, top_p=0.7, temp=0.95,
                                repeat_penalty=1.0):
        try:
            pipeline = chatglm_cpp.Pipeline(self.model_path)
            # 2. 定义生成参数
            generation_kwargs = dict(
                max_length=max_length,
                max_context_length=max_context_length,
                do_sample=temp > 0,
                top_k=top_k,
                top_p=top_p,
                temperature=temp,
                repetition_penalty=repeat_penalty,
                stream=True,
            )

            self.update_chat_log_signal.emit("", "gpt-start-translation")
            for response_text in pipeline.chat([message], **generation_kwargs):
                self.update_chat_log_signal.emit(response_text, "gpt")
            self.update_chat_log_signal.emit("", "gpt-end-translation")
            self.set_button_state_signal.emit(False)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")
            self.set_button_state_signal.emit(False)

    # 切换API版本
    @Slot()
    def api_radio_button_toggled(self):
        if self.api_gpt35_radio_button.isChecked():
            self.selected_api = "gpt-3.5-turbo"
        elif self.api_gpt4_radio_button.isChecked():
            self.selected_api = "gpt-4"
        elif self.api_local_model_radio_button.isChecked():
            if self.model_path:
                self.selected_api = "local model"
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    "Local model not loaded.",
                )
                self.api_gpt35_radio_button.click()

    # 导出聊天记录为.txt文件
    @Slot()
    def export_chat(self):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f"chat_{timestamp}.txt"  # 文件名为chat_时间戳.txt

        try:
            with open(file_name, "w") as f:
                f.write(self.chat_log.toPlainText())
            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"The chat has been exported to {file_name}."
            )
        except Exception as e:
            # 错误处理
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred while exporting the chat: {str(e)}",
            )

    # 清空聊天记录
    @Slot()
    def clear(self):
        self.conversation_history.clear()
        self.chat_log.clear()

    # 录音
    def record(self):
        p = pyaudio.PyAudio()
        frames = []
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=10000,
                        input=True,
                        frames_per_buffer=2048)
        self.recording_state_signal.emit(True)
        # 开始录音
        while not self.recording.is_set():
            try:
                # 读取音频数据
                data = stream.read(2048)
                frames.append(data)
                print("1")
                if self.recording.wait(timeout=0.05):  # timeout设置为1秒检查一次
                    print("0")
                    break
            except Exception as e:
                print(e)
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存录音
        output_folder = "records"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "recorded_audio.wav")

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(10000)
            wf.writeframes(b''.join(frames))
            
        self.upload_audio(output_path)

    # @Slot(bool)
    # def update_button_text(self, is_recording):
    #     if self.sender_button == 1:
    #         self.record_send_button.setText("Stop Record" if is_recording else "Record to Transcriptions")
    #     else:
    #         self.record_translate_button.setText("Stop Record" if is_recording else "Record to Translate")

    # 修改 finish_recording 方法来停止计时器
    def finish_recording(self):
        self.recording.set()
        self.record_thread.join()
        self.recording_timer.stop()  # 停止计时器
        self.recording_start_time = None
        self.recording_state_signal.emit(False)
        self.recording.clear()
        if self.sender_button == 1:
            self.record_send_button.setText("Record to Transcriptions")
        elif self.sender_button == 2:
            self.record_translate_button.setText("Record to Translate")

    @Slot()
    def start_recording(self):
        sender = self.sender()  # 获取触发信号的按钮

        if sender == self.record_send_button:
            self.sender_button = 1
        else:
            self.sender_button = 2

        if self.sender_button == 1:
            if self.record_send_button.text() == "Record to Transcriptions":
                self.record_send_button.setText("Recording... 0s")
                self.recording_start_time = time.time()
                self.recording_timer.start(1000)  # 每秒更新一次
                self.record_thread = threading.Thread(target=self.record, args=())
                self.record_thread.start()
            else:
                self.finish_recording()
        else:
            if self.record_translate_button.text() == "Record to Translate":
                self.record_translate_button.setText("Recording... 0s")
                self.recording_start_time = time.time()
                self.recording_timer.start(1000)  # 每秒更新一次
                self.record_thread = threading.Thread(target=self.record, args=())
                self.record_thread.start()
            else:
                self.finish_recording()

    def upload_audio(self, audio_file_path):
            # Disable the send button to prevent multiple clicks
        self.set_button_state_signal.emit(True)
        self.set_api_button_state_signal.emit(True)

        self.update_chat_log_signal.emit("您发送了一条语音", "user")

        message_thread = threading.Thread(target=self.process_audio, args=(audio_file_path, ))
        message_thread.start()

    def process_audio(self, audio_file_path):
        try:
            # openai.api_key = self.api_key  # 设置openai API密钥
            client = OpenAI(api_key = self.api_key)

            # 构建API请求
            with open(audio_file_path, "rb") as audio_file:
                if self.sender_button == 1:
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                else:
                    transcript = client.audio.translations.create(model="whisper-1", file=audio_file)
            response = transcript.text

            self.update_chat_log_signal.emit("", "gpt-start-translation")
            text_chunks = response.split("\n")
            for chunk in text_chunks:  # 遍历数据流的事件
                self.update_chat_log_signal.emit(chunk, "gpt-translation")
            self.update_chat_log_signal.emit("", "gpt-end-translation")
            self.set_button_state_signal.emit(False)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")
            self.set_button_state_signal.emit(False)

    def update_recording_time(self):
        if self.recording_start_time:
            elapsed_time = time.time() - self.recording_start_time
            button_text = f"Recording... {int(elapsed_time)}s"
            if self.sender_button == 1:
                self.record_send_button.setText(button_text)
            elif self.sender_button == 2:
                self.record_translate_button.setText(button_text)

    # 设置按钮样式
    def demo_ui(self):
        self.chat_log.setStyleSheet("""
                            QTextEdit {
                                border: 2px solid black;
                                border-radius: 10px;
                                padding: 8px;
                                font-size: 16px;
                            }
                            QScrollBar:vertical {
                                border: 1px solid #696969;
                                background: #FFFFFF;
                                width: 10px; /* 调整滚动条宽度为10像素 */
                                margin: 22px 0 22px 0;
                                border-radius: 4px;
                            }
                            QScrollBar::handle:vertical {
                                background: #696969;
                                min-height: 20px;
                                border-radius: 4px;
                            }
                            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                                border: 1px solid grey;
                                background: #696969;
                                height: 15px;
                                border-radius: 4px;
                                subcontrol-origin: margin;
                            }
                            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                                background: none;
                            }
                        """)
        self.chat_input.setStyleSheet("""
                    QTextEdit {
                        border: none;
                        border-radius: 10px;
                        padding: 8px;
                        font-size: 16px;
                        background-color: #FFFFFF;
                    }
                    QScrollBar:vertical {
                        border: 1px solid #696969;
                        background: #FFFFFF;
                        width: 10px; 
                        margin: 22px 0 22px 0;
                        border-radius: 4px;
                    }
                    QScrollBar::handle:vertical {
                        background: #696969;
                        min-height: 20px;
                        border-radius: 4px;
                    }
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                        border: 1px solid grey;
                        background: #696969;
                        height: 15px;
                        border-radius: 4px;
                        subcontrol-origin: margin;
                    }
                    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                        background: none;
                    }
                """)

        self.api_group_box.setStyleSheet("""
                QGroupBox {
                    background-color: #FFFFFF;
                    border: 2px solid black;
                    border-radius: 5px;
                    margin-top: 1ex;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 3px;
                    font-size: 61px;  /* make font size larger */
                    font-weight: bold; /* make font bold */
                }
                """)

        # 设置GPT-3.5单选按钮的样式
        self.api_gpt35_radio_button.setStyleSheet("""
            QRadioButton {
                font-size: 16px;
                color: #333333;
                background-color: #C5E1A5;
                border: 2px solid #8BC34A;
                border-radius: 5px;
                padding: 6px 12px;
            }

            QRadioButton:checked {
                background-color: #8BC34A;
                color: #FFFFFF;
                border: 2px solid #8BC34A;
            }

            QRadioButton:hover {
                background-color: #D7EED6;
                border: 2px solid #C5E1A5;
            }

            QRadioButton:disabled {
                background-color: #E0E0E0;
                border: 2px solid #B0B0B0;
                color: #888888;
            }
        """)

        # 设置GPT-4单选按钮的样式
        self.api_gpt4_radio_button.setStyleSheet("""
            QRadioButton {
                font-size: 16px;
                color: #333333;
                background-color: #E6E6FA;
                border: 2px solid #9370DB;
                border-radius: 5px;
                padding: 6px 12px;
            }

            QRadioButton:checked {
                background-color: #9370DB;
                color: #FFFFFF;
                border: 2px solid #9370DB;
            }

            QRadioButton:hover {
                background-color: #D8BFD8;
                border: 2px solid #A020F0;
            }

            QRadioButton:disabled {
                background-color: #E0E0E0;
                border: 2px solid #B0B0B0;
                color: #888888;
            }
        """)

        # 设置Local Model单选按钮的样式
        self.api_local_model_radio_button.setStyleSheet("""
            QRadioButton {
                font-size: 16px;
                color: #333333;
                background-color: #FFDAB9;
                border: 2px solid #FF8C00;
                border-radius: 5px;
                padding: 6px 12px;
            }

            QRadioButton:checked {
                background-color: #FF6347;
                color: #FFFFFF;
                border: 2px solid #FF6347;
            }

            QRadioButton:hover {
                background-color: #FFE4B5;
                border: 2px solid #FFA500;
            }

            QRadioButton:disabled {
                background-color: #E0E0E0;
                border: 2px solid #B0B0B0;
                color: #888888;
            }
        """)

        self.par_group_box.setStyleSheet("""
                QGroupBox {
                    background-color: #FFFFFF;
                    border: 2px solid black;
                    border-radius: 5px;
                    margin-top: 1ex;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 3px;
                    font-size: 6px;  /* make font size larger */
                    font-weight: bold; /* make font bold */
                }
                """)
        self.temperature_input.setStyleSheet("""
                    QLineEdit {
                        border: none;  /* Remove border */
                        font-size: 16px;  /* Increase font size */
                        padding: 10px;  /* Add some padding */
                        background-color: #FFFFFF;  /* White background color */
                        border-radius: 10px;  /* Add rounded corners */
                    }
                """)
        self.temperature_input.setStyleSheet("""
                            QLineEdit {
                                border: none;  /* Remove border */
                                font-size: 16px;  /* Increase font size */
                                padding: 10px;  /* Add some padding */
                                background-color: #FFFFFF;  /* White background color */
                                border-radius: 10px;  /* Add rounded corners */
                            }
                        """)

        self.max_tokens_input.setStyleSheet("""
                    QLineEdit {
                        border: none;  /* Remove border */
                        font-size: 16px;  /* Increase font size */
                        padding: 10px;  /* Add some padding */
                        background-color: #FFFFFF;  /* White background color */
                        border-radius: 10px;  /* Add rounded corners */
                    }
                """)

        self.trans_group_box.setStyleSheet("""
                QGroupBox {
                    background-color: #FFFFFF;
                    border: 2px solid black;
                    border-radius: 5px;
                    margin-top: 1ex;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 3px;
                    font-size: 6px;  /* make font size larger */
                    font-weight: bold; /* make font bold */
                }
                """)
        self.language_label.setStyleSheet("""
                    QLabel {
                        font-size: 15px;  /* Decrease font size */
                        color: #000000;  /* Darker text color */
                    }
                """)
        self.language_combobox.setStyleSheet("""
                    QComboBox {
                        font-size: 16px;
                        padding: 10px;
                        border: 2px solid #4CAF50;  /* Green border */
                        border-radius: 10px;  /* Rounded corners */
                        background-color: #E8F5E9;  /* Light Green background */
                    }
                    QComboBox:hover {
                        border: 2px solid #388E3C;  /* Darker green border when hovered */
                    }
                    QComboBox QAbstractItemView {
                        font-size: 16px;
                        padding: 10px;
                        selection-background-color: #C8E6C9;  /* Slightly darker light green selection */
                        selection-color: black;  /* Black text for selected item */
                    }
                    QComboBox::drop-down {
                        border: 0;  /* No border for the drop-down arrow */
                        padding-right: 8px;  /* Adjust padding for the drop-down arrow */
                    }
                    QComboBox::down-arrow {
                        image: url(/path-to-your-icon/arrow-down-icon.png);  /* Customize drop-down arrow icon */
                    }
                    QComboBox::item:selected {
                        color: black;
                    }
                    QComboBox::item {
                        color: black;
                    }
                """)
        self.style_label.setStyleSheet("""
                    QLabel {
                        font-size: 15px;  /* Decrease font size */
                        color: #000000;  /* Darker text color */
                    }
                """)
        self.style_combobox.setStyleSheet("""
                    QComboBox {
                        font-size: 16px;
                        padding: 10px;
                        border: 2px solid #4CAF50;  /* Green border */
                        border-radius: 10px;  /* Rounded corners */
                        background-color: #E8F5E9;  /* Light Green background */
                    }
                    QComboBox:hover {
                        border: 2px solid #388E3C;  /* Darker green border when hovered */
                    }
                    QComboBox QAbstractItemView {
                        font-size: 16px;
                        padding: 10px;
                        selection-background-color: #C8E6C9;  /* Slightly darker light green selection */
                        selection-color: black;  /* Black text for selected item */
                    }
                    QComboBox::drop-down {
                        border: 0;  /* No border for the drop-down arrow */
                        padding-right: 8px;  /* Adjust padding for the drop-down arrow */
                    }
                    QComboBox::down-arrow {
                        image: url(/path-to-your-icon/arrow-down-icon.png);  /* Customize drop-down arrow icon */
                    }
                    QComboBox::item:selected {
                        color: black;
                    }
                    QComboBox::item {
                        color: black;
                    }
                """)

        self.send_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;  /* No fill color */
                        color: #4B0082;  /* Indigo */
                        border: 2px solid #4B0082;  /* Indigo */
                        border-radius: 15px;  /* Rounded corners */
                        padding: 10px 25px;  /* Padding: vertical, horizontal */
                        font-size: 16px;  /* Text size */
                        font-family: "Arial";  /* Font family */
                    }
                    QPushButton:hover {
                        border: 2px solid #9400D3;  /* Violet border on hover */
                        color: #9400D3;  /* Violet text on hover */
                    }
                    QPushButton:pressed {
                        border: 2px solid #8A2BE2;  /* Blue Violet border on pressed */
                        color: #8A2BE2;  /* Blue Violet text on pressed */
                    }
                    QPushButton:disabled {
                        color: #D3D3D3;  /* Light Gray text when disabled */
                        border: 2px solid #D3D3D3;  /* Light Gray border when disabled */
                    }
                """)
        self.translate_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;  /* No fill color */
                        color: #2E8B57;  /* Sea Green */
                        border: 2px solid #2E8B57;  /* Sea Green */
                        border-radius: 15px;  /* Rounded corners */
                        padding: 10px 25px;  /* Padding: vertical, horizontal */
                        font-size: 16px;  /* Text size */
                        font-family: "Arial";  /* Font family */
                    }
                    QPushButton:hover {
                        border: 2px solid #3CB371;  /* Medium Sea Green border on hover */
                        color: #3CB371;  /* Medium Sea Green text on hover */
                    }
                    QPushButton:pressed {
                        border: 2px solid #66CDAA;  /* Medium Aquamarine border on pressed */
                        color: #66CDAA;  /* Medium Aquamarine text on pressed */
                    }
                    QPushButton:disabled {
                        color: #D3D3D3;  /* Light Gray text when disabled */
                        border: 2px solid #D3D3D3;  /* Light Gray border when disabled */
                    }
                """)

        self.export_button.setStyleSheet("""
                            QPushButton {
                                background-color: #003366;  /* Dark Blue */
                                color: #FFFFFF;  /* White */
                                border: 2px solid #5F9E6E;  /* Dark Sea Green */
                                border-radius: 15px;  /* Rounded corners */
                                padding: 10px 25px;  /* Padding: vertical, horizontal */
                                font-size: 16px;  /* Text size */
                                font-family: "Arial";  /* Font family */
                            }
                            QPushButton:hover {
                                background-color: #336699;  /* Lighter Dark Blue */
                                border: 2px solid #20B2AA;  /* Light Sea Green */
                            }
                            QPushButton:pressed {
                                background-color: #6699CC;  /* Even Lighter Dark Blue */
                                border: 2px solid #3CB371;  /* Medium Sea Green */
                            }
                            QPushButton:disabled {
                                color: #D3D3D3;  /* Light Gray text when disabled */
                                border: 2px solid #D3D3D3;  /* Light Gray border when disabled */
                                background-color: #A9A9A9;  /* Dark Gray background when disabled */
                            }
                        """)
        self.record_translate_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;  /* No fill color */
                        color: #8B0000;  /* Dark Red */
                        border: 2px solid #8B0000;  /* Dark Red */
                        border-radius: 15px;  /* Rounded corners */
                        padding: 10px 25px;  /* Padding: vertical, horizontal */
                        font-size: 16px;  /* Text size */
                        font-family: "Arial";  /* Font family */
                    }
                    QPushButton:hover {
                        border: 2px solid #CD5C5C;  /* Indian Red border on hover */
                        color: #CD5C5C;  /* Indian Red text on hover */
                    }
                    QPushButton:pressed {
                        border: 2px solid #FF6347;  /* Tomato border on pressed */
                        color: #FF6347;  /* Tomato text on pressed */
                    }
                    QPushButton:disabled {
                        color: #D3D3D3;  /* Light Gray text when disabled */
                        border: 2px solid #D3D3D3;  /* Light Gray border when disabled */
                        background-color: #A9A9A9;  /* Dark Gray background when disabled */
                    }
                """)
        self.record_send_button.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;  /* No fill color */
                        color: #00008B;  /* Dark Blue */
                        border: 2px solid #00008B;  /* Dark Blue */
                        border-radius: 15px;  /* Rounded corners */
                        padding: 10px 25px;  /* Padding: vertical, horizontal */
                        font-size: 16px;  /* Text size */
                        font-family: "Arial";  /* Font family */
                    }
                    QPushButton:hover {
                        border: 2px solid #4682B4;  /* Steel Blue border on hover */
                        color: #4682B4;  /* Steel Blue text on hover */
                    }
                    QPushButton:pressed {
                        border: 2px solid #1E90FF;  /* Dodger Blue border on pressed */
                        color: #1E90FF;  /* Dodger Blue text on pressed */
                    }
                    QPushButton:disabled {
                        color: #D3D3D3;  /* Light Gray text when disabled */
                        border: 2px solid #D3D3D3;  /* Light Gray border when disabled */
                        background-color: #A9A9A9;  /* Dark Gray background when disabled */
                    }
                """)

        # 对clear_button的优化
        self.clear_button.setStyleSheet("""
                    QPushButton {
                        background-color: #DD4132;  /* Tomato Red */
                        color: #FFFFFF;  /* White */
                        border: 2px solid #FAE03C;  /* Daffodil Yellow */
                        border-radius: 15px;  /* Rounded corners */
                        padding: 10px 25px;  /* Padding: vertical, horizontal */
                        font-size: 16px;  /* Text size */
                        font-family: "Arial";  /* Font family */
                    }
                    QPushButton:hover {
                        background-color: #E94E77;  /* Pink */
                        border: 2px solid #FFD662;  /* Sunflower Yellow */
                    }
                    QPushButton:pressed {
                        background-color: #D2386C;  /* Rose */
                        border: 2px solid #ECC81A;  /* Golden Poppy */
                    }
                    QPushButton:disabled {
                        color: #D3D3D3;  /* Light Gray text when disabled */
                        border: 2px solid #D3D3D3;  /* Light Gray border when disabled */
                        background-color: #A9A9A9;  /* Dark Gray background when disabled */
                    }
                """)