from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
import threading

import chatglm_cpp
from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from PySide6.QtCore import Slot, QTimer
from PySide6.QtGui import QClipboard
from openai import OpenAI

class MinTab(QtWidgets.QWidget):
    update_chat_log_signal = Signal(str, str)  # 传递聊天信息更新的信号，包括内容和发送者
    def __init__(self,api_key):
        super().__init__()
        self.language = None
        self.message_thread=None
        self.setObjectName("MinTab")
        self.setGeometry(10, 10, 400, 300)
        self.setMinimumSize(400, 300)
        self.setMaximumSize(400, 300)
        self.setStyleSheet("background-color: white;")
        self.api_key = api_key

        self.chat_log = QtWidgets.QPlainTextEdit(self)
        self.chat_log.setObjectName("plainTextEdit")
        self.chat_log.setReadOnly(True)

        # 获取剪贴板实例
        self.clipboard = QClipboard()
        # 上一次剪贴板内容
        self.last_clipboard_text = ""

        # 使用定时器实时监测剪贴板内容
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_clipboard)
        self.timer.start(100)  # 设置定时器间隔（毫秒）


        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setObjectName("comboBox")
        languages = ["Chinese", "English", "German", "French", "Japanese"]
        flags = ["icon/China.png", "icon/America.png", "icon/Germany.jpg", "icon/France.jpg", "icon/Japan.png"]
        for lang, flag in zip(languages, flags):
            self.comboBox.addItem(QIcon(flag), lang)

        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")

        self.restart_button = QtWidgets.QPushButton(self)
        self.restart_button.setObjectName("pushButton_2")

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setObjectName("label_2")

        self.comboBox_1 = QtWidgets.QComboBox(self)
        self.comboBox_1.setObjectName("comboBox_1")
        styles = ["gpt-3.5-turbo", "gpt-4", "Local Model"]
        for style in styles:
            self.comboBox_1.addItem(style)

        self.gridLayoutWidget = QtWidgets.QWidget(self)
        self.gridLayoutWidget.setGeometry(10, 10, 381, 281)
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.addWidget(self.chat_log, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.comboBox, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.restart_button, 3, 0, 1, 2)
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.comboBox_1, 2, 1, 1, 1)

        self.label.setText(QCoreApplication.translate("Form", u"Target Language：", None))
        self.restart_button.setText(QCoreApplication.translate("Form", u"Restart", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Selected Model:", None))

        self.demo_ui()

        self.update_chat_log_signal.connect(self.update_chat_log)
        self.restart_button.clicked.connect(self.clear)

    @Slot()
    def check_clipboard(self):
        clipboard_text = self.clipboard.text()
        if clipboard_text != self.last_clipboard_text:
            # 如果剪贴板中的文本与上一次不同，执行翻译
            self.chat_log.clear()
            self.translate(clipboard_text)
            self.last_clipboard_text = clipboard_text

    # 翻译功能
    @Slot()
    def translate(self, message):
        self.language = self.comboBox.currentText()
        self.selected_api = self.comboBox_1.currentText()
        if not message:
            return  # 如果没有输入，返回
        selected_language = self.language
        request = f"Please translate the following sentence to {selected_language},give me translation outcome without anything else: {message}"
        #self.update_chat_log_signal.emit(message, "user")

        try:
            if self.selected_api == "local model":
                self.message_thread = threading.Thread(target=self.local_translate_message, args=(request,))
            else:
                self.message_thread = threading.Thread(target=self.translate_message, args=(request,))
            self.message_thread.start()
        except Exception as e:
            print(f"Exception in translate: {str(e)}")

    # 调用api让gpt翻译
    def translate_message(self, message):
        try:
            user_message = {"role": "user", "content": message}

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.selected_api,
                messages=[user_message],  # Use the conversation history
                stream=True
            )

            self.update_chat_log_signal.emit("", "gpt-start-translation")
            for chunk in response:  # 遍历数据流的事件
                chunk_message = chunk.choices[0].delta.content  # 提取消息
                if chunk_message is not None:
                    self.update_chat_log_signal.emit(chunk_message, "gpt-translation")
            self.update_chat_log_signal.emit("", "gpt-end-translation")


        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")

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

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Emit the signal to update the chat log with the error message
            self.update_chat_log_signal.emit(error_msg, "error")


    # 清空聊天记录
    @Slot()
    def clear(self):
        self.chat_log.clear()

    # 更新聊天记录并设置外观颜色
    @Slot(str, str)
    def update_chat_log(self, message, message_type):
        response_cursor = self.chat_log.textCursor()
        if message_type == "user":
            response_cursor.insertHtml("<span style='color: black; font-style: italic;'>You: </span>")
            response_cursor.insertText(f"{message}\n\n")
        elif message_type == "gpt-start-translation":
            response_cursor.insertHtml("<span style='color: blue;'> </span>")
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

    def demo_ui(self):
        self.chat_log.setStyleSheet("""
                    QPlainTextEdit {
                        padding: 8px 15px;
                        background-color: #ffffff;
                        border: 1px solid #1f618d;
                        color: #1f618d;
                        font-size: 13px;
                        border-radius: 5px;
                    }
                """)
        self.comboBox.setStyleSheet("""
                    QComboBox {
                        padding: 8px 15px;
                        background-color: #ffffff;
                        border: 1px solid #1f618d;
                        color: #1f618d;
                        font-size: 13px;
                        border-radius: 5px;
                    }

                    QComboBox:hover {
                        background-color: #f2f2f2;
                    }

                    QComboBox:pressed, QComboBox::drop-down:pressed {
                        background-color: #d5dbdb;
                    }

                    QComboBox::drop-down {
                        border: none;
                    }

                    QComboBox::down-arrow {
                        image: url(/path/to/your/icon.png);
                    }

                    QComboBox QAbstractItemView {
                        border: 1px solid #1f618d;
                        selection-background-color: #d5dbdb;
                        color: #1f618d;
                    }
                """)
        self.restart_button.setStyleSheet("""
                    QPushButton {
                        padding: 8px 15px;
                        background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                          stop: 0 #ffffff, stop: 1 #ffffff);
                        border: 1px solid #e74c3c;
                        color: #e74c3c;
                        font-size: 13px;
                        border-radius: 5px;
                    }

                    QPushButton:hover {
                        background-color: #f2f2f2;
                        border-color: #e74c3c;
                        color: #e74c3c;
                    }

                    QPushButton:pressed {
                        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                    stop: 0 #e74c3c, stop: 1 #ff6666);
                        border-color: #e74c3c;
                        color: #ffffff;
                        border-style: inset;
                    }
                """)

        self.comboBox_1.setStyleSheet("""
                    QComboBox {
                        padding: 8px 15px;
                        background-color: #ffffff;
                        border: 1px solid #1f618d;
                        color: #1f618d;
                        font-size: 13px;
                        border-radius: 5px;
                    }

                    QComboBox:hover {
                        background-color: #f2f2f2;
                    }

                    QComboBox:pressed, QComboBox::drop-down:pressed {
                        background-color: #d5dbdb;
                    }

                    QComboBox::drop-down {
                        border: none;
                    }

                    QComboBox::down-arrow {
                        image: url(/path/to/your/icon.png);
                    }

                    QComboBox QAbstractItemView {
                        border: 1px solid #1f618d;
                        selection-background-color: #d5dbdb;
                        color: #1f618d;
                    }
                """)

