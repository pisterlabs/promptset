# ui.py
import sys
import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, \
    QTabWidget, QTextEdit, QLineEdit, QTextBrowser, QSizePolicy, QScrollArea
import requests
from PyQt5.QtCore import Qt
from openai import OpenAI, OpenAIError
from PyQt5.QtGui import QFont
import datetime
import sounddevice as sd
import wavio as wv
from tools.speech2text import speech2text
from tools.text2image import text2image
from tools.text2lyrics import text2lyrics,assistant
OPENAI_API_KEY1='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'
OPENAI_API_KEY2='sk-5xH3t0BNBnGAEBJ9190a84A1DaA045C7B53aF04143E99049'

proxy_url = 'http://localhost'
proxy_port = '7890'  # Need to change it according to specific conditions
# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'
# Inside the MyWindow class definition

# Inside the MyWindow class definition

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # 设置默认的初始化提示
        self.init_prompt = "You are a helpful assistant who can write song lyrics."
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Voice Assistant")

        # 获取屏幕的宽度和高度
        screen_rect = QApplication.desktop().screenGeometry()
        screen_width, screen_height = screen_rect.width(), screen_rect.height()

        # 设置窗口初始大小为屏幕宽度的 80% 和高度的 60%
        self.resize(int(screen_width * 0.6), int(screen_height * 0.8))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # 创建选项卡部件
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.lyrics_scroll_area = None
        # 设置边框颜色为浅粉色
        border_color = '#FFB6C1'

        # 设置样式表
        self.setStyleSheet(f"""
               QMainWindow {{
                   border: 2px solid black;
                   background-color: {border_color};
               }}

               QTabWidget::pane {{
                   border: 2px solid black;
                   background-color: {border_color};
               }}

               QTabBar::tab {{
                   border: 2px solid black;
               }}

               QTabBar::tab:selected {{
                   background-color: {border_color};
                   color: black;
               }}
           """)
        # Customize font for cute and artistic style
        font = QFont()
        font.setFamily("Comic Sans MS")  # Replace with your preferred cute font
        font.setPointSize(12)  # Adjust the font size as needed
        # Explicitly set font for tab bar
        font_top_tabs = QFont()
        font_top_tabs.setFamily("Comic Sans MS")  # Replace with your preferred Japanese-style font
        font_top_tabs.setPointSize(14)  # Adjust the font size as needed

        # Set the font for the tab bar
        self.tabs.tabBar().setFont(font_top_tabs)

        # Apply style sheet for additional customization
        self.tabs.setStyleSheet(
            "QTabBar::tab { height: 40px; width: 150px; border: 2px solid black; border-radius: 15px; margin-top: 5px; }"
            "QTabBar::tab:hover { font-size: 16px; border: 4px solid black; border-radius: 15px; }"
        )

        # Tab 1: Speech to Text
        self.tab_speech_to_text = QWidget()
        self.tab_speech_to_text.setFont(font_top_tabs)
        self.layout_speech_to_text = QVBoxLayout(self.tab_speech_to_text)

        self.btn_choose_audio_file = QPushButton("Choose Audio File", self)
        self.btn_choose_audio_file.setFont(font)
        self.btn_choose_audio_file.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 10px; }"
            "QPushButton:hover { border: 2px solid #007B9C; font-weight: bold; }")
        self.btn_choose_audio_file.clicked.connect(self.choose_audio_file)

        self.btn_record_audio = QPushButton("Record Audio", self)
        self.btn_record_audio.setFont(font)
        self.btn_record_audio.setStyleSheet(
            "QPushButton { background-color: #008CBA; color: white; border-radius: 10px; }"
        "QPushButton:hover { border: 2px solid #007B9C; font-weight: bold; }")
        self.btn_record_audio.clicked.connect(self.record_audio)

        self.text_speech_to_text = QTextEdit()
        self.text_speech_to_text.setReadOnly(True)

        self.layout_speech_to_text.addWidget(self.btn_choose_audio_file)
        self.layout_speech_to_text.addWidget(self.btn_record_audio)
        self.layout_speech_to_text.addWidget(self.text_speech_to_text)
        self.tabs.addTab(self.tab_speech_to_text, "Speech to Text")

        # Tab 2: Text to Image
        self.tab_text_to_image = QWidget()
        self.layout_text_to_image = QVBoxLayout(self.tab_text_to_image)
        self.input_text_to_image = QLineEdit(self)
        self.input_text_to_image.setPlaceholderText("Enter Text for Image Generation")
        self.input_text_to_image.setStyleSheet(
            "QLineEdit { border: 2px solid #4CAF50; padding: 5px; border-radius: 10px; }")
        self.btn_text_to_image = QPushButton("Generate Image", self)
        self.btn_text_to_image.setFont(font)
        self.btn_text_to_image.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 10px; }"
        "QPushButton:hover { border: 2px solid #007B9C; font-weight: bold; }")
        self.btn_text_to_image.clicked.connect(self.text_to_image)
        self.text_text_to_image = QTextEdit()
        self.text_text_to_image.setReadOnly(True)

        # QLabel for displaying generated image
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)  # Center-align the image
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Button to save the image
        self.btn_save_image = QPushButton("Save Image")
        self.btn_save_image.setFont(font)
        self.btn_save_image.setStyleSheet("QPushButton { background-color: #008CBA; color: white; border: 2px solid #008CBA; }"
                                          "QPushButton:hover { border: 2px solid #007B9C; font-weight: bold; }")
        self.btn_save_image.clicked.connect(self.save_image)

        self.layout_text_to_image.addWidget(self.input_text_to_image)
        self.layout_text_to_image.addWidget(self.btn_text_to_image)
        self.layout_text_to_image.addWidget(self.label_image)
        self.layout_text_to_image.addWidget(self.btn_save_image)
        self.tabs.addTab(self.tab_text_to_image, "Text to Image")
        self.tab_text_to_image.setFont(font_top_tabs)

        # Tab 3: Text to Lyrics
        # 初始化聊天历史
        self.chat_history = []
        self.dialog_history = []
        self.tab_text_to_lyrics = QWidget()
        self.layout_text_to_lyrics = QVBoxLayout(self.tab_text_to_lyrics)

        # Create a scroll area for the Lyrics tab
        self.lyrics_scroll_area = QScrollArea(self.tab_text_to_lyrics)
        self.lyrics_scroll_area.setWidgetResizable(True)

        # Create a widget to hold the dialog history
        scroll_content_widget_lyrics = QWidget(self.lyrics_scroll_area)
        self.lyrics_scroll_area.setWidget(scroll_content_widget_lyrics)

        # Set up the layout for the dialog history in Lyrics tab
        dialog_history_layout_lyrics = QVBoxLayout(scroll_content_widget_lyrics)



        self.dialog_history_lyrics = QTextEdit(scroll_content_widget_lyrics)
        self.dialog_history_lyrics.setReadOnly(True)
        dialog_history_layout_lyrics.addWidget(self.dialog_history_lyrics)
        # Add the scroll area to the Lyrics tab layout
        self.layout_text_to_lyrics.addWidget(self.lyrics_scroll_area)

        self.input_text_to_lyrics = QTextEdit(self)
        self.btn_input_end = QPushButton("Input End", self)
        self.btn_input_end.setFont(font)
        self.btn_input_end.setStyleSheet(
            "QPushButton { background-color: #008CBA; color: white; border-radius: 10px; }"
            "QPushButton:hover { border: 2px solid #007B9C; font-weight: bold; }")
        self.btn_input_end.clicked.connect(self.handle_user_input)
        self.btn_text_to_lyrics = QPushButton("Get Lyrics", self)
        self.btn_text_to_lyrics.setFont(font)
        self.btn_text_to_lyrics.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 10px; }"
            "QPushButton:hover { border: 2px solid #007B9C; font-weight: bold; }")
        self.btn_text_to_lyrics.clicked.connect(self.text_to_lyrics)

        # Remove the unnecessary QTextBrowser
        # self.layout_text_to_lyrics.addWidget(self.dialog_history)

        self.layout_text_to_lyrics.addWidget(self.input_text_to_lyrics)
        self.layout_text_to_lyrics.addWidget(self.btn_input_end)
        self.layout_text_to_lyrics.addWidget(self.btn_text_to_lyrics)
        self.tabs.addTab(self.tab_text_to_lyrics, "Text to Lyrics")
        self.tab_text_to_lyrics.setFont(font_top_tabs)
        # 在输入文本框中显示 "User:"，并设置为只读
        self.input_text_to_lyrics.setPlainText("User:")


        # 初始化 ChatGPT 的提示
        self.init_prompt = "You are a helpful assistant who can write song lyrics."

    def text_to_lyrics(self):
        chat_history = [message["content"] for message in self.chat_history if message["role"] == "assistant"]

        # Check if there is any user input
        if not chat_history:
            self.dialog_history.append("Please enter valid text for chat.")
            return

        # Concatenate the entire chat history into a single string
        user_input = "\n".join(chat_history)

        # Call the assistant function to get ChatGPT's response

        # 使用：
        client = OpenAI(api_key="sk-5xH3t0BNBnGAEBJ9190a84A1DaA045C7B53aF04143E99049",
                        base_url="https://oneapi.xty.app/v1")

        try:
            # Call assistant function with the entire chat history as user input
            result = assistant(client=client, init_prompt=user_input, user_input='help me generate the lyrics based on history conversation@')

            # Display ChatGPT's response in the dialog history
            # Display ChatGPT's response in the dialog history
            if result:
                # Append ChatGPT's response to the chat history
                self.chat_history.append({"role": "assistant", "content": result})
                # Display ChatGPT's response in the second text box
                current_lyrics = self.dialog_history_lyrics.toPlainText()
                if current_lyrics:
                    self.dialog_history_lyrics.setPlainText(f"{current_lyrics}\nChatGPT: {result}\n")
                else:
                    self.dialog_history_lyrics.setPlainText(f"ChatGPT: {result}\n")

        except Exception as e:
            self.dialog_history.append(f"Error in chat: {str(e)}")

    def handle_user_input(self):
        # Get the user input

        user_input = self.input_text_to_lyrics.toPlainText()

        # Check if the user input is not empty
        if user_input.strip():
            self.input_text_to_lyrics.setPlainText(f"User:")
            # Add the user input to the dialog history
            self.dialog_history.append(f"User: {user_input}")

            # Call the assistant function to get ChatGPT's response
            api_key = OPENAI_API_KEY2
            # 使用：
            client = OpenAI(api_key="sk-5xH3t0BNBnGAEBJ9190a84A1DaA045C7B53aF04143E99049",
                            base_url="https://oneapi.xty.app/v1")
            result = assistant(client=client, init_prompt=self.init_prompt, user_input=user_input)

            # Check if the response from ChatGPT is not empty
            if result:
                # Add ChatGPT's response to the dialog history
                self.dialog_history.append(f"ChatGPT: {result}")

                # Add ChatGPT's response to chat history
                self.chat_history.append({"role": "assistant", "content": result})

                # Display both user input and ChatGPT's response in the same text box
                current_content = self.input_text_to_lyrics.toPlainText()
                if current_content:
                    self.input_text_to_lyrics.setPlainText(f"{user_input}\nChatGPT: {result}\nUser:")
                else:
                    self.input_text_to_lyrics.setPlainText(f"{user_input}\nChatGPT: {result}\nUser:")
            else:
                self.dialog_history.append("Error in chat")
        else:
            self.dialog_history.append("Please enter valid text for chat.")

    def choose_audio_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Audio File", "",
                                                   "Audio Files (*.wav *.mp3 *.m4a);;All Files (*)", options=options)

        if file_name:
            # Call the speech to text function with the selected file and API key
            api_key = OPENAI_API_KEY1
            result = speech2text(file_name, api_key)
            self.text_speech_to_text.setPlainText(result)

    # Inside the MyWindow class definition

    def text_to_image(self):
        # 获取用户输入的文本和 n
        prompt = 'generate a round-shaped album image based on lyrics of a song '+self.input_text_to_image.text()

        # Call the text to image function with the API key, user input, and n
        api_key = OPENAI_API_KEY2
        result_url = text2image(prompt, api_key)

        # 显示生成的图片
        if result_url:
            pixmap = self.load_pixmap_from_url(result_url)
            if pixmap:
                # 调整图片大小
                pixmap = pixmap.scaledToHeight(500)

                # 在 label_image 中显示图片
                self.label_image.setPixmap(pixmap)
                self.label_image.setAlignment(Qt.AlignCenter)  # 确保居中对齐
                self.label_image.setScaledContents(True)  # 确保图像自适应大小
                self.text_text_to_image.setText(result_url)
            else:
                self.text_text_to_image.setText("Error loading image.")

    def save_image(self):
        # Get the currently displayed pixmap
        current_pixmap = self.label_image.pixmap()

        if current_pixmap:
            # Ask the user for the file name and location
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Image (*.jpg);;All Files (*)", options=options)

            if file_name:
                # Save the pixmap to the specified file
                current_pixmap.save(file_name, "JPG")
                print(f"Image saved to: {file_name}")
            else:
                print("Image not saved.")
        else:
            print("No image to save.")


    def audio_to_text(self, audio_path):
        api_key = OPENAI_API_KEY1
        # Call the speech to text function with the selected file
        result = speech2text(audio_path,api_key)
        self.text_speech_to_text.setPlainText(result)
    def record_audio(self):
        # 录音配置
        samplerate = 16000
        duration = 5  # 录音秒数

        print("开始录音...")

        recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
        sd.wait()

        print("录音结束")

        # 保存录音文件
        if not os.path.exists("audio"):
            os.mkdir("audio")

        now = datetime.datetime.now()
        filename = now.strftime("audio/%Y-%m-%d_%H%M%S.wav")
        wv.write(filename, recording, samplerate, sampwidth=2)

        print("保存为音频文件:", filename)

        # 将录音文件路径显示在界面上
        self.text_speech_to_text.setPlainText(f"Recorded Audio File: {filename}")

        # 在录音结束后调用语音转文本功能
        self.audio_to_text(filename)
    def load_pixmap_from_url(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                pixmap = QPixmap()
                pixmap.loadFromData(response.content)
                return pixmap
            else:
                print(f"Error loading image: {response.status_code}")
        except Exception as e:
            print(f"Error loading image: {str(e)}")
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_window = MyWindow()
    my_window.show()
    sys.exit(app.exec_())
