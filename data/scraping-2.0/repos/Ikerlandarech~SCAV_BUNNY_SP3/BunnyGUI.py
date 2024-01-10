import openai
import sys
import os

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtGui import QTextCursor


os.environ["QT_QPA_PLATFORM"] = "wayland"

class ChatBotGUI(QWidget):

    response_received = pyqtSignal(str)

    def __init__(self, openai_api_key):
        super().__init__()

        self.setWindowTitle('Bunny GUI')
        self.setGeometry(100, 100, 1245, 700)
        self.setMaximumSize(1245, 700)

        self.init_ui()

        openai.api_key = "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"

        initial_chatbot_response = "Hi I'm bunny, I heard that in a few days you have the SCAV exam :) Luckily for you the students have been using me for years as an example for their ffmpeg commands, I know them all by heart from all the times they have used them with me so if you want to know anything about butterflies or about any ffmpeg commands let me know and I will be happy to help you, although I may look silly I have a lot of knowledge on the subject."
        self.append_message(initial_chatbot_response, is_user=False)

    def init_ui(self):

        bbb_background_path = "./assets/bbb-background.jpg"

        self.setStyleSheet(f"background-image: url({bbb_background_path}); background-position: top left; background-repeat: no-repeat;")
        self.conversation_box = QTextEdit(self) 
        self.conversation_box.setReadOnly(True)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Ask Bunny any question about audio and video encoding systems! He is very friendly and would love to help :)") 
        self.input_box.returnPressed.connect(self.process_user_input)

        font = self.conversation_box.document().defaultFont()
        font.setPointSize(12)
        self.conversation_box.document().setDefaultFont(font)

        layout = QVBoxLayout()
        layout.addWidget(self.conversation_box)
        layout.addWidget(self.input_box)

        self.setLayout(layout)

        self.show()
    
    def process_user_input(self):
        user_input = self.input_box.text()
        self.input_box.clear()
        self.append_message(user_input, is_user=True)

        timeout = QTimer(self)
        timeout.setSingleShot(True)
        timeout.timeout.connect(lambda: self.generate_and_append_response(user_input))
        timeout.start(500)

    def generate_and_append_response(self, user_input):
        response = self.generate_response(user_input)
        self.append_message(response, is_user=False)

    def append_message(self, message, is_user):
        current_text = self.conversation_box.toPlainText()
        role_label = '<b>User:</b>' if is_user else '<b>Bunny:</b>'
        text_color = 'darkgreen' if not is_user else 'black'
        formatted_message = f'<span style="color:{text_color}">{role_label} {message}</span>'

        if not is_user:
            words = message.split()
            self.conversation_box.append(f'<span style="color:{text_color}">{role_label} </span>')
            self.show_words_one_at_a_time(words)
        else:
            self.conversation_box.append(formatted_message)
            self.conversation_box.verticalScrollBar().setValue(self.conversation_box.verticalScrollBar().maximum())

    def show_words_one_at_a_time(self, words):
        cursor = self.conversation_box.textCursor()
        role_label = '<b>Bunny:</b>'
        text_color = 'darkblue'

        def show_next_word(index):
            if index < len(words):
                word = words[index]
                cursor.insertHtml(f'<span style="color:{text_color}"> {word} </span>')
                cursor.movePosition(QTextCursor.End)
                self.conversation_box.setTextCursor(cursor)
                QTimer.singleShot(50, lambda: show_next_word(index + 1))
            else:
                cursor.insertHtml('<br>')
                cursor.movePosition(QTextCursor.End)
                self.conversation_box.setTextCursor(cursor)
                self.conversation_box.verticalScrollBar().setValue(self.conversation_box.verticalScrollBar().maximum())

        show_next_word(0)

    def generate_response(self, user_input):
        bunny_identity = "Your name is Bunny, you are a very friendly overweight rabbit. You love to smell flowers and sleep in your burrow. One day while smelling a flower, you fall in love with a beautiful butterfly, which is killed by a group of 3 ruffians that you hate. You follow another butterfly but they want to keep killing it, so they try to annoy you by throwing stones and fruits in your face between the 3 of them. And when they see the opportunity, they kill the other butterfly with a stone and play with its corpse. You angrily hatch a master plan to avenge these 3 and manage to set a trap for them. Now you can live peacefully and go back to your burrow to sleep and smell flowers and follow butterflies. You are also an expertise on helping people with ffmpeg commands you know them all by heart so if they ask you about any ffmpeg command you should help them."
        stream = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant", "content": (f"{bunny_identity}\n")},
                {"role": "user", "content": f"{user_input}\n"},
                {"role": "assistant", "content": "You have to help the user providing the ffmpeg commands that he ask you to write. Take into account that the user has some knowledge already about the ffmpeg libraries so he has it all set and installed, just provide the line with the command. Remember that you are bunny and you should answer as he would, you are an expertise on ffmpeg commands. Remember to tell the user all the time that you love butterflies a lot. You can also provide some bunny noises on the text to make it look more realistic.\n"}
            ],
            stream=True,
        )
        response_content = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content

        return response_content.strip()


def main():

    app = QApplication(sys.argv)
    chatbot_gui = ChatBotGUI("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")
    chatbot_gui.response_received.connect(chatbot_gui.append_message)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
