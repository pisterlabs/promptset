import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QPainter, QPalette, QBitmap, QColor
import os
import openai

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.txt'), 'r') as config_file:
    openai_key = next((line.split("'")[1] for line in config_file if line.startswith("openai.api_key")), None)
if not openai_key:
    raise ValueError("API keys not found in the config file.")

openai.api_key = openai_key

class CircleButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(15, 15)
        self.setMask(self.create_ellipse_mask(self.size()))
        self.setStyleSheet("background-color: rgba(128, 128, 128, 0.8); border-radius: 7.5px;")

    def create_ellipse_mask(self, size):
        mask = QBitmap(size)
        mask.fill(Qt.color0)
        painter = QPainter(mask)
        painter.setBrush(Qt.color1)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, size.width(), size.height())
        painter.end()
        return mask

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.palette().color(QPalette.Button))
        painter.drawEllipse(self.rect())
        painter.end()


class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 330, 230)

        self.chat_layout = QVBoxLayout(self)
        self.chat_box = QTextEdit()
        self.chat_box.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.25); /* 80% transparent black */border: 4px solid rgba(255, 255, 255, 0.2);color: white;"

        )
        self.chat_box.setTextColor(Qt.white)

        self.chat_layout.addWidget(self.chat_box)

        self.user_input_layout = QHBoxLayout()
        self.user_input = QTextEdit()
        self.user_input.setFixedHeight(30)
        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(round(self.send_button.sizeHint().width() * 0.6))
        self.send_button.setStyleSheet("background-color: rgba(0, 0, 0, 0.35); color: rgba(255, 255, 255, 0.55);")
        self.send_button.clicked.connect(self.send_message)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setFixedWidth(round(self.clear_button.sizeHint().width() * 0.45))
        self.clear_button.setStyleSheet("background-color: rgba(0, 0, 0, 0.35); color: rgba(255, 255, 255, 0.55);")
        self.clear_button.clicked.connect(self.clear_message)

        self.user_input_layout.addWidget(self.user_input)
        self.user_input_layout.addWidget(self.send_button)
        self.user_input_layout.addWidget(self.clear_button)

        # Set the user's text box color
        user_input_palette = self.user_input.palette()
        user_input_palette.setColor(QPalette.Base, self.send_button.palette().color(QPalette.Button))
        user_input_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        self.user_input.setPalette(user_input_palette)

        self.circle_button = CircleButton()
        self.circle_button.clicked.connect(self.toggle_circle)

        self.circle_button2 = CircleButton()
        self.circle_button2.clicked.connect(self.toggle_chatbox_size)

        self.chat_layout.addLayout(self.user_input_layout)

        self.user_input.installEventFilter(self)

        self.user_input_layout.insertWidget(0, self.circle_button)
        self.user_input_layout.insertWidget(1, self.circle_button2)

        self.circle_grey = False
        self.moving = False

    def send_message(self):
        user_input = self.user_input.toPlainText()
        self.update_chat_box("Waiting for response...")

        # Use QTimer to delay the chatbot response
        QTimer.singleShot(200, lambda: self.generate_response(user_input))

    def generate_response(self, user_input):
        response = self.get_openai_response(user_input)
        self.update_chat_box(response)

    def clear_message(self):
        self.user_input.clear()
        self.chat_box.clear()

    def get_openai_response(self, user_input):
        openai.api_key = openai.api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="You are CaliBot. Reply using 250 words or less while maintaining proper language etiquette: " + user_input,
            max_tokens=350,
            temperature=0.7,
            n=1
        )
        return response['choices'][0]['text']

    def update_chat_box(self, response):
        self.chat_box.setPlainText(response)

    def eventFilter(self, obj, event):
        if obj == self.user_input and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    def toggle_circle(self):
        self.circle_grey = not self.circle_grey
        if self.circle_grey:
            self.circle_button.setStyleSheet("background-color: rgba(128, 128, 128, 0.2);")
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.show()
        else:
            self.circle_button.setStyleSheet("background-color: rgba(128, 128, 128, 0.8);")
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.show()

    def toggle_chatbox_size(self):
        current_width = self.width()
        current_height = self.height()
        if current_width == 330 and current_height == 230:
            self.setGeometry(self.x(), self.y(), 400, 500)
        elif current_width == 400 and current_height == 500:
            self.setGeometry(self.x(), self.y(), 50, 50)
        else:
            self.setGeometry(self.x(), self.y(), 330, 230)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.moving = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.moving and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.moving = False


app = QApplication(sys.argv)

# Change the color and transparency of the text box
style = """
    QTextEdit {
        background-color: rgba(128, 128, 128, 0.05);
        border: 4px solid rgba(255, 255, 255, 0.05);
        color: white;
        background-color: white);
    }
"""
app.setStyleSheet(style)

widget = ChatWidget()
widget.show()
sys.exit(app.exec_())
