from PySide2 import QtCore
from PySide2 import QtWidgets
from functools import partial
import openai
import os


class ChatGptWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        vbox = QtWidgets.QVBoxLayout()

        self.setGeometry(500, 300, 300, 150)
        self.setWindowTitle("Chat GPT Sample Widget")

        self.input = QtWidgets.QPlainTextEdit()
        self.input.move(20, 20)

        self.button = QtWidgets.QPushButton("Ask a question", self)
        self.button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.button.move(20, 100)

        vbox.addWidget(self.input)
        vbox.addWidget(self.button)

        self.setLayout(vbox)
        self.connect(self.button, QtCore.SIGNAL("clicked()"), self.exec_chatgpt)

    def exec_chatgpt(self):
        input_text = self.input.toPlainText()

        if len(input_text) > 0:
            print("---------------")
            print(">>> ", input_text)
            openai.api_key = "YOUR-API-KEY"
            completions = openai.Completion.create(
                engine="text-davinci-003",
                prompt=input_text,
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            for choice in completions.choices:
                print(choice.text)


dialog = ChatGptWidget()
dialog.show()
