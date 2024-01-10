import keyring
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QRadioButton, QMessageBox

import openai
from bardapi import Bard

class ModelThread(QThread):
    response_received = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        self.model = None
        self.prompt = ''

    def run(self):
        try:
            if self.model is not None:
                if self.model == 'bard':
                    response = Bard(self.token_bard).get_answer(self.prompt)['content']
                else:
                    openai.api_key = self.token_openai
                    if self.model == 'gpt-3.5-turbo':
                        response = openai.ChatCompletion.create(
                            model=self.model, 
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": self.prompt}
                            ]
                        )['choices'][0]['message']['content'].strip()
                    else:
                        response = openai.Completion.create(engine=self.model, prompt=self.prompt, max_tokens=100)['choices'][0]['text'].strip()
                self.response_received.emit(response)
        except Exception as e:
            self.response_received.emit(str(e))


class ModelAssistant(QWidget):
    def __init__(self):
        super().__init__()

        self.prompt_history = []
        self.prompt_index = -1

        self.token_bard_entry = QLineEdit(self)
        self.token_openai_entry = QLineEdit(self)
        self.prompt_entry = QLineEdit(self)
        self.result_text = QTextEdit(self)
        self.submit_button = QPushButton('Submit', self)
        self.copy_button = QPushButton('Copy', self)

        self.model_thread = ModelThread()

        layout = QVBoxLayout(self)

        token_bard_layout = QHBoxLayout()
        token_bard_layout.addWidget(QLabel('Enter BardAI token:'))
        token_bard_layout.addWidget(self.token_bard_entry)
        layout.addLayout(token_bard_layout)

        token_openai_layout = QHBoxLayout()
        token_openai_layout.addWidget(QLabel('Enter OpenAI token:'))
        token_openai_layout.addWidget(self.token_openai_entry)
        layout.addLayout(token_openai_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Select model:'))
        self.bard_button = QRadioButton('Bard')
        self.gpt3_turbo_button = QRadioButton('GPT-3.5-turbo')
        self.gpt4_button = QRadioButton('GPT-4')
        model_layout.addWidget(self.bard_button)
        model_layout.addWidget(self.gpt3_turbo_button)
        model_layout.addWidget(self.gpt4_button)
        layout.addLayout(model_layout)

        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel('Enter prompt:'))
        prompt_layout.addWidget(self.prompt_entry)
        prompt_layout.addWidget(self.submit_button)

        layout.addLayout(prompt_layout)
        layout.addWidget(self.result_text)
        layout.addWidget(self.copy_button)

        self.loader_label = QLabel('Generating...', self)
        layout.addWidget(self.loader_label)
        self.loader_label.hide()

        self.submit_button.clicked.connect(self.get_response)
        self.copy_button.clicked.connect(self.copy_response)
        self.model_thread.response_received.connect(self.show_response)

        self.load_token()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.get_response()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.copy_response()
        elif event.key() == Qt.Key_Up:
            if self.prompt_history:
                self.prompt_index = min(self.prompt_index + 1, len(self.prompt_history) - 1)
                self.prompt_entry.setText(self.prompt_history[self.prompt_index])
        elif event.key() == Qt.Key_Down:
            if self.prompt_history:
                self.prompt_index = max(self.prompt_index - 1, -1)
                if self.prompt_index == -1:
                    self.prompt_entry.clear()
                else:
                    self.prompt_entry.setText(self.prompt_history[self.prompt_index])

    def load_token(self):
        token_bard = keyring.get_password('model', 'api_token_bard')
        token_openai = keyring.get_password('model', 'api_token_openai')
        model = keyring.get_password('model', 'model')
        if token_bard:
            self.token_bard_entry.setText(token_bard)
        if token_openai:
            self.token_openai_entry.setText(token_openai)
        if model:
            if model == 'bard':
                self.bard_button.setChecked(True)
            elif model == 'gpt-3.5-turbo':
                self.gpt3_turbo_button.setChecked(True)
            elif model == 'gpt-4':
                self.gpt4_button.setChecked(True)

    def save_token(self):
        keyring.set_password('model', 'api_token_bard', self.token_bard_entry.text())
        keyring.set_password('model', 'api_token_openai', self.token_openai_entry.text())
        if self.bard_button.isChecked():
            keyring.set_password('model', 'model', 'bard')
        elif self.gpt3_turbo_button.isChecked():
            keyring.set_password('model', 'model', 'gpt-3.5-turbo')
        elif self.gpt4_button.isChecked():
            keyring.set_password('model', 'model', 'gpt-4')

    def get_response(self):
        self.save_token()
        token_bard = self.token_bard_entry.text()
        token_openai = self.token_openai_entry.text()
        prompt = self.prompt_entry.text()

        self.prompt_history.insert(0, prompt)
        self.prompt_index = -1

        if not token_bard or not token_openai:
            QMessageBox.warning(self, 'Warning', 'Please enter the API tokens.')
            return

        if self.bard_button.isChecked():
            self.model_thread.model = 'bard'
            self.model_thread.token_bard = token_bard
        elif self.gpt3_turbo_button.isChecked():
            self.model_thread.model = 'gpt-3.5-turbo'
            self.model_thread.token_openai = token_openai
        elif self.gpt4_button.isChecked():
            self.model_thread.model = 'gpt-4'
            self.model_thread.token_openai = token_openai

        self.model_thread.prompt = prompt
        self.model_thread.start()
        self.loader_label.show()

    def show_response(self, response):
        self.result_text.clear()
        self.result_text.setMarkdown(response)
        self.loader_label.hide()

    def copy_response(self):
        self.result_text.selectAll()
        self.result_text.copy()
