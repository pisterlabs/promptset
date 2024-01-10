import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPlainTextEdit, QPushButton, \
    QHBoxLayout, QDialog, QMessageBox, QCheckBox, QComboBox, QSpinBox, QLineEdit, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
import openai
import datetime

# -----------------------------------------------------------
# gpt-3.5-turbo chat
#
# (C) 2023 A.M. RU, Stav.
# Released under GNU Public License (GPL)
# @author zeroc0de <zeroc0de@mail.ru>
# Date 2023.06.04
# -----------------------------------------------------------


class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки")
        self.resize(300, 200)

        self.save_history_checkbox = QCheckBox("Хранить историю чата")
        self.history_option_combo = QComboBox()
        self.history_option_combo.addItem("Не учитывать в ответе историю переписки")
        self.history_option_combo.addItem("Учитывать в ответе всю историю переписки")
        self.history_option_combo.addItem("Учитывать в ответе последние 5 сообщений")
        self.history_option_combo.addItem("Учитывать в ответе последние 3 сообщения")
        self.max_tokens_spinbox_label = QLabel("Количество токенов:")
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setMinimum(50)
        self.max_tokens_spinbox.setMaximum(2016)
        self.max_tokens_spinbox.setValue(50)
        
        self.settings_file = 'settings.json'
        
        self.api_key_label = QLabel("API ключ:")
        self.api_key_edit = QLineEdit()

        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_settings)


        layout = QVBoxLayout()
        layout.addWidget(self.save_history_checkbox)
        layout.addWidget(self.history_option_combo)
        layout.addWidget(self.max_tokens_spinbox_label)
        layout.addWidget(self.max_tokens_spinbox)
        layout.addWidget(self.api_key_label)
        layout.addWidget(self.api_key_edit)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as file:
                settings = json.load(file)
                save_history = settings.get('save_history', False)
                history_option = settings.get('history_option', 0)
                api_key = settings.get('api_key', '')
                max_tokens = settings.get('max_tokens', 50)
                self.save_history_checkbox.setChecked(save_history)
                self.history_option_combo.setCurrentIndex(history_option)
                self.max_tokens_spinbox.setValue(max_tokens)
                self.api_key_edit.setText(api_key)
        except FileNotFoundError:
            # Если файл settings.txt не найден, используются значения по умолчанию
            self.save_history_checkbox.setChecked(False)
            self.history_option_combo.setCurrentIndex(0)
            self.max_tokens_spinbox.setValue(50)
            self.api_key_edit.setText('')

    def save_settings(self):
        save_history = self.save_history_checkbox.isChecked()
        history_option = self.history_option_combo.currentIndex()
        max_tokens = self.max_tokens_spinbox.value()
        api_key = self.api_key_edit.text()
        settings = {
            "save_history": save_history,
            "history_option": history_option,
            "api_key": api_key,
            "max_tokens": max_tokens
        }
        with open(self.settings_file, "w") as file:
            json.dump(settings, file)
        self.accept()

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatGPT. Model: gpt-3.5-turbo")
        self.resize(400, 600)

        self.settings_dialog = None
        self.save_history = False
        self.history_option = 0
        self.max_tokens = 50
        self.api_key = ""
        self.messages = []

        self.chat_text_edit = QTextEdit()
        self.chat_text_edit.setReadOnly(True)

        # Заменяем QPlainTextEdit на QTextEdit
        self.input_text_edit = QTextEdit()
        self.input_text_edit.installEventFilter(self)

        self.send_button = QPushButton("Отправить")
        self.send_button.clicked.connect(self.send_message)
        self.settings_button = QPushButton("Настройки")
        self.settings_button.clicked.connect(self.open_settings)
        
        date = datetime.datetime.now()
        self.formatted_date = date.strftime("%Y-%m-%d")
        self.history_file = 'history/' + self.formatted_date + '_history.json'
        self.settings_file = 'settings.json'
        
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        main_layout.addWidget(self.chat_text_edit)
        main_layout.addWidget(self.input_text_edit)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.settings_button)

        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.load_settings()
        openai.api_key = self.api_key

    def display_message(self, message):
        self.chat_text_edit.append(message)

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as file:
                settings = json.load(file)
                self.save_history = settings.get('save_history', False)
                self.history_option = settings.get('history_option', 0)
                self.max_tokens = settings.get('max_tokens', 50)
                self.api_key = settings.get('api_key', '')
        except FileNotFoundError:
            self.save_history = False
            self.history_option = 0
            self.max_tokens = 50
            self.api_key = ''

    def save_chat_history(self):
        chat_history = {
            "messages": self.messages
        }
        if self.save_history:
            history = self.chat_text_edit.toPlainText()
            os.makedirs('history', exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as file:
                json.dump(chat_history, file, ensure_ascii=False)
                # file.write(history)

    def load_chat_history(self):
        try:
            with open(self.history_file, 'r', encoding='utf-8') as file:
                history = json.load(file)
                self.messages = history["messages"]
                # Отображение сохраненной истории в окне чата
                for message in self.messages:
                    role = message["role"]
                    content = message["content"]
                    self.display_message(f"{role}: - {content}")

                self.save_chat_history()  # Сохранение истории чата после загрузки

        except FileNotFoundError:
            # Если файл chat_history.txt не найден, не загружаем историю чата
            self.messages = []

    def send_message(self):
        message = self.input_text_edit.toPlainText().rstrip()
        self.input_text_edit.clear()
        if message:
            self.messages.append({"role": "user", "content": message})
            response = self.get_chatbot_response(message)
            # Сохранение ответа модели в истории чата
            self.messages.append({"role": "assistant", "content": response})

            self.display_message("\nuser: - " + message)
            self.display_message("\nassistant: - " + response)
            self.input_text_edit.clear()

            if self.save_history:
                self.save_chat_history()
            

    def get_chatbot_response(self, message):
        history = self.chat_text_edit.toPlainText()

        if self.history_option == 1:
            # Включение всей истории переписки в запрос
            prompt = history + "\nuser: " + message
        elif self.history_option == 2:
            # Включение последних 5 сообщений из истории переписки в запрос
            last_messages = self.messages[-5:]
            last_messages_content = [m["content"] for m in last_messages]
            last_messages_str = "\n".join(last_messages_content)
            prompt = last_messages_str + "\nuser: " + message
        elif self.history_option == 3:
            # Включение последних 3 сообщений из истории переписки в запрос
            last_messages = self.messages[-3:]
            last_messages_content = [m["content"] for m in last_messages]
            last_messages_str = "\n".join(last_messages_content)
            prompt = last_messages_str + "\nuser: " + message
        else:
            prompt = message

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content

    def open_settings(self):
        self.settings_dialog = SettingsDialog()
        self.settings_dialog.load_settings()
        if self.settings_dialog.exec_() == QDialog.Accepted:
            self.load_settings()
        self.settings_dialog = None

    def eventFilter(self, source, event):
        if (event.type() == QKeyEvent.KeyPress and
                source is self.input_text_edit and
                event.key() == Qt.Key_Return and
                event.modifiers() == Qt.ControlModifier):
            self.send_message()
            return True
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        self.save_chat_history()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.load_chat_history()
    window.show()
    sys.exit(app.exec_())
