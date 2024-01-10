from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QPushButton, QLabel, QSizePolicy
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
import sys
import openai
import configparser
import datetime
import os

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['OpenAI']['api_key']

class SettingsWindow(QWidget):
    def __init__(self):
        super(SettingsWindow, self).__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.apiKeyLabel = QLabel('OpenAI API Key:', self)
        layout.addWidget(self.apiKeyLabel)

        self.apiKeyEntry = QLineEdit(self)
        self.apiKeyEntry.setPlaceholderText('Enter OpenAI API Key')
        layout.addWidget(self.apiKeyEntry)

        self.saveBtn = QPushButton('Save', self)
        self.saveBtn.clicked.connect(self.save_api_key)
        layout.addWidget(self.saveBtn)

        self.setLayout(layout)

    def save_api_key(self):
        new_api_key = self.apiKeyEntry.text()
        config['OpenAI']['api_key'] = new_api_key
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        self.close()


class HistoryWindow(QWidget):
    def __init__(self, history):
        super(HistoryWindow, self).__init__()
        self.initUI(history)

    def initUI(self, history):
        self.setStyleSheet("background-color: #111111;")

        layout = QVBoxLayout()

        self.historyBox = QTextEdit(self)
        self.historyBox.setStyleSheet("background-color: #111111; color: white;")

        self.historyBox.setReadOnly(True)
        self.historyBox.setPlainText("\n".join(history))

        layout.addWidget(self.historyBox)

        self.setLayout(layout)


class MovieApp(QWidget):
    def __init__(self):
        super(MovieApp, self).__init__()
        self.initUI()
        self.search_history = []

    def initUI(self):
        layout = QVBoxLayout()
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.replyBox = QTextEdit(self)
        self.replyBox.setReadOnly(True)
        self.replyBox.setPlaceholderText('Your movie recommendations will appear here.')
        layout.addWidget(self.replyBox)

        self.tagEntry = QLineEdit(self)
        self.tagEntry.setPlaceholderText('Enter tags separated by commas')
        self.tagEntry.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.tagEntry)

        self.searchBtn = QPushButton('Search Movies', self)
        self.searchBtn.clicked.connect(self.perform_search)
        layout.addWidget(self.searchBtn)

        self.settingsBtn = QPushButton('Settings', self)
        self.settingsBtn.clicked.connect(self.open_settings)
        layout.addWidget(self.settingsBtn)

        self.historyBtn = QPushButton('History', self)
        self.historyBtn.clicked.connect(self.open_history)
        self.historyBtn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.exportBtn = QPushButton('Export', self)
        self.exportBtn.clicked.connect(self.export_history)
        self.exportBtn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        top_layout.addWidget(self.exportBtn, alignment=Qt.AlignRight)
        top_layout.addWidget(self.historyBtn, alignment=Qt.AlignRight)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.replyBox)
        main_layout.addWidget(self.tagEntry)
        main_layout.addWidget(self.searchBtn)
        main_layout.addWidget(self.settingsBtn)

        self.setLayout(main_layout)
        self.setLayout(layout)

        self.retry_count = 0
        self.exclude_movies = []
        self.searched_tags = []

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(15, 15, 15))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)

    def export_history(self):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        file_name = f"{current_date}-MovieGuru.txt"
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        with open(os.path.join(desktop, file_name), 'w') as f:
            f.write('\n'.join(self.search_history))
        self.replyBox.append("Search history exported successfully.")

    def on_text_changed(self):
        if self.retry_count == 0:
            self.searchBtn.setEnabled(True)

    def open_settings(self):
        self.settings_window = SettingsWindow()
        self.settings_window.setWindowTitle('Settings')
        self.settings_window.resize(300, 100)
        self.settings_window.show()

    def update_searched_tags(self, new_tags):
        self.searched_tags.extend(new_tags)

    def construct_request(self):
        tags = ', '.join(self.searched_tags)
        if self.exclude_movies:
            prompt = f"The user is not interested in these movies: {self.exclude_movies}. Provide a list of 5 different movies. Title (Release date), sorted in ascending order from these tags: {tags}. Feel free to use similar tags to find more movies. Do not add comments."
        else:
            prompt = f"Provide a list of 5 movies. Title (Release date), sorted in ascending order from these tags: {tags}. Do not add comments."
        return prompt

    def get_response_from_openai(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            api_key=api_key,
            temperature=0.75,
            top_p=0.8,
            best_of=6,
            frequency_penalty=0.5,
            presence_penalty=2
        )
        return response.choices[0].text.strip()

    def open_history(self):
        self.history_window = HistoryWindow(self.search_history)
        self.history_window.setWindowTitle('Search History')
        self.history_window.resize(450, 350)
        self.history_window.show()

    def perform_search(self):
        new_tags = [tag.strip() for tag in self.tagEntry.text().split(',')]
        self.update_searched_tags(new_tags)

        prompt = self.construct_request()
        response = self.get_response_from_openai(prompt)

        self.replyBox.setText(response)
        self.replyBox.append(f"\nCurrent tags being searched: {' '.join(self.searched_tags)}")

        self.search_history.append(f"Tags: {', '.join(new_tags)}\n\n{response}")

        self.tagEntry.clear()
        self.retry_count += 1

        self.searchBtn.setText("Search Again")

        if self.retry_count >= 4:
            self.replyBox.append("\nAdd New Tag to Enhance Search Functionality.")
            self.retry_count = 0
            self.exclude_movies = []
            self.searchBtn.setDisabled(True)
        else:
            self.exclude_movies.extend(response.split(','))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MovieApp()
    window.setWindowTitle('MovieGuru')
    window.resize(500, 400)
    window.show()
    sys.exit(app.exec_())
