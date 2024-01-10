import Constants
import sys
import openai 
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QTextEdit,
    QMessageBox
)


openai.api_key = Constants.API_KEY

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        #widgets being created
        self.logo_label = QLabel()
        self.logo_pixmap = QPixmap("assets/chat_image.webp").scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(self.logo_pixmap)

        self.input_label = QLabel('Ask a question:')
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText('Type here')
        self.answer_label = QLabel('Answer: ')
        self.answer_field = QTextEdit()
        self.answer_field.setReadOnly(True)
        self.submit_button = QPushButton('Submit')
        self.submit_button.setStyleSheet(
            """"
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3e8e41;
            }
            """
        )

        self.popular_question_group = QGroupBox('Try these questions')
        self.popular_questions_layout = QVBoxLayout()
        self.popular_question = ["What is the most famous sport in the world today?", "Best Date location recommendations", "Tell me a random cool fact"]
        self.question_buttons = []

        #layout down here
        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)

        #Add Input Field
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.submit_button)
        layout.addLayout(input_layout)

        #answer field
        layout.addWidget(self.answer_label)
        layout.addWidget(self.answer_field)

        for question in self.popular_question:
            button = QPushButton(question)
            button.setStyleSheet(
                """
                QPushButton{
                    background-color: #FFFFFF
                    border: 2px solid #00AEFF;
                    color: #00AEFF;
                    padding: 10px 20px;
                    font-size: 18px;
                    font-weight: bold
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #00AEFF;
                    color: #FFFFFF
                }
                """
            )
            button.clicked.connect(lambda _, q = question: self.input_field.setText(q))
            self.popular_questions_layout.addWidget(button)
            self.question_buttons.append(button)
        self.popular_question_group.setLayout(self.popular_questions_layout)
        layout.addWidget(self.popular_question_group)

        self.setLayout(layout)


        self.setWindowTitle('Test Bot')
        self.setGeometry(200,200,600, 600)
        
        #connect submit button
        self.submit_button.clicked.connect(self.get_answer)

    def get_answer(self):
        
        
        try:
            question = self.input_field.text()

            response = openai.chat.completions.create(
                model='gpt-3.5-turbo-1106',
                messages=[{"role": "system", "content": "You are very knowledgeable in sports, dating life, tips, and fun facts. Answer the following questions in a concise way, limit it to no more than 4 sentences. Use bullet points sometimes if you feel the need but don't limit yourself to it unnecessarily."},
                          {"role": "user", "content": f'{question}'}],
                max_tokens= 250,
                n=1,
                stop=None,
                temperature=0.5,
                #stream = True
            )

            answer = response.choices[0].message.content

            self.answer_field.setText(answer)

        except openai.APIError as e:
            QMessageBox.critical(self, "OpenAI Error", f"An error occurred with OpenAI: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")



if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

