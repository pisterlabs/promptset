# This script is supposed to take job description and make a resume according to it.
import openai
import sys
import pandas as pd
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtQuick import QQuickWindow
from PyQt6.QtWidgets import QApplication, QDialog, QFormLayout
from PyQt6.QtWidgets import (QPushButton, QLineEdit, QPlainTextEdit, QLabel)
from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot, QThread
from docEditor import DocEditor


# Your OPENAI_API_KEY
openai.api_key = ""
prompt_text = pd.read_csv("prompt.csv", encoding="cp1252")
# job_text["prompt1"][0] - first part of prompt
# job_text["prompt2"][0] - second part of prompt
PROMPT1 = prompt_text["prompt1"][0]
PROMPT2 = prompt_text["prompt2"][0]
# class worker(QObject):
#     finished = pyqtSignal()
#     progress = pyqtSignal(int)

#     def run(self):
#         "Long running tasks"


class Form(QDialog):
    
    
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)

        self.company_name = ""
        self.label = QLabel("Company Name:")
        self.company_text_box = QLineEdit()
        self.jd_label = QLabel("Job Description:")


        self.le = QPlainTextEdit()
        self.le.setObjectName("host")
        # self.le.setText("Host")

        self.pb = QPushButton()
        self.pb.setObjectName("connect")
        self.pb.setText("GENERATE RESUME")
        self.pb.clicked.connect(self.button_click)

        layout = QFormLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.company_text_box)
        layout.addWidget(self.jd_label)
        layout.addWidget(self.le)
        layout.addWidget(self.pb)
        self.setLayout(layout)
        self.setGeometry(100, 50, 800, 800)
        self.setWindowTitle("Resume AI")
    
    def get_gpt_response(self, message):
        prompt=[
            {"role": "system", "content": "You are a professional technical resume writer",
             "role": "user", "content":message}]
        response = openai.ChatCompletion.create(
          model = "gpt-4",
          messages = prompt,
          temperature=1,
          max_tokens=5000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
        )
        return response.choices[0].message.content

    def get_resume(self, update_resume):
        doc = DocEditor(update_resume)
        doc.run(self.company_name)

    def get_prompt(self, jd):
        # This function takes job description and makes a resume according to it.
        # jd - job description
        # resume - resume
        prompt = PROMPT1 + jd + PROMPT2
        QApplication.processEvents()
        resume = self.get_gpt_response(prompt)
        self.get_resume(resume)
        self.le.setPlainText(prompt + resume)

    def button_click(self):
        # jd(job description) is a QString object
        jd = self.le.toPlainText()
        self.company_name = self.company_text_box.text()
        self.get_prompt(jd)

    
if __name__ == '__main__':
    QQuickWindow.setSceneGraphBackend('software')
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    app.exec()
    engine = QQmlApplicationEngine()
