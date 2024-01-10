from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QSlider,
    QTextEdit,
    QPushButton,
    QComboBox,
    QGridLayout,
    QLineEdit,
)
import openai
from cogs.worker import Worker
from cogs.apis import load_models, load_api_key
from datetime import datetime

load_api_key()


class Tab1(QWidget):
    def __init__(self, parent=None):
        super(Tab1, self).__init__(parent)

        # adding widgets
        self.tempAmount = float(0.1)
        self.answerAmount = 1
        self.tokenAmount = 10

        self.tempSlide = QSlider(Qt.Orientation.Horizontal)
        self.tokenSlide = QSlider(Qt.Orientation.Horizontal)
        self.amountSlide = QSlider(Qt.Orientation.Horizontal)

        self.promptEdit = QTextEdit()
        self.responseBox = QTextEdit()

        self.tokenStatus = QLabel("Tokens: 10")
        self.tempStatus = QLabel("Temp: 0.0")
        self.amountStatus = QLabel("Answers: 1")
        self.engineStatus = QLabel("Engine: ")
        self.finished = QLabel()

        self.sendButton = QPushButton("Send prompt")
        self.engine = QComboBox()
        model_list = load_models()
        self.engine.addItems(model_list)

        # connections
        self.engine.currentIndexChanged.connect(self.selection_change)

        self.tokenSlide.setRange(10, 1000)
        self.tokenSlide.valueChanged.connect(self.value_change)

        self.amountSlide.setRange(1, 5)
        self.amountSlide.valueChanged.connect(self.value_change)

        self.tempSlide.setRange(0, 10)
        self.tempSlide.valueChanged.connect(self.value_change)

        self.responseBox.setReadOnly(True)

        self.sendButton.clicked.connect(self.send_prompt_thread)

    def init_ui(self):
        print("Engine: ", self.engine.currentText)

        l = QGridLayout()
        l.aw = lambda w, r, c: l.addWidget(w, r, c)
        l.aw(self.tempStatus, 0, 0)
        l.aw(self.tempSlide, 0, 1)
        l.aw(self.tokenStatus, 1, 0)
        l.aw(self.tokenSlide, 1, 1)
        l.aw(self.amountStatus, 2, 0)
        l.aw(self.amountSlide, 2, 1)
        l.aw(self.engineStatus, 3, 0)
        l.aw(self.engine, 3, 1)
        l.addWidget(self.promptEdit, 4, 0, 1, 2)
        l.addWidget(self.sendButton, 5, 0, 1, 2)
        l.addWidget(self.responseBox, 6, 0, 1, 2)
        l.addWidget(self.finished, 7, 0, 1, 2)
        self.setLayout(l)

    def selection_change(self):
        self.cEngine = self.engine.currentText()
        self.engineStatus.setText("Engine: " + self.cEngine)

    def value_change(self):
        self.tokenAmount = self.tokenSlide.value()
        self.tempAmount = self.tempSlide.value() / 10
        self.answerAmount = self.amountSlide.value()
        self.tokenStatus.setText("Tokens: " + str(self.tokenAmount))
        self.tempStatus.setText("Temp: " + str(self.tempAmount))
        self.amountStatus.setText("Answers: " + str(self.answerAmount))

    def send_prompt_thread(self):
        self.sendButton.setEnabled(False)
        self.finished.setText("Sending prompt...")
        self.finished.repaint()
        self.send_prompt()
        self.sendButton.setEnabled(True)

    def send_prompt(self):
        print("Sending prompt...")
        print("Engine: ", self.cEngine)

        response = openai.Completion.create(
            engine=self.cEngine,
            prompt=self.promptEdit.toPlainText(),
            temperature=float(self.tempAmount),
            max_tokens=int(self.tokenAmount),
            top_p=0.5,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=int(self.answerAmount),
        )

        self.responseBox.setText("")

        answers = [
            ("[+] ANSWER " + str(i + 1), response.choices[i].text)
            for i in range(int(self.answerAmount))
        ]

        with open("Responses.txt", "a+") as f:
            f.write(
                f"Prompt: {self.promptEdit.toPlainText()} | Engine: {self.cEngine}\n"
            )
            for i, (label, text) in enumerate(answers, start=1):
                self.responseBox.append(f"{label}\n{text}\n\n\n")
                f.write(
                    f"\n[+]----------ANSWER-{i}---------\n{text}\n[+]--------------------------\n\n"
                )

        self.finished.setText("[+] Done! Also saved to Responses.txt")
        print("Done!")


messages = []


# Chat tab
class Tab2(QWidget):
    def __init__(self, parent=None):
        super(Tab2, self).__init__(parent)

        # adding widgets
        self.responseLabel = QLabel("Response:")
        self.responseBox = QTextEdit()
        self.responseBox.setReadOnly(True)
        self.promptEdit = QLineEdit()

    def init_ui(self):
        l = QGridLayout()
        l.addWidget(self.responseLabel, 0, 0)
        l.addWidget(self.responseBox, 1, 0)
        l.addWidget(self.promptEdit, 2, 0)
        self.setLayout(l)

        self.promptEdit.setPlaceholderText("Enter your prompt here...")
        self.promptEdit.returnPressed.connect(self.generate_response)

    def generate_response(self):

        global messages

        prompt = self.promptEdit.text()
        self.promptEdit.setText("")
        self.responseBox.append(f"\nUser: {prompt}\n\nBot: ")

        messages.append({"role": "user", "content": f"{prompt}"})

        self.thread = QThread()
        self.worker = Worker(
            responseBox=self.responseBox,
            chat_engine=chat_engine,
            tempAmount=tempAmount,
            tokenAmount=tokenAmount,
            messages=messages,
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        self.worker.progress.connect(self.update_text_box)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

    def update_text_box(self, text_chunk):
        self.responseBox.insertPlainText(text_chunk)


# Chat Settings Tab
class Tab3(QWidget):
    def __init__(self, parent=None):
        super(Tab3, self).__init__(parent)

        # setting default variables
        global chat_engine
        chat_engine = "gpt-4"

        global tempAmount
        tempAmount = 0.1

        global tokenAmount
        tokenAmount = 10

        # adding widgets
        self.tempAmount = float(0.1)
        self.tokenAmount = 10

        self.tempSlide = QSlider(Qt.Orientation.Horizontal)
        self.tokenSlide = QSlider(Qt.Orientation.Horizontal)

        self.tokenStatus = QLabel("Tokens: 10")
        self.tempStatus = QLabel("Temp: 0.0")
        self.engineStatus = QLabel("Engine: ")

        self.tokenSlide.setRange(10, 1000)
        self.tokenSlide.valueChanged.connect(self.selection_change)

        self.tempSlide.setRange(0, 10)
        self.tempSlide.valueChanged.connect(self.selection_change)

        self.exportButton = QPushButton("Export Chat")
        self.exportButton.clicked.connect(self.export_chat)
        self.engineBox = QComboBox()

        model_list = load_models()
        self.engineBox.addItems(model_list)

        self.engineBox.setCurrentText(chat_engine)
        self.engineBox.currentIndexChanged.connect(self.selection_change)

    def selection_change(self):
        global tokenAmount
        self.tokenAmount = self.tokenSlide.value()
        tokenAmount = self.tokenAmount
        self.tokenStatus.setText(f"Tokens: {self.tokenAmount}")

        global tempAmount
        self.tempAmount = self.tempSlide.value() / 10
        tempAmount = self.tempAmount
        self.tempStatus.setText(f"Temp: {self.tempAmount}")

        global chat_engine
        chat_engine = self.engineBox.currentText()

    def export_chat(self, messages):
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
        try:
            with open(filename, "w") as f:
                lines = (f"{msg['role']}: {msg['content']}" for msg in messages)
                f.write("\n".join(lines))
        except Exception as e:
            print(f"Error: {e}")

    def init_ui(self):
        layout = QGridLayout()
        layout.setRowStretch(10, 1)
        law = lambda w, r, c: layout.addWidget(w, r, c)
        law(self.engineBox, 0, 0)
        law(self.tempStatus, 1, 0)
        law(self.tempSlide, 2, 0)
        law(self.tokenStatus, 3, 0)
        law(self.tokenSlide, 4, 0)
        law(self.exportButton, 5, 0)

        self.setLayout(layout)
