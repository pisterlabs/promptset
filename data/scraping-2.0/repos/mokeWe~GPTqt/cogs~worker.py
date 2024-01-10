from PyQt6.QtCore import pyqtSignal, QObject
import openai


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)

    def __init__(self, responseBox, chat_engine, tempAmount, tokenAmount, messages):
        super().__init__()
        self.responseBox = responseBox
        self.chat_engine = chat_engine
        self.tempAmount = tempAmount
        self.tokenAmount = tokenAmount
        self.messages = messages

    def generate_response(self):
        global messages

    def generate_response(self):
        global messages

        response = openai.ChatCompletion.create(
            model=self.chat_engine,
            temperature=self.tempAmount,
            max_tokens=self.tokenAmount,
            messages=self.messages,
            stream=True,
        )

        collected_chunks = []
        collected_messages = []

        for chunk in response:
            collected_chunks.append(chunk)
            chunk_message = chunk["choices"][0]["delta"]
            collected_messages.append(chunk_message)
            self.progress.emit("".join(chunk_message.get("content", "")))

        full_reply = "".join([m.get("content", "") for m in collected_messages])

        messages.append({"role": "assistant", "content": f"{full_reply}"})

    def run(self):
        self.generate_response()
        self.finished.emit()
