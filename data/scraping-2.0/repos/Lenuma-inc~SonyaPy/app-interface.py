import voice
import sounddevice as sd
import vosk
import json
import queue
import openai
import sys
import threading
from interface import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

voice.bot_speak("Соня вас внимательно слушает ...")

messages = [
    {"role": "system", "content": "Ты голосовой ассистент по имени Соня."}
]
q = queue.Queue()

model = vosk.Model("model_small_ru")

device = sd.default.device  # <--- по умолчанию
samplerate = int(
    sd.query_devices(device[0], "input")["default_samplerate"]
)  # получаем частоту микрофона


def callback(indata, frames, time, status):
    q.put(bytes(indata))


def main():
    # постоянная прослушка микрофона
    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=16000,
        device=device[0],
        dtype="int16",
        channels=1,
        callback=callback,
    ):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                data = json.loads(rec.Result())["text"]
                recognize(data)
                clear_text(data)


def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages


# преобразование текста в речь
def recognize(data):
    print("Пользователь сказал: " + data)
    if data.startswith("соня"):
        # обращаются к ассистенту
        text = clear_text(data)
        print("Сервер получил: " + text)
        user_item = QtWidgets.QListWidgetItem()
        user_item.setTextAlignment(QtCore.Qt.AlignRight)
        user_item.setText('Вы сказали:' + '\n' + data)
        ui.chat_history.addItem(user_item)
        update_chat(messages, "user", text)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response = completion.choices[0].message.content
        if response != None:
            answer = response.lower()
            print("Сервер ответил: " + answer)
            bot_item = QtWidgets.QListWidgetItem()
            bot_item.setTextAlignment(QtCore.Qt.AlignLeft)
            bot_item.setText('Соня:' + '\n' + answer)
            ui.chat_history.addItem(bot_item)
            voice.bot_speak(answer)
        else:
            voice.bot_speak("Сервер ничего не ответил")

# очистка текста от имени бота
def clear_text(data):
    text = data
    text = text.replace('соня', '').strip()
    text = text.replace('  ', ' ').strip()
    return text


def start_thread():
    my_thread = threading.Thread(target=main, args=())
    my_thread.start()

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
ui.start.clicked.connect(start_thread)
sys.exit(app.exec_())
