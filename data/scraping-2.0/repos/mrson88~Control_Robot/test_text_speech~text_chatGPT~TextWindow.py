import sys
import speech_recognition as sr
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QWidget
from Text_UI import Ui_Form
import pyttsx3
import openai
openai.api_key='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'
#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
class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.voice_id = 'vietnam'
        self.engine.setProperty('voice', self.voice_id)
        self.speech_rate = 150
        self.engine.setProperty('rate', self.speech_rate)
        self.ui.horizontalSlider.setSliderPosition(80)
        #engineNames = QTextToSpeech.availableEngines()

        self.ui.pushButton_say.clicked.connect(self.say)
        self.ui.pushButton_listen.clicked.connect(self.listen)
        # -----
        self.items = []
        self.item_listen=[]
        self.model = QStandardItemModel(self)
        self.recognizer = sr.Recognizer()

        # -----------
        if self.engine :
            self.engine.say('xin chào sơn. Tôi có thể giúp gì cho bạn')
            self.engine.runAndWait()

        else:
            self.ui.pushButton_say.setEnabled(False)

    def say(self):
        self.ui.pushButton_say.setEnabled(False)
        self.ui.pushButton_listen.setEnabled(False)
        chat_reserve_say=self.chat_to_chatGPT(self.ui.lineEdit.text())
        self.engine.setProperty('volume', float(self.ui.horizontalSlider.value() / 100))
        self.engine.say(chat_reserve_say)
        self.items.append(self.ui.lineEdit.text())
        self.items.append(chat_reserve_say)
        print(self.items)
        self.ui.lineEdit.clear()
        self.model.removeRows(0, self.model.rowCount())
        for item in self.items:
            standardItem = QStandardItem(item)
            self.model.appendRow(standardItem)
        self.ui.listView.setModel(self.model)
        self.ui.listView.scrollToBottom()
        self.engine.runAndWait()
        self.ui.pushButton_say.setEnabled(True)
    
    def listen(self):
        self.ui.pushButton_listen.setEnabled(False)
        text_listen=self.convert_speech_to_text()
        print('text=',text_listen)
        chat_reserve_listen=self.chat_to_chatGPT(str(text_listen))
        print(chat_reserve_listen)
        self.engine.setProperty('volume', float(self.ui.horizontalSlider.value() / 100))
        self.engine.say(chat_reserve_listen)
        self.item_listen.append(text_listen)
        self.item_listen.append(chat_reserve_listen)
        self.model.removeRows(0, self.model.rowCount())
        for item in self.item_listen:
            standardItem = QStandardItem(item)
            self.model.appendRow(standardItem)
        self.ui.listView.setModel(self.model)
        self.ui.listView.scrollToBottom()
        self.engine.runAndWait()
        self.ui.pushButton_listen.setEnabled(True)

    def stateChanged(self, state):
        if (state == self.engine.runAndWait()):
            self.ui.pushButton_say.setEnabled(True)
            self.ui.pushButton_listen.setEnabled(True)

    def convert_speech_to_text(self):
        # Use the default microphone as the audio source
        with sr.Microphone() as source:
            text=''
            print("Listening...")
            # Adjust the energy threshold to account for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)
            # Listen for speech and convert it to text
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio, language='vi-VN')


                print("Converted text:", text)
            except sr.UnknownValueError:
                print("Unable to recognize speech")
            except sr.RequestError as e:
                print("Request error:", e)
            return text

    def chat_to_chatGPT(self,chat):
        completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[
            {"role": "user", "content": chat}
        ])
        return completion.choices[0].message.content
app = QApplication(sys.argv)
window = Window()
window.show()

sys.exit(app.exec())
