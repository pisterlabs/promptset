import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from mainApp.AI.openAI.example import openAI
from mainApp.widgets.CommonMenu.mainMenu import MainMenuBar
from mainApp.widgets.chatWidget.chatBoxColorText import ChatBox


class mainWindow(QMainWindow):
    chatBox: ChatBox
    mainMenu: MainMenuBar
    rectangle = QRect(0, 0, 800, 800)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initMenu()
        self.chatGpt = openAI()
        self.initUI()
        self.initStyle()
        self.initGeometry()
        self.initConnections()


    def initUI(self):
        mainWidget = QWidget(self)
        self.chatBox = ChatBox()
        mainWidget.setLayout(self.initLayout())

        self.setCentralWidget(mainWidget)

    def initLayout(self):
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.chatBox)
        return mainLayout

    def initStyle(self):
        pass

    def initGeometry(self):
        self.setWindowTitle("chat Gpt v0.1")
        centerScreen = QDesktopWidget().availableGeometry().center() - self.rectangle.center()
        # open widget in the center of the screen
        self.setGeometry(QRect(centerScreen, self.rectangle.size()))
        # setta il size della finestra
        self.setFixedSize(self.rectangle.size())

    def initConnections(self):
        self.chatBox.questionComing.connect(self.getAnswer)
        self.chatGpt.answerReceived.connect(self.setAnswer)

    def contextMenuEvent(self, event) -> None:
        contextMenu = QMenu(self)

    def initMenu(self):
        self.mainMenu = MainMenuBar(self)


    def initStartingValues(self):
        pass

    def getAnswer(self, question):
        return self.chatGpt.getAnswer(question)

    def setAnswer(self, answer):
        self.chatBox.addAnswerFromChatBot(answer)