
from PyQt5 import QtCore, QtGui, QtWidgets
import getpass
import os
import openai
import time
from datetime import datetime
import socket

# Our API_KEY

openai.api_key = "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"

# your API_KEY like this : sk-fsE2pREdZ1FZYZTwjkNIT3BlbkFJtsLlyikcO6hKXsDCrO


# Get the user name 
user = getpass.getuser()

_hour = datetime.now().hour
_minute = datetime.now().minute

if _hour<10 : 
    _hour = f"0{_hour}"

if _minute<10 : 
    _minute = f'0{_minute}'

def function():
    print(f'Bonjour {user}')


# Function to check a connection

def CheckConnection():
    try : 
        socket.gethostbyname('google.com')
        return True
    except : 
        return False
    
# print(CheckConnection())
    
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        
        # MainWindow Configuration
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 770)
        MainWindow.setMinimumSize(QtCore.QSize(500, 220))
        MainWindow.setMaximumSize(QtCore.QSize(900, 770))
        MainWindow.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:1 rgba(0, 116, 140, 255));\n"
"background-color: rgb(0, 116, 140);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        
        #  Font
        
        font = QtGui.QFont()
        font.setPointSize(10)
        
        
 
        # Question label
        
        self.question = QtWidgets.QLabel(self.centralwidget)
        self.question.setGeometry(QtCore.QRect(20, 90, 101, 41))
        self.question.setStyleSheet("QLabel#question{\n"
"  margin-bottom: 0;\n"
"  font-weight: normal;\n"
"  text-align: center;\n"
"  white-space: nowrap;\n"
"  vertical-align: middle;\n"
"  background-image: none;\n"
"  border: 1px solid transparent;\n"
"  padding: 6px 12px;\n"
"  font-size: 14px;\n"
"  border-radius: 4px;\n"
"  color: #fff;\n"
"  background-color: #d9534f;\n"
"  border-color: #d43f3a;\n"
"  color: #fff;\n"
"  background-color: #5bc0de;\n"
"  border-color: #46b8da;\n"
"}\n")
        
        self.question.setObjectName("question")
        
        # Answers label
        
        self.answer = QtWidgets.QLabel(self.centralwidget)
        self.answer.setGeometry(QtCore.QRect(390, 90, 91, 41))
        self.answer.setStyleSheet("QLabel#answer{\n"
"  margin-bottom: 0;\n"
"  font-weight: normal;\n"
"  text-align: center;\n"
"  white-space: nowrap;\n"
"  vertical-align: middle;\n"
"  background-image: none;\n"
"  border: 1px solid transparent;\n"
"  padding: 6px 12px;\n"
"  font-size: 14px;\n"
"  border-radius: 4px;\n"
"  color: #fff;\n"
"  background-color: #d9534f;\n"
"  border-color: #d43f3a;\n"
"  color: #fff;\n"
"  background-color: #5bc0de;\n"
"  border-color: #46b8da;\n"
"}")
        self.answer.setObjectName("answer")
        
        # input text
        
        self.input_text = QtWidgets.QTextEdit(self.centralwidget)
        self.input_text.setGeometry(QtCore.QRect(20, 150, 301, 171))
        self.input_text.setSizeIncrement(QtCore.QSize(0, 100))
        self.input_text.setStyleSheet("QTextEdit#input_text{\n"
"  height: 46px;\n"
"  padding: 10px 16px;\n"
"  font-size: 18px;\n"
"  line-height: 1.3333333;\n"
"  border-radius: 6px;\n"
"  color: #8a6d3b;\n"
"  background-color: #fcf8e3;\n"
"  border-color: #8a6d3b;\n"
"}\n"
"")
        self.input_text.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.input_text.setObjectName("input_text")
        
        # output text
        
        self.output_text = QtWidgets.QTextEdit(self.centralwidget)
        self.output_text.setGeometry(QtCore.QRect(390, 150, 451, 521))
        self.output_text.setSizeIncrement(QtCore.QSize(0, 5))
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("QTextEdit#output_text{\n"
"  height: 46px;\n"
"  padding: 10px 16px;\n"
"  font-size: 18px;\n"
"  line-height: 1.3333333;\n"
"  border-radius: 6px;\n"
"  color: #8a6d3b;\n"
"  background-color: #fcf8e3;\n"
"  border-color: #8a6d3b;\n"
"}")
        self.output_text.setObjectName("output_text")
        
        # Progression bar
        
        self.Progress = QtWidgets.QProgressBar(self.centralwidget)
        self.Progress.setGeometry(QtCore.QRect(640, 90, 211, 41))
        font_progression = QtGui.QFont()
        font_progression.setPointSize(10)
        font_progression.setBold(True)
        font_progression.setItalic(False)
        font_progression.setUnderline(False)
        font_progression.setWeight(75)
        font_progression.setStrikeOut(False)
        font_progression.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.Progress.setFont(font_progression)
        self.Progress.setCursor(QtGui.QCursor(QtCore.Qt.ForbiddenCursor))
        self.Progress.setMouseTracking(True)
        self.Progress.setFocusPolicy(QtCore.Qt.TabFocus)
        self.Progress.setStyleSheet("color: rgb(0, 0, 0);")
        self.Progress.setInputMethodHints(QtCore.Qt.ImhNone)
        self.Progress.setProperty("value", 0)
        self.Progress.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.Progress.setOrientation(QtCore.Qt.Horizontal)
        self.Progress.setInvertedAppearance(False)
        self.Progress.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.Progress.setObjectName("Progress")
        
        # Request Button
        
        self.request_button = QtWidgets.QPushButton(self.centralwidget)
        self.request_button.setGeometry(QtCore.QRect(230, 330, 91, 31))
        self.request_button.setStyleSheet("QPushButton#request_button{\n"
"  margin-bottom: 0;\n"
"  font-weight: normal;\n"
"  text-align: center;\n"
"  white-space: nowrap;\n"
"  vertical-align: middle;\n"
"  background-image: none;\n"
"  border: 1px solid transparent;\n"
"  padding: 6px 12px;\n"
"  font-size: 14px;\n"
"  border-radius: 4px;\n"
"  color: #fff;\n"
"  background-color: #d9534f;\n"
"  border-color: #d43f3a;\n"
" color: #fff;\n"
"  background-color: #5cb85c;\n"
"  border-color: #4cae4c;\n"
"}\n"
"\n"
"\n"
"QPushButton#request_button:hover {\n"
"  color: #fff;\n"
"  background-color: #449d44;\n"
"  border-color: #398439;\n"
"}")
        self.request_button.setObjectName("request_button")
        
        #  Clear Button output
        
        self.clearOutput = QtWidgets.QPushButton(self.centralwidget)
        self.clearOutput.setGeometry(QtCore.QRect(750, 680, 91, 31))
        self.clearOutput.setStyleSheet(" QPushButton#clearOutput{\n"
"  margin-bottom: 0;\n"
"  font-weight: normal;\n"
"  text-align: center;\n"
"  white-space: nowrap;\n"
"  vertical-align: middle;\n"
"  background-image: none;\n"
"  border: 1px solid transparent;\n"
"  padding: 6px 12px;\n"
"  font-size: 14px;\n"
"  border-radius: 4px;\n"
"  color: #fff;\n"
"  background-color: #d9534f;\n"
"  border-color: #d43f3a;\n"
"}\n"
"\n"
" QPushButton#clearOutput:hover {\n"
"  color: #fff;\n"
"  background-color: #c9302c;\n"
"  border-color: #ac2925;\n"
"}")
        self.clearOutput.setObjectName("clearOutput")
        
        #  Clear Button input
        
        self.ClearInput = QtWidgets.QPushButton(self.centralwidget)
        self.ClearInput.setGeometry(QtCore.QRect(120, 330, 81, 31))
        self.ClearInput.setStyleSheet("QPushButton#ClearInput{\n"
"  margin-bottom: 0;\n"
"  font-weight: normal;\n"
"  text-align: center;\n"
"  white-space: nowrap;\n"
"  vertical-align: middle;\n"
"  background-image: none;\n"
"  border: 1px solid transparent;\n"
"  padding: 6px 12px;\n"
"  font-size: 14px;\n"
"  border-radius: 4px;\n"
"  color: #fff;\n"
"  background-color: #d9534f;\n"
"  border-color: #d43f3a;\n"
"}\n"
"\n"
"\n"
"QPushButton#ClearInput:hover {\n"
"  color: #fff;\n"
"  background-color: #c9302c;\n"
"  border-color: #ac2925;\n"
"}")
        self.ClearInput.setObjectName("ClearInput")

        # username 
        
        # self.username = QtWidgets.QTextEdit(self.centralwidget)
        # self.username.setGeometry(QtCore.QRect(210, 10, 551, 51))
        # self.username.setObjectName("username")
        # self.username.insertPlainText(f' {user}')
        
        self.textEdit_2 = QtWidgets.QLabel(self.centralwidget)
        self.textEdit_2.setFont(font_progression)
        self.textEdit_2.setGeometry(QtCore.QRect(630, 10, 261, 41))
        self.textEdit_2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.textEdit_2.setStyleSheet("QLabel#username{\n"
"  margin-bottom: 0;\n"
"  font-weight: normal;\n"
"  text-align: center;\n"
"  white-space: nowrap;\n"
"  vertical-align: middle;\n"
"  background-image: none;\n"
"  padding: 6px 12px;\n"
"  font-size: 14px;\n"
"  border-radius: 4px;\n"
"  color: #fff;\n"
"  background-color: #d9534f;\n"
"  border-color: #d43f3a;\n"
"  color: #fff;\n"
"  background-color:rgb(0, 116, 140);\n"
"  border-color: #46b8da;\n"
"}")
        self.textEdit_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.setText(f'{user}')
        
        #  Display Time
        
        self.DispTime = QtWidgets.QTextEdit(self.centralwidget)
        self.DispTime.setGeometry(QtCore.QRect(20, 10, 121, 41))
        self.DispTime.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.BlankCursor))
        self.DispTime.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.DispTime.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.DispTime.setObjectName("DispTime")
        self.DispTime.append('{hour} : {min}')
        
        # Copy Right Label
        
        self.CopyRight = QtWidgets.QLabel(self.centralwidget)
        self.CopyRight.setGeometry(QtCore.QRect(360, 740, 191, 21))
        # Font for CopyRight 
        font = QtGui.QFont()
        font.setPointSize(10)
        self.CopyRight.setFont(font)
        self.CopyRight.setObjectName("CopyRight")
        
        # documentation
        
        self.doc = QtWidgets.QTextEdit(self.centralwidget)
        self.doc.setGeometry(QtCore.QRect(10, 390, 331, 321))
        self.doc.setStyleSheet("QTextEdit#doc{\n"
"  height: 46px;\n"
"  padding: 10px 16px;\n"
"  font-size: 18px;\n"
"  line-height: 1.3333333;\n"
"  border-radius: 6px;\n"
"  color: white;\n"
"  background-color: #fcf8e3;\n"
"  background-color: #00748c;\n"
"  border-color: rgba(0, 116, 140, 255);\n"
"  height: 46px;\n"
"  padding: 10px 16px;\n"
"  font-size: 18px;\n"
"  line-height: 1.3333333;\n"
"  border-radius: 6px;\n"
"}")
        self.doc.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.doc.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.doc.setObjectName("doc")
    
        # Line above and under documentation
        
        self.horiz_line = QtWidgets.QFrame(self.centralwidget)
        self.horiz_line.setGeometry(QtCore.QRect(30, 370, 301, 20))
        self.horiz_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.horiz_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.horiz_line.setObjectName("horiz_line")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(20, 720, 301, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        
        # Other design
        
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(20, 10, 121, 41))
        self.textEdit.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.BlankCursor))
        self.textEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit.setObjectName("textEdit")
#         self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
#         self.textEdit_2.setGeometry(QtCore.QRect(210, 10, 551, 51))
#         self.textEdit_2.setObjectName("textEdit_2")
        
#         self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
#         self.textEdit_2.setGeometry(QtCore.QRect(290, 10, 331, 55))
#         self.textEdit_2.setStyleSheet("QTextEdit#textEdit_2{\n"
# "  margin-bottom: 0;\n"
# "  font-weight: normal;\n"
# "  text-align: center;\n"
# "  white-space: nowrap;\n"
# "  vertical-align: middle;\n"
# "  background-image: none;\n"
# "  border: 1px solid transparent;\n"
# "  padding: 6px 12px;\n"
# "  font-size: 14px;\n"
# "  border-radius: 4px;\n"
# "  color: #fff;\n"
# "  background-color: #d9534f;\n"
# "  border-color: #FFFFFF;\n"
# "  color: #fff;\n"
# "  background-color: #5bc0de;\n"
# "  border-color: #46b8da;\n"
# "}")
#         self.textEdit_2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.textEdit_2.setReadOnly(True)
#         self.textEdit_2.setObjectName("textEdit_2")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(360, 740, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        
        # Menu bar widget
        
        MainWindow.setCentralWidget(self.centralwidget)
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 726, 21))
        # self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        # Signal and Slot
        
        def insert_prompt():
            # self.input_text.append(f'{user}, vous avez un problème de mise à niveau ou votre clé a été expiré')
            return self.input_text.toPlainText()
        
        
        # Our API_FUNCTION
        
        # Response for our request 
        
        def API_FUNCTION():
            
            if CheckConnection() : 
                
                try:
                    
                    # try until the openai key is avalable
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt= insert_prompt(),
                        temperature=1,
                        max_tokens=2000,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=1
                    )
        
                    return response["choices"][0]["text"]
               
                except openai.KeyError:
                    
                    return f"<b style='color:background-color: #D9534F'>{user},  un problème est survenue.Veillez réessayer."
            
            else : 
                
                return "<b style='color:#D9534F;'>Vérifier votre connection internet.</b>"
                
            
                
        
        def insert_answers():
            
            self.output_text.clear()
            for i in range(0, 100, 2):
                self.Progress.setProperty("value", i)
                time.sleep(0.05)
                
            self.output_text.append(API_FUNCTION())
            self.Progress.setProperty("value", 100)
            
            
            
        def clear_input():
            # This function is used to clear input if the request was finished
            self.output_text.clear()
            self.input_text.clear()
            self.Progress.setProperty("value", 0)
            
        
        # self.retranslateUi(MainWindow)
        # self.clear_button.clicked.connect(clear_input) 
        # self.clear_button.clicked.connect(self.input_text.clear) 
        # QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # MainWindow.setCentralWidget(self.centralwidget)
        
        # self.retranslateUi(MainWindow)
        # # MainWindow.iconSizeChanged['QSize'].connect(self.answer.clear) # type: ignore
        # QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.retranslateUi(MainWindow)
        self.ClearInput.clicked.connect(self.input_text.clear) # type: ignore
        self.ClearInput.clicked.connect(self.input_text.clear) # type: ignore
        self.request_button.clicked.connect(insert_answers)
        self.clearOutput.clicked.connect(self.output_text.clear) # type: ignore
        self.clearOutput.clicked.connect(self.output_text.clear) # type: ignore
        MainWindow.iconSizeChanged['QSize'].connect(self.answer.clear) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "baroukh404"))
        self.input_text.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:18px; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.input_text.setPlaceholderText(_translate("MainWindow", "Ecrivez votre question ici."))
        self.output_text.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:18px; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.output_text.setPlaceholderText(_translate("MainWindow", "Votre réponse s\'affichera ici."))
        self.question.setText(_translate("MainWindow", "QUESTION"))
        self.answer.setText(_translate("MainWindow", "REPONSE"))
        self.doc.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:18px; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Cette application est basé sur l\'Intélligence Artificielle NLP. Il peut faire de son mieux pour répondre à n\'importe quelles questions dans beaucoup de domaine.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">- Ecrivez dans la zone &quot;QUESTION&quot; votre question bien formuler et bien compréhensible</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">- Vous pouvez écrire en anglais ou en français pour obtenir un meilleur resultat</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">- Si vous avez des questions ou amélioration à cette application, n\'hésiter pas à contacter l\'auteur via baroukh404@gmail.com.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Voici un example : </span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Question : &quot;Dis-moi que ceci est un test&quot;.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Réponse : &quot;Ceci est un test&quot;.</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p></body></html>"))
        self.request_button.setText(_translate("MainWindow", "SEND"))
        self.ClearInput.setText(_translate("MainWindow", "CLEAR"))
        self.clearOutput.setText(_translate("MainWindow", "CLEAR"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:20pt; font-weight:600;\">{_hour} : {_minute}</span></p></body></html>"))
#         self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
# "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
# "p, li { white-space: pre-wrap; }\n"
# "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
# f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:20pt;\">Bienvenue {user}</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "CopyRight.  Tous droits réserver"))

# Main console

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
