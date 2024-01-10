from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
from PyQt5 import QtGui,QtWidgets,QtCore
from window.Ui_untitled import Ui_MainWindow
import sys
from core.get_response import get_res
import openai
from core.chat import use 

class MyWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.answer.setReadOnly(True)
        self.submit.clicked.connect(self.submit_handler)
        
        self.keyisok=False
        self.model=0
        self.models.addItem("text-davinci-003")
        self.models.addItem("text-curie-001")
        self.models.addItem("text-babbage-001")
        self.models.addItem("text-ada-001")
        self.models.addItem("gpt-3.5-turbo")
        self.models.currentIndexChanged[int].connect(self.get_value)
        self.exit.clicked.connect(QtCore.QCoreApplication.quit)
        self.setWindowTitle("ChatGPT小玩具")
        self.setWindowIcon(QtGui.QIcon("./resource/favicon.ico"))

    def get_value(self,i):
        
        self.model=i

    
            

    def submit_handler(self):
        if self.model==0:
           self.modelselect="text-davinci-003"
        elif self.model==1:
            self.modelselect="text-curie-001"
        elif self.model==2:
            self.modelselect="text-babbage-001"
        elif self.model==3:
            self.modelselect="text-ada-001"
        elif self.model==4:
            self.modelselect="gpt-3.5-turbo"
        key_txt=self.key.toPlainText()   
        if key_txt=='':
            msg_box = QMessageBox(QMessageBox.Critical, '错误！', '请输入API key!')
            msg_box.exec_() 
        else:

                 
            text=self.question.toPlainText()
            if text=='':
                msg_box = QMessageBox(QMessageBox.Critical, '错误！', '请输入文字')
                msg_box.exec_()
            else:
                try:
                    msg_box = QMessageBox(QMessageBox.Information, '提示！', '使用'+self.modelselect+'完成您的任务！')
                    msg_box.exec_()
                    if self.modelselect=="gpt-3.5-turbo":
                        res_txt=use(text,key_txt)
                    else:
                        res_txt=get_res(text,self.modelselect,key_txt)
                    self.answer.setPlainText(res_txt)
                except openai.error.RateLimitError as e:
                    msg_box = QMessageBox(QMessageBox.Critical, '错误！', '网络不好，请重试！如重试不行，请连接VPN')
                    msg_box.exec_()
                except openai.error.AuthenticationError as e2:
                    msg_box = QMessageBox(QMessageBox.Critical, '错误！', 'API key无效，请重新输入！')
                    msg_box.exec_()
                except UnicodeEncodeError as e3:
                    msg_box = QMessageBox(QMessageBox.Critical, '错误！', 'API key错误！')
                    msg_box.exec_()
                except openai.error.APIConnectionError as e1:
                    msg_box = QMessageBox(QMessageBox.Critical, '错误！', '请检查网络连接！')
                    print(e1)
                    
                    msg_box.exec_()
                except openai.error.APIError as e0:
                    msg_box = QMessageBox(QMessageBox.Critical, '错误！', '服务器人太多啦，请稍后重试！')
                    msg_box.exec_()
            
        
    


            
        


def run():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # application 对象
    app = QApplication(sys.argv)
    
    # QMainWindow对象
    mainwindow = MyWindow()



    # 显示
    mainwindow.show()
    app.exec_()
