# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from openaiTest import OpenaiRequest
from module import Mydocuments, print_docx, print_hwp, print_pdf

class Ui_Dialog(object):
    def __init__(self):
        self.doc = Mydocuments()
        self.doc.get_documents()
        self.rowCnt = 0
        self.token = None
    
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(620, 953)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 510, 471, 31))
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(40, 570, 541, 71))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 540, 401, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(270, 150, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(40, 710, 541, 221))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 660, 111, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(40, 420, 438, 30))
        self.label_5.setObjectName("label_3")
        self.textEdit_3 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_3.setGeometry(QtCore.QRect(40, 450, 438, 30))
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(490, 450, 90, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_2.clicked.connect(self.on_create_document_clicked)

        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(40, 200, 541, 192))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setColumnWidth(0, 130)
        self.tableWidget.setColumnWidth(1, 360)
        self.tableWidget.setRowCount(500)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.textEdit_2 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_2.setGeometry(QtCore.QRect(40, 90, 541, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(40, 60, 401, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(40, 20, 471, 31))
        self.label_4.setObjectName("label_4")
        self.pushButton.clicked.connect(self.on_search_document_clicked)
        self.pushButton_3.clicked.connect(self.on_save_token_clicked)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def clear_tableWidget(self):
        _translate = QtCore.QCoreApplication.translate
        
        for i in range(501):
            item = QtWidgets.QTableWidgetItem()
            item.setText(_translate("Dialog", ""))
            self.tableWidget.setItem(i, 0, item)
            
            item = QtWidgets.QTableWidgetItem()
            item.setText(_translate("Dialog", ""))
            self.tableWidget.setItem(i, 1, item)
        
        self.rowCnt = 0
        
    def on_save_token_clicked(self):
        token = self.textEdit_3.toPlainText()
        self.token = token
    
    def on_search_document_clicked(self):
        self.clear_tableWidget()
        content = self.textEdit_2.toPlainText()
        
        if content == "":
            print("내용을 입력하세요.")
        else:
            self.doc.search_documents([content])
                        
            i = 0
            _translate = QtCore.QCoreApplication.translate
            
            for path in self.doc.get_search_cache():
                item = QtWidgets.QTableWidgetItem()
                item.setText(_translate("Dialog", path))
                self.tableWidget.setItem(i, 1, item)
                
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                item.setText(path.split('\\')[-1])
                self.tableWidget.setItem(i, 0, item)
                
                i += 1
            
            self.rowCnt = i - 1
            
    def on_create_document_clicked(self):
        content = self.textEdit.toPlainText()
        
        if content == "":
            print("내용을 입력하세요.")
        else:
            fulltext = ''
            for i in range(self.rowCnt + 1):
                item = self.tableWidget.item(i, 0)
                if item is not None and item.checkState() == QtCore.Qt.Checked:
                    text = item.text()
                    path = self.tableWidget.item(i, 1).text()
                    
                    if text.endswith(".pdf"):
                        fulltext += print_pdf(path) + '\n'
                    elif text.endswith(".docx"):
                        fulltext += print_docx(path) + '\n'
                    elif text.endswith(".hwp"):
                        fulltext += print_hwp(path) + '\n'
                        
            result = OpenaiRequest.openaiRequst(self, self.token, fulltext + '\n' + content)
            self.textBrowser.setPlainText(result) 

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "아래 작성하려는 문서의 주제 혹은 간략한 요약을 입력하세요."))
        self.label_2.setText(_translate("Dialog", "예시) 자동차 판매 보고서 만들어 줘"))
        self.pushButton.setText(_translate("Dialog", "File Search"))
        self.pushButton_2.setText(_translate("Dialog", "Create Document"))
        self.pushButton_3.setText(_translate("Dialog", "Save"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Dialog", "1"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "파일 이름"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "파일 경로"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setSortingEnabled(__sortingEnabled)
        self.label_3.setText(_translate("Dialog", "예시) 자동차"))
        self.label_4.setText(_translate("Dialog", "아래 작성하려는 문서의 주제를 단어로 입력하세요."))
        self.label_5.setText(_translate("Dialog", "Open Ai API Token"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
