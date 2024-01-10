# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import resource_rc
import pandas as pd
import time
import openai
import pyperclip
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import requests

openai.api_key = "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"

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1591, 903)
        MainWindow.setStyleSheet("#MainWindow{\n"
"background-color: rgb(52, 53, 65)\n"
"}\n"
"\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(52, 53, 65)\n"
"")
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 350, 61))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(17)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("color: #fff")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 241, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(10)
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: #fff")
        self.label_2.setObjectName("label_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(170, 100, 281, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("#comboBox{\n"
"border: 1px solid #ced4da;\n"
"border-radius: 4px;\n"
"padding-left: 10px;\n"
"color: #fff\n"
"} \n"
"\n"
"#comboBox::drop{\n"
"border: 0px\n"
"\n"
"\n"
"} ")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 100, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: #fff")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 150, 631, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: #fff")
        self.label_4.setObjectName("label_4")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(20, 240, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        font.setItalic(False)
        self.checkBox.setFont(font)
        self.checkBox.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox.setObjectName("checkBox")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 200, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: #fff")
        self.label_5.setObjectName("label_5")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(20, 280, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_2.setObjectName("checkBox_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 320, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("color: #fff")
        self.label_6.setObjectName("label_6")
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(20, 360, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_3.setFont(font)
        self.checkBox_3.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(20, 400, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_4.setFont(font)
        self.checkBox_4.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(20, 440, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_5.setFont(font)
        self.checkBox_5.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_5.setObjectName("checkBox_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 480, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color: #fff")
        self.label_7.setObjectName("label_7")
        self.checkBox_6 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_6.setGeometry(QtCore.QRect(20, 510, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_6.setFont(font)
        self.checkBox_6.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_7 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_7.setGeometry(QtCore.QRect(20, 550, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_7.setFont(font)
        self.checkBox_7.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_7.setObjectName("checkBox_7")
        self.checkBox_8 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_8.setGeometry(QtCore.QRect(20, 590, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_8.setFont(font)
        self.checkBox_8.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_8.setObjectName("checkBox_8")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(20, 630, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: #fff")
        self.label_8.setObjectName("label_8")
        self.checkBox_9 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_9.setGeometry(QtCore.QRect(20, 670, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_9.setFont(font)
        self.checkBox_9.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_9.setObjectName("checkBox_9")
        self.checkBox_10 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_10.setGeometry(QtCore.QRect(20, 750, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_10.setFont(font)
        self.checkBox_10.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_10.setObjectName("checkBox_10")
        self.checkBox_11 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_11.setGeometry(QtCore.QRect(20, 790, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_11.setFont(font)
        self.checkBox_11.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_11.setObjectName("checkBox_11")
        self.checkBox_12 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_12.setGeometry(QtCore.QRect(20, 710, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_12.setFont(font)
        self.checkBox_12.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_12.setObjectName("checkBox_12")
        self.checkBox_13 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_13.setGeometry(QtCore.QRect(20, 830, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_13.setFont(font)
        self.checkBox_13.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_13.setObjectName("checkBox_13")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(270, 200, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("color: #fff")
        self.label_9.setObjectName("label_9")
        self.checkBox_14 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_14.setGeometry(QtCore.QRect(270, 230, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_14.setFont(font)
        self.checkBox_14.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_14.setObjectName("checkBox_14")
        self.checkBox_15 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_15.setGeometry(QtCore.QRect(270, 270, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_15.setFont(font)
        self.checkBox_15.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_15.setObjectName("checkBox_15")
        self.checkBox_16 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_16.setGeometry(QtCore.QRect(270, 310, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_16.setFont(font)
        self.checkBox_16.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_16.setObjectName("checkBox_16")
        self.checkBox_17 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_17.setGeometry(QtCore.QRect(270, 390, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_17.setFont(font)
        self.checkBox_17.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_17.setObjectName("checkBox_17")
        self.checkBox_18 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_18.setGeometry(QtCore.QRect(270, 350, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_18.setFont(font)
        self.checkBox_18.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_18.setObjectName("checkBox_18")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(270, 420, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("color: #fff")
        self.label_10.setObjectName("label_10")
        self.checkBox_19 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_19.setGeometry(QtCore.QRect(270, 470, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_19.setFont(font)
        self.checkBox_19.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_19.setObjectName("checkBox_19")
        self.checkBox_20 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_20.setGeometry(QtCore.QRect(270, 550, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_20.setFont(font)
        self.checkBox_20.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_20.setObjectName("checkBox_20")
        self.checkBox_21 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_21.setGeometry(QtCore.QRect(270, 510, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_21.setFont(font)
        self.checkBox_21.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_21.setObjectName("checkBox_21")
        self.checkBox_22 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_22.setGeometry(QtCore.QRect(270, 590, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_22.setFont(font)
        self.checkBox_22.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_22.setObjectName("checkBox_22")
        self.checkBox_23 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_23.setGeometry(QtCore.QRect(270, 630, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(-1)
        self.checkBox_23.setFont(font)
        self.checkBox_23.setStyleSheet("color: #fff;\n"
"font-size: 18px;\n"
"border-radius: 4px;\n"
"\n"
"QCheckBox::indicator{\n"
"width: 25px;\n"
"heigh: 25px;\n"
"};\n"
"\n"
"QCheckBox::hover{\n"
"    color: #B31031;\n"
"};\n"
"\n"
"QCheckBox::indicator:unchecked{\n"
"image: url(:/checkbox/unchecked.png);\n"
"} ;\n"
"\n"
"\n"
"QCheckBox::indicator:checked{\n"
"image: url(:/checkbox/checkmark.png);\n"
"} \n"
"")
        self.checkBox_23.setObjectName("checkBox_23")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(250, 700, 321, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 760, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(720, 90, 631, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("color: #fff")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(720, 190, 631, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("color: #fff")
        self.label_12.setObjectName("label_12")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(720, 130, 381, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(720, 235, 381, 70))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit_2.setPlainText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(720, 320, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("color: #fff")
        self.label_13.setObjectName("label_13")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(720, 360, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(1050, 320, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("color: #fff")
        self.label_14.setObjectName("label_14")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(1050, 360, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit_4.setText("")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(720, 410, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("color: #fff")
        self.label_15.setObjectName("label_15")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(720, 460, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit_5.setText("")
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(1050, 410, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(12)
        font.setItalic(False)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("color: #fff")
        self.label_16.setObjectName("label_16")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(1050, 460, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit_6.setText("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(720, 520, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(930, 520, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit_7 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(720, 580, 641, 251))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(13)
        self.lineEdit_7.setFont(font)
        self.lineEdit_7.setStyleSheet("background-color: rgb(50, 54, 60);\n"
"border: 2px solid white;\n"
"border-radius: 5px;\n"
"padding-left: 10px;\n"
"padding-right: 10px;\n"
"color: #fff")
        self.lineEdit_7.setPlainText("")
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(720, 850, 331, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(1130, 520, 281, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        

        self.pushButton_11 = QPushButton(self.centralwidget)
        self.pushButton_11.setObjectName(u"pushButton_11")
        self.pushButton_11.setGeometry(QtCore.QRect(720, 50, 51, 31))
        self.pushButton_11.setFont(font)
        self.pushButton_11.setStyleSheet(u"background-color:#fff;\n"
"border-radius: 10px")

        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(1080, 850, 331, 41))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(11)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setStyleSheet("background-color:#fff;\n"
"border-radius: 10px")
        MainWindow.setCentralWidget(self.centralwidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7.setObjectName("pushButton_7")


        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(1530, 0, 55, 61))
        self.label_17.setObjectName("label_17")


        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(1130, 130, 241, 171))
        self.label_18.setObjectName("label_18")

        self.filterListForDiet = []
        self.filterListForFlavorPalate = []
        self.filterListForCourse = []
        self.filterListForPrepTime = []
        self.filterListCookTime = []
        self.filterListRegion = []
        self.selected_rows = pd.DataFrame()

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.messageBox("FoodExpress v0.1", "Credits: \nMade By: Aditya Balsane\nDataset: Indian Food 101 by Neha Prabhavalkar\nPowered By GPT-3.5 Turbo")

    def messageBox(self, title, message):
        messageBoxWidget = QMessageBox()
        messageBoxWidget.setWindowTitle(title)
        messageBoxWidget.setIcon(QMessageBox.Information)
        messageBoxWidget.setText(message)
        messageBoxWidget.setStandardButtons(QMessageBox.Ok)
        messageBoxWidget.exec_()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FoodExpress"))
        self.label.setText(_translate("MainWindow", "Welcome to FoodExpress v0.1"))
        self.label_2.setText(_translate("MainWindow", "Culinary Journeys Awaits..."))
        self.comboBox.setItemText(0, _translate("MainWindow", "indian_foods.csv"))
        self.comboBox.setItemText(1, _translate("MainWindow", "coming soon..."))
        self.label_3.setText(_translate("MainWindow", "Choose dataset: "))
        self.label_4.setText(_translate("MainWindow", "To receive meal recommendations, please select from the following options:"))
        self.checkBox.setText(_translate("MainWindow", "Vegetarian"))
        self.label_5.setText(_translate("MainWindow", "Diet"))
        self.checkBox_2.setText(_translate("MainWindow", "Non-Vegetarian"))
        self.label_6.setText(_translate("MainWindow", " Flavor Palate"))
        self.checkBox_3.setText(_translate("MainWindow", "Sweet"))
        self.checkBox_4.setText(_translate("MainWindow", "Spicy"))
        self.checkBox_5.setText(_translate("MainWindow", "Bitter"))
        self.label_7.setText(_translate("MainWindow", "Course    "))
        self.checkBox_6.setText(_translate("MainWindow", "Dessert"))
        self.checkBox_7.setText(_translate("MainWindow", "Snack"))
        self.checkBox_8.setText(_translate("MainWindow", "Main Course"))
        self.label_8.setText(_translate("MainWindow", "Prep Time"))
        self.checkBox_9.setText(_translate("MainWindow", "Less than 5 minutes  "))
        self.checkBox_10.setText(_translate("MainWindow", "15 minutes  "))
        self.checkBox_11.setText(_translate("MainWindow", "30 minutes  "))
        self.checkBox_12.setText(_translate("MainWindow", "5-10 minutes  "))
        self.checkBox_13.setText(_translate("MainWindow", "More than 30 minutes  "))
        self.label_9.setText(_translate("MainWindow", "Cook time"))
        self.checkBox_14.setText(_translate("MainWindow", "5-10 minutes  "))
        self.checkBox_15.setText(_translate("MainWindow", "15 minutes  "))
        self.checkBox_16.setText(_translate("MainWindow", "30 minutes  "))
        self.checkBox_17.setText(_translate("MainWindow", "More than 60 minutes  "))
        self.checkBox_18.setText(_translate("MainWindow", "60 minutes  "))
        self.label_10.setText(_translate("MainWindow", "Region"))
        self.checkBox_19.setText(_translate("MainWindow", "North"))
        self.checkBox_20.setText(_translate("MainWindow", "South"))
        self.checkBox_21.setText(_translate("MainWindow", "North-East"))
        self.checkBox_22.setText(_translate("MainWindow", "West"))
        self.checkBox_23.setText(_translate("MainWindow", "East"))
        self.pushButton.setText(_translate("MainWindow", "Generate Recommendations"))
        self.pushButton_2.setText(_translate("MainWindow", "Clear Selection"))
        self.label_11.setText(_translate("MainWindow", "Recipe Name"))
        self.label_12.setText(_translate("MainWindow", "Recipe Ingredients"))
        self.label_13.setText(_translate("MainWindow", "State of Origin"))
        self.label_14.setText(_translate("MainWindow", "Flavor Profile"))
        self.label_15.setText(_translate("MainWindow", "Prep time"))
        self.label_16.setText(_translate("MainWindow", "Cook time"))
        self.pushButton_3.setText(_translate("MainWindow", "Clear Results"))
        self.pushButton_4.setText(_translate("MainWindow", "Generate Recipe"))
        self.pushButton_5.setText(_translate("MainWindow", "Copy Recipe"))
        self.pushButton_6.setText(_translate("MainWindow", "Generate with Ingredients"))
        self.pushButton_7.setText(_translate("MainWindow", "Save Recipe"))
        self.pushButton_11.setText(_translate("MainWindow", ">>"))

        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/icons/gpt.png\" width=\"50\" height=\"50\"/></p></body></html>"))

        self.pushButton_2.clicked.connect(self.clearFilter)
        self.pushButton.clicked.connect(self.generateFilterList)
        self.pushButton_3.clicked.connect(self.clearGeneratedResults)
        self.pushButton_4.clicked.connect(self.generateRecipe)
        self.pushButton_5.clicked.connect(self.copyResultToClipBoard)
        self.pushButton_6.clicked.connect(self.generateWithIngredients)
        self.pushButton_7.clicked.connect(self.saveRecipe)
        self.pushButton_11.clicked.connect(self.chooseAnotherSample)


    def saveRecipe(self):
        timestamp = int(time.time())
        filename = self.lineEdit.text()
        file_name = f"{filename}_{timestamp}.txt"
        with open(file_name, "w") as f:
            f.write(self.lineEdit_7.toPlainText().strip())
        self.messageBox("FoodExpress", "File Created! \nFile Name: " + file_name)

    def copyResultToClipBoard(self): 
        pyperclip.copy(self.lineEdit_7.toPlainText().strip())
        self.messageBox("FoodExpress", "Copied to clipboard")

    def generateWithIngredients(self):

         if self.lineEdit.text() == "" or self.lineEdit_2.toPlainText().strip()  == "":
            self.messageBox("FoodExpress", "Sorry!! No Recipe found")
         else:
            self.lineEdit_7.setPlainText("Please Wait...")

            print('Awaiting Response....')
            prompt = "Please make a step by step Recipe guide with this Ingredients" + self.lineEdit_2.toPlainText().strip()
            response = self.makePromptRequest(prompt)
            print("Response Received...")
            self.lineEdit_7.setPlainText(response)


    def generateRecipe(self):
        

        if self.lineEdit.text() == "" or self.lineEdit_2.toPlainText().strip()  == "":
            self.messageBox("FoodExpress", "Sorry!! No Recipe found")
        else:
            self.lineEdit_7.setPlainText("Please Wait...")

            print('Awaiting Response....')
            prompt = "Please make a step by step Recipe guide for indian meal named " + self.lineEdit.text()
            response = self.makePromptRequest(prompt)
            print("Response Received...")
            self.lineEdit_7.setPlainText(response)



    def makePromptRequest(self, prompt):
         msg = [{"role": "user", "content": prompt}]
         request = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msg
        )

         reply = request.choices[0].message.content
         return reply


  
    def clearGeneratedResults(self):
          self.lineEdit.setText("")
          self.lineEdit_2.setPlainText("")
          self.lineEdit_3.setText("")
          self.lineEdit_4.setText("")
          self.lineEdit_5.setText("")
          self.lineEdit_6.setText("")
          self.clearFilter()


    def chooseAnotherSample(self):
        if self.selected_rows.empty:
            self.messageBox("FoodExpress", "Sorry!! No records found")
        else:
          #      print(selected_rows)
          random_row =  self.selected_rows.sample(n=1)
          print(random_row)
          self.lineEdit.setText(random_row['name'].iloc[0])
          self.lineEdit_2.setPlainText(random_row['ingredients'].iloc[0])
          self.lineEdit_3.setText(random_row['state'].iloc[0]) 
          self.lineEdit_4.setText(random_row['flavor_profile'].iloc[0])
          self.lineEdit_5.setText(str(random_row['prep_time'].iloc[0]))
        #  self.lineEdit_6.setText(str(random_row['cook_time'].iloc[0]))
          self.lineEdit_6.setText(str(self.filterListCookTime[0]))


    def getRecommendations(self):
         
        df = pd.read_csv('indian_food.csv')
        strTest = ['M30', 'M60']
        if any(ext in self.filterListCookTime for ext in strTest):
            self.selected_rows = df[(df['diet'] == self.filterListForDiet[0]) & (df['flavor_profile'] == self.filterListForFlavorPalate[0]) & (df['course'] == self.filterListForCourse[0]) 
                           & (df['prep_time'] == self.filterListForPrepTime[0]) & (df['cook_time'] <= 60) & (df['region'] == self.filterListRegion[0])]

        elif any(ext in self.filterListForPrepTime for ext in strTest):
             self.selected_rows = df[(df['diet'] == self.filterListForDiet[0]) & (df['flavor_profile'] == self.filterListForFlavorPalate[0]) & (df['course'] == self.filterListForCourse[0]) 
                           & (df['prep_time'] <= 30) & (df['cook_time'] == self.filterListCookTime[0]) & (df['region'] == self.filterListRegion[0])]
        elif any(ext in self.filterListCookTime for ext in strTest) and any(ext in self.filterListForPrepTime for ext in strTest):
             self.selected_rows = df[(df['diet'] == self.filterListForDiet[0]) & (df['flavor_profile'] == self.filterListForFlavorPalate[0]) & (df['course'] == self.filterListForCourse[0]) 
                           & (df['prep_time'] <= self.filterListForPrepTime[0]) & (df['cook_time'] <= self.filterListCookTime[0]) & (df['region'] == self.filterListRegion[0])]
        else:
             self.selected_rows = df[(df['diet'] == self.filterListForDiet[0]) & (df['flavor_profile'] == self.filterListForFlavorPalate[0]) & (df['course'] == self.filterListForCourse[0]) 
                           & (df['prep_time'] >= self.filterListForPrepTime[0]) & (df['cook_time'] >= self.filterListCookTime[0]) & (df['region'] == self.filterListRegion[0])]
     
        if  self.selected_rows.empty:
            self.messageBox("FoodExpress", "Sorry!! No records found")
        else:
          #      print(selected_rows)
          random_row =  self.selected_rows.sample(n=1)
          print(random_row)
          self.lineEdit.setText(random_row['name'].iloc[0])
          self.lineEdit_2.setPlainText(random_row['ingredients'].iloc[0])
          self.lineEdit_3.setText(random_row['state'].iloc[0]) 
          self.lineEdit_4.setText(random_row['flavor_profile'].iloc[0])
          self.lineEdit_5.setText(str(random_row['prep_time'].iloc[0]))
        #  self.lineEdit_6.setText(str(random_row['cook_time'].iloc[0]))
          self.lineEdit_6.setText(str(self.filterListCookTime[0]))
          self.generateImage()


    def generateImage(self):
         self.thread = GenerateImageThread('Generate an realastic flicker image for indian dish named '+ self.lineEdit.text())
         self.thread.image_generated.connect(self.displayImage)
         self.thread.start()


    def displayImage(self, image):
        pixmap = QPixmap(image)
        self.label_18.setPixmap(pixmap)

    def generatefilterListForCourse(self):
        self.filterListForCourse.clear()
        if(self.checkBox_6.isChecked()):
            self.filterListForCourse.append('dessert')
        elif(self.checkBox_7.isChecked()):
            self.filterListForCourse.append('snack')
        elif(self.checkBox_8.isChecked()):
            self.filterListForCourse.append('main course')
        print(self.filterListForCourse)


    def generatefilterListForPrepTime(self):
        self.filterListForPrepTime.clear()
        if(self.checkBox_9.isChecked()):
            self.filterListForPrepTime.append(5)
        elif(self.checkBox_12.isChecked()):
            self.filterListForPrepTime.append(10)
        elif(self.checkBox_10.isChecked()):
            self.filterListForPrepTime.append(15)
        elif(self.checkBox_11.isChecked()):
            self.filterListForPrepTime.append(30)
        elif(self.checkBox_13.isChecked()):
            self.filterListForPrepTime.append('M30')
        print(self.filterListForPrepTime)

    def generatefilterListCookTime(self):
        self.filterListCookTime.clear() 
        if(self.checkBox_14.isChecked()):
            self.filterListCookTime.append(10)
        elif(self.checkBox_15.isChecked()):
            self.filterListCookTime.append(15)
        elif(self.checkBox_16.isChecked()):
            self.filterListCookTime.append(30)
        elif(self.checkBox_18.isChecked()):
            self.filterListCookTime.append(60)
        elif(self.checkBox_17.isChecked()):
            self.filterListCookTime.append('M60')
        print(self.filterListCookTime)

    def generatefilterListForFlavorPalate(self):
        self.filterListForFlavorPalate.clear()
        if(self.checkBox_3.isChecked()):
            self.filterListForFlavorPalate.append('sweet')
        elif(self.checkBox_4.isChecked()):
            self.filterListForFlavorPalate.append('spicy')
        elif(self.checkBox_5.isChecked()):
            self.filterListForFlavorPalate.append('bitter')
        print(self.filterListForFlavorPalate)

    def generatefilterListForDiet(self):
        self.filterListForDiet.clear()
        if(self.checkBox.isChecked()):
            self.filterListForDiet.append('vegetarian')
        elif(self.checkBox_2.isChecked()):
            self.filterListForDiet.append('non vegetarian')
        print(self.filterListForDiet)

    def generatefilterListRegion(self):
        self.filterListRegion.clear()
        if(self.checkBox_19.isChecked()):
            self.filterListRegion.append('North')
        elif(self.checkBox_21.isChecked()):
            self.filterListRegion.append('North East')
        elif(self.checkBox_21.isChecked()):
            self.filterListRegion.append('North East')
        elif(self.checkBox_20.isChecked()):
            self.filterListRegion.append('South')
        elif(self.checkBox_22.isChecked()):
            self.filterListRegion.append('West')
        elif(self.checkBox_23.isChecked()):
            self.filterListRegion.append('East')
        print(self.filterListRegion)


    def generateFilterList(self):
        self.generatefilterListForDiet()
        self.generatefilterListForFlavorPalate()
        self.generatefilterListForCourse()
        self.generatefilterListForPrepTime()
        self.generatefilterListCookTime()
        self.generatefilterListRegion()
        self.getRecommendations()

    def clearFilter(self):
        self.checkBox.setCheckState(False)
        self.checkBox_2.setCheckState(False)
        self.checkBox_3.setCheckState(False)
        self.checkBox_4.setCheckState(False)
        self.checkBox_5.setCheckState(False)
        self.checkBox_6.setCheckState(False)
        self.checkBox_7.setCheckState(False)
        self.checkBox_8.setCheckState(False)
        self.checkBox_9.setCheckState(False)
        self.checkBox_10.setCheckState(False)
        self.checkBox_11.setCheckState(False)
        self.checkBox_12.setCheckState(False)
        self.checkBox_13.setCheckState(False)
        self.checkBox_14.setCheckState(False)
        self.checkBox_15.setCheckState(False)
        self.checkBox_16.setCheckState(False)
        self.checkBox_17.setCheckState(False)
        self.checkBox_18.setCheckState(False)
        self.checkBox_19.setCheckState(False)
        self.checkBox_20.setCheckState(False)
        self.checkBox_21.setCheckState(False)
        self.checkBox_22.setCheckState(False)
        self.checkBox_23.setCheckState(False)

class GenerateImageThread(QThread):
    image_generated = pyqtSignal(QImage)

    def __init__(self, prompt1):
        super().__init__()
        self.prompt1 = prompt1

    def run(self):
        response = openai.Image.create(
        prompt=self.prompt1,
        n=1,
        size="256x256"
        )
        image_url = response['data'][0]['url']
        print(image_url)
        response = requests.get(image_url)
        image_data = response.content

        image = QImage()
        image.loadFromData(image_data)

        self.image_generated.emit(image)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
