#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- encoding:UTF-8 -*-
# coding=utf-8
# coding:utf-8

import codecs
from PyQt6.QtWidgets import (QWidget, QPushButton, QApplication,
							 QLabel, QHBoxLayout, QVBoxLayout, QLineEdit,
							 QSystemTrayIcon, QMenu, QComboBox, QDialog, QMenuBar, QFileDialog,
							 QTextEdit, QListWidget, QPlainTextEdit, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QRect, QObjectCleanupHandler
from PyQt6.QtGui import QAction, QIcon, QColor
import PyQt6.QtGui
import webbrowser
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import sys
import subprocess
import shutil
import html2text
import jieba
import glob
import datetime
import csv
from transformers import GPT2Tokenizer
import openai
import time
import markdown2
import signal
import pyperclip
import urllib3
import logging
import httpx
import asyncio
import re
from biplist import readPlist


app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

BasePath = '/Applications/Banana.app/Contents/Resources/'
# BasePath = ''  # test

# Create the icon
icon = QIcon(BasePath + "banana.icns")

# Create the tray
tray = QSystemTrayIcon()
tray.setIcon(icon)
tray.setVisible(True)

# Create the menu
menu = QMenu()

action3 = QAction("🍌 Show bananas!")
menu.addAction(action3)

action4 = QAction("🍴 Save this link in Safari!")
menu.addAction(action4)

menu.addSeparator()

action5 = QAction("🙆‍ Manually embed (AI)!")
menu.addAction(action5)

action6 = QAction("🤖 Chat with AI!")
action6.setCheckable(True)
menu.addAction(action6)

menu.addSeparator()

action8 = QAction("👀 Show delete button")
action8.setCheckable(True)
menu.addAction(action8)

action9 = QAction("🔖 Show Safari bookmarks!")
action9.setCheckable(True)
menu.addAction(action9)

menu.addSeparator()

action7 = QAction("⚙️ Settings")
menu.addAction(action7)

menu.addSeparator()

action2 = QAction("🆕 Check for Updates")
menu.addAction(action2)

action1 = QAction("ℹ️ About")
menu.addAction(action1)

menu.addSeparator()

# Add a Quit option to the menu.
quit = QAction("Quit")
quit.triggered.connect(app.quit)
menu.addAction(quit)

# Add the menu to the tray
tray.setContextMenu(menu)

# create a system menu
btna4 = QAction("&Show bananas!")
btna5 = QAction("&Save this link in Safari!")
btna6 = QAction("&Show Safari bookmarks!")
btna6.setCheckable(True)
sysmenu = QMenuBar()
file_menu = sysmenu.addMenu("&Actions")
file_menu.addAction(btna4)
file_menu.addAction(btna5)
file_menu.addAction(btna6)


class window_about(QWidget):  # 增加说明页面(About)
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):  # 说明页面内信息
		self.setUpMainWindow()
		self.resize(400, 380)
		self.center()
		self.setWindowTitle('About')
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

	def setUpMainWindow(self):
		widg1 = QWidget()
		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'banana.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setMaximumWidth(100)
		l1.setMaximumHeight(100)
		l1.setScaledContents(True)
		blay1 = QHBoxLayout()
		blay1.setContentsMargins(0, 0, 0, 0)
		blay1.addStretch()
		blay1.addWidget(l1)
		blay1.addStretch()
		widg1.setLayout(blay1)

		widg2 = QWidget()
		lbl0 = QLabel('Banana', self)
		font = PyQt6.QtGui.QFont()
		font.setFamily("Arial")
		font.setBold(True)
		font.setPointSize(20)
		lbl0.setFont(font)
		blay2 = QHBoxLayout()
		blay2.setContentsMargins(0, 0, 0, 0)
		blay2.addStretch()
		blay2.addWidget(lbl0)
		blay2.addStretch()
		widg2.setLayout(blay2)

		widg3 = QWidget()
		lbl1 = QLabel('Version 0.1.5', self)
		blay3 = QHBoxLayout()
		blay3.setContentsMargins(0, 0, 0, 0)
		blay3.addStretch()
		blay3.addWidget(lbl1)
		blay3.addStretch()
		widg3.setLayout(blay3)

		widg4 = QWidget()
		lbl2 = QLabel('Thanks for your love🤟.', self)
		blay4 = QHBoxLayout()
		blay4.setContentsMargins(0, 0, 0, 0)
		blay4.addStretch()
		blay4.addWidget(lbl2)
		blay4.addStretch()
		widg4.setLayout(blay4)

		widg5 = QWidget()
		lbl3 = QLabel('感谢您的喜爱！', self)
		blay5 = QHBoxLayout()
		blay5.setContentsMargins(0, 0, 0, 0)
		blay5.addStretch()
		blay5.addWidget(lbl3)
		blay5.addStretch()
		widg5.setLayout(blay5)

		widg6 = QWidget()
		lbl4 = QLabel('♥‿♥', self)
		blay6 = QHBoxLayout()
		blay6.setContentsMargins(0, 0, 0, 0)
		blay6.addStretch()
		blay6.addWidget(lbl4)
		blay6.addStretch()
		widg6.setLayout(blay6)

		widg7 = QWidget()
		lbl5 = QLabel('※\(^o^)/※', self)
		blay7 = QHBoxLayout()
		blay7.setContentsMargins(0, 0, 0, 0)
		blay7.addStretch()
		blay7.addWidget(lbl5)
		blay7.addStretch()
		widg7.setLayout(blay7)

		widg8 = QWidget()
		bt1 = QPushButton('The Author', self)
		bt1.setMaximumHeight(20)
		bt1.setMinimumWidth(100)
		bt1.clicked.connect(self.intro)
		bt2 = QPushButton('Github Page', self)
		bt2.setMaximumHeight(20)
		bt2.setMinimumWidth(100)
		bt2.clicked.connect(self.homepage)
		blay8 = QHBoxLayout()
		blay8.setContentsMargins(0, 0, 0, 0)
		blay8.addStretch()
		blay8.addWidget(bt1)
		blay8.addWidget(bt2)
		blay8.addStretch()
		widg8.setLayout(blay8)

		widg9 = QWidget()
		bt3 = QPushButton('🍪\n¥5', self)
		bt3.setMaximumHeight(50)
		bt3.setMinimumHeight(50)
		bt3.setMinimumWidth(50)
		bt3.clicked.connect(self.donate)
		bt4 = QPushButton('🥪\n¥10', self)
		bt4.setMaximumHeight(50)
		bt4.setMinimumHeight(50)
		bt4.setMinimumWidth(50)
		bt4.clicked.connect(self.donate2)
		bt5 = QPushButton('🍜\n¥20', self)
		bt5.setMaximumHeight(50)
		bt5.setMinimumHeight(50)
		bt5.setMinimumWidth(50)
		bt5.clicked.connect(self.donate3)
		bt6 = QPushButton('🍕\n¥50', self)
		bt6.setMaximumHeight(50)
		bt6.setMinimumHeight(50)
		bt6.setMinimumWidth(50)
		bt6.clicked.connect(self.donate4)
		blay9 = QHBoxLayout()
		blay9.setContentsMargins(0, 0, 0, 0)
		blay9.addStretch()
		blay9.addWidget(bt3)
		blay9.addWidget(bt4)
		blay9.addWidget(bt5)
		blay9.addWidget(bt6)
		blay9.addStretch()
		widg9.setLayout(blay9)

		widg10 = QWidget()
		lbl6 = QLabel('© 2023 Ryan-the-hito. All rights reserved.', self)
		blay10 = QHBoxLayout()
		blay10.setContentsMargins(0, 0, 0, 0)
		blay10.addStretch()
		blay10.addWidget(lbl6)
		blay10.addStretch()
		widg10.setLayout(blay10)

		main_h_box = QVBoxLayout()
		main_h_box.addWidget(widg1)
		main_h_box.addWidget(widg2)
		main_h_box.addWidget(widg3)
		main_h_box.addWidget(widg4)
		main_h_box.addWidget(widg5)
		main_h_box.addWidget(widg6)
		main_h_box.addWidget(widg7)
		main_h_box.addWidget(widg8)
		main_h_box.addWidget(widg9)
		main_h_box.addWidget(widg10)
		main_h_box.addStretch()
		self.setLayout(main_h_box)

	def intro(self):
		webbrowser.open('https://github.com/Ryan-the-hito/Ryan-the-hito')

	def homepage(self):
		webbrowser.open('https://github.com/Ryan-the-hito/Banana')

	def donate(self):
		dlg = CustomDialog()
		dlg.exec()

	def donate2(self):
		dlg = CustomDialog2()
		dlg.exec()

	def donate3(self):
		dlg = CustomDialog3()
		dlg.exec()

	def donate4(self):
		dlg = CustomDialog4()
		dlg.exec()

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def activate(self):  # 设置窗口显示
		self.show()


class CustomDialog(QDialog):  # (About1)
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setUpMainWindow()
		self.setWindowTitle("Thank you for your support!")
		self.center()
		self.resize(400, 390)
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

	def setUpMainWindow(self):
		widge_all = QWidget()
		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'wechat5.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setMaximumSize(160, 240)
		l1.setScaledContents(True)
		l2 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'alipay5.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l2.setPixmap(png)  # 在l2里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l2.setMaximumSize(160, 240)
		l2.setScaledContents(True)
		bk = QHBoxLayout()
		bk.setContentsMargins(0, 0, 0, 0)
		bk.addWidget(l1)
		bk.addWidget(l2)
		widge_all.setLayout(bk)

		m1 = QLabel('Thank you for your kind support! 😊', self)
		m2 = QLabel('I will write more interesting apps! 🥳', self)

		widg_c = QWidget()
		bt1 = QPushButton('Thank you!', self)
		bt1.setMaximumHeight(20)
		bt1.setMinimumWidth(100)
		bt1.clicked.connect(self.cancel)
		bt2 = QPushButton('Donate later~', self)
		bt2.setMaximumHeight(20)
		bt2.setMinimumWidth(100)
		bt2.clicked.connect(self.cancel)
		blay8 = QHBoxLayout()
		blay8.setContentsMargins(0, 0, 0, 0)
		blay8.addStretch()
		blay8.addWidget(bt1)
		blay8.addWidget(bt2)
		blay8.addStretch()
		widg_c.setLayout(blay8)

		self.layout = QVBoxLayout()
		self.layout.addWidget(widge_all)
		self.layout.addWidget(m1)
		self.layout.addWidget(m2)
		self.layout.addStretch()
		self.layout.addWidget(widg_c)
		self.layout.addStretch()
		self.setLayout(self.layout)

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def cancel(self):  # 设置取消键的功能
		self.close()


class CustomDialog2(QDialog):  # (About2)
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setUpMainWindow()
		self.setWindowTitle("Thank you for your support!")
		self.center()
		self.resize(400, 390)
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

	def setUpMainWindow(self):
		widge_all = QWidget()
		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'wechat10.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setMaximumSize(160, 240)
		l1.setScaledContents(True)
		l2 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'alipay10.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l2.setPixmap(png)  # 在l2里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l2.setMaximumSize(160, 240)
		l2.setScaledContents(True)
		bk = QHBoxLayout()
		bk.setContentsMargins(0, 0, 0, 0)
		bk.addWidget(l1)
		bk.addWidget(l2)
		widge_all.setLayout(bk)

		m1 = QLabel('Thank you for your kind support! 😊', self)
		m2 = QLabel('I will write more interesting apps! 🥳', self)

		widg_c = QWidget()
		bt1 = QPushButton('Thank you!', self)
		bt1.setMaximumHeight(20)
		bt1.setMinimumWidth(100)
		bt1.clicked.connect(self.cancel)
		bt2 = QPushButton('Donate later~', self)
		bt2.setMaximumHeight(20)
		bt2.setMinimumWidth(100)
		bt2.clicked.connect(self.cancel)
		blay8 = QHBoxLayout()
		blay8.setContentsMargins(0, 0, 0, 0)
		blay8.addStretch()
		blay8.addWidget(bt1)
		blay8.addWidget(bt2)
		blay8.addStretch()
		widg_c.setLayout(blay8)

		self.layout = QVBoxLayout()
		self.layout.addWidget(widge_all)
		self.layout.addWidget(m1)
		self.layout.addWidget(m2)
		self.layout.addStretch()
		self.layout.addWidget(widg_c)
		self.layout.addStretch()
		self.setLayout(self.layout)

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def cancel(self):  # 设置取消键的功能
		self.close()


class CustomDialog3(QDialog):  # (About3)
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setUpMainWindow()
		self.setWindowTitle("Thank you for your support!")
		self.center()
		self.resize(400, 390)
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

	def setUpMainWindow(self):
		widge_all = QWidget()
		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'wechat20.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setMaximumSize(160, 240)
		l1.setScaledContents(True)
		l2 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'alipay20.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l2.setPixmap(png)  # 在l2里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l2.setMaximumSize(160, 240)
		l2.setScaledContents(True)
		bk = QHBoxLayout()
		bk.setContentsMargins(0, 0, 0, 0)
		bk.addWidget(l1)
		bk.addWidget(l2)
		widge_all.setLayout(bk)

		m1 = QLabel('Thank you for your kind support! 😊', self)
		m2 = QLabel('I will write more interesting apps! 🥳', self)

		widg_c = QWidget()
		bt1 = QPushButton('Thank you!', self)
		bt1.setMaximumHeight(20)
		bt1.setMinimumWidth(100)
		bt1.clicked.connect(self.cancel)
		bt2 = QPushButton('Donate later~', self)
		bt2.setMaximumHeight(20)
		bt2.setMinimumWidth(100)
		bt2.clicked.connect(self.cancel)
		blay8 = QHBoxLayout()
		blay8.setContentsMargins(0, 0, 0, 0)
		blay8.addStretch()
		blay8.addWidget(bt1)
		blay8.addWidget(bt2)
		blay8.addStretch()
		widg_c.setLayout(blay8)

		self.layout = QVBoxLayout()
		self.layout.addWidget(widge_all)
		self.layout.addWidget(m1)
		self.layout.addWidget(m2)
		self.layout.addStretch()
		self.layout.addWidget(widg_c)
		self.layout.addStretch()
		self.setLayout(self.layout)

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def cancel(self):  # 设置取消键的功能
		self.close()


class CustomDialog4(QDialog):  # (About4)
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setUpMainWindow()
		self.setWindowTitle("Thank you for your support!")
		self.center()
		self.resize(400, 390)
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

	def setUpMainWindow(self):
		widge_all = QWidget()
		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'wechat50.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setMaximumSize(160, 240)
		l1.setScaledContents(True)
		l2 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'alipay50.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l2.setPixmap(png)  # 在l2里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l2.setMaximumSize(160, 240)
		l2.setScaledContents(True)
		bk = QHBoxLayout()
		bk.setContentsMargins(0, 0, 0, 0)
		bk.addWidget(l1)
		bk.addWidget(l2)
		widge_all.setLayout(bk)

		m1 = QLabel('Thank you for your kind support! 😊', self)
		m2 = QLabel('I will write more interesting apps! 🥳', self)

		widg_c = QWidget()
		bt1 = QPushButton('Thank you!', self)
		bt1.setMaximumHeight(20)
		bt1.setMinimumWidth(100)
		bt1.clicked.connect(self.cancel)
		bt2 = QPushButton('Donate later~', self)
		bt2.setMaximumHeight(20)
		bt2.setMinimumWidth(100)
		bt2.clicked.connect(self.cancel)
		blay8 = QHBoxLayout()
		blay8.setContentsMargins(0, 0, 0, 0)
		blay8.addStretch()
		blay8.addWidget(bt1)
		blay8.addWidget(bt2)
		blay8.addStretch()
		widg_c.setLayout(blay8)

		self.layout = QVBoxLayout()
		self.layout.addWidget(widge_all)
		self.layout.addWidget(m1)
		self.layout.addWidget(m2)
		self.layout.addStretch()
		self.layout.addWidget(widg_c)
		self.layout.addStretch()
		self.setLayout(self.layout)

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def cancel(self):  # 设置取消键的功能
		self.close()


class window_update(QWidget):  # 增加更新页面（Check for Updates）
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):  # 说明页面内信息

		self.lbl = QLabel('Current Version: v0.1.5', self)
		self.lbl.move(30, 45)

		lbl0 = QLabel('Download Update:', self)
		lbl0.move(30, 75)

		lbl1 = QLabel('Latest Version:', self)
		lbl1.move(30, 15)

		self.lbl2 = QLabel('', self)
		self.lbl2.move(122, 15)

		bt1 = QPushButton('Google Drive', self)
		bt1.setFixedWidth(120)
		bt1.clicked.connect(self.upd)
		bt1.move(150, 75)

		bt2 = QPushButton('Baidu Netdisk', self)
		bt2.setFixedWidth(120)
		bt2.clicked.connect(self.upd2)
		bt2.move(150, 105)

		self.resize(300, 150)
		self.center()
		self.setWindowTitle('Banana Updates')
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

	def upd(self):
		pass

	def upd2(self):
		pass

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def activate(self):  # 设置窗口显示
		self.show()
		self.checkupdate()

	def checkupdate(self):
		targetURL = 'https://github.com/Ryan-the-hito/Banana/releases'
		try:
			# Fetch the HTML content from the URL
			urllib3.disable_warnings()
			logging.captureWarnings(True)
			s = requests.session()
			s.keep_alive = False  # 关闭多余连接
			response = s.get(targetURL, verify=False)
			response.encoding = 'utf-8'
			html_content = response.text
			# Parse the HTML using BeautifulSoup
			soup = BeautifulSoup(html_content, "html.parser")
			# Remove all images from the parsed HTML
			for img in soup.find_all("img"):
				img.decompose()
			# Convert the parsed HTML to plain text using html2text
			text_maker = html2text.HTML2Text()
			text_maker.ignore_links = True
			text_maker.ignore_images = True
			plain_text = text_maker.handle(str(soup))
			# Convert the plain text to UTF-8
			plain_text_utf8 = plain_text.encode(response.encoding).decode("utf-8")

			for i in range(10):
				plain_text_utf8 = plain_text_utf8.replace('\n\n\n\n', '\n\n')
				plain_text_utf8 = plain_text_utf8.replace('\n\n\n', '\n\n')
				plain_text_utf8 = plain_text_utf8.replace('   ', ' ')
				plain_text_utf8 = plain_text_utf8.replace('  ', ' ')

			pattern2 = re.compile(r'(v\d+\.\d+\.\d+)\sLatest')
			result = pattern2.findall(plain_text_utf8)
			result = ''.join(result)
			nowversion = self.lbl.text().replace('Current Version: ', '')
			if result == nowversion:
				alertupdate = result + '. You are up to date!'
				self.lbl2.setText(alertupdate)
				self.lbl2.adjustSize()
			else:
				alertupdate = result + ' is ready!'
				self.lbl2.setText(alertupdate)
				self.lbl2.adjustSize()
		except:
			alertupdate = 'No Intrenet'
			self.lbl2.setText(alertupdate)
			self.lbl2.adjustSize()


class CustomDialog_warn(QDialog):  # save to banana
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setUpMainWindow()
		self.center()
		self.resize(300, 300)
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
		self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

	def setUpMainWindow(self):
		l0 = QLabel('Save to Banana🍌?', self)
		font = PyQt6.QtGui.QFont()
		font.setFamily("Arial")
		font.setBold(True)
		font.setPointSize(30)
		l0.setFont(font)

		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'banana.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setFixedSize(150, 150)
		l1.setScaledContents(True)

		self.choose_folder = QComboBox(self)
		self.choose_folder.setCurrentIndex(0)
		home_dir = str(Path.home())
		tarname1 = "BananaAppPath"
		fulldir1 = os.path.join(home_dir, tarname1)
		tarname2 = "Folder.txt"
		self.fullfolder = os.path.join(fulldir1, tarname2)
		textc = codecs.open(self.fullfolder, 'r', encoding='utf-8').read()
		if textc == '':
			with open(self.fullfolder, 'w', encoding='utf-8') as f0:
				f0.write('Default folder')
			self.choose_folder.addItems(['Default folder'])
		if textc != '':
			listc = textc.split('\n')
			while '' in listc:
				listc.remove('')
			self.choose_folder.addItems(listc)
		self.choose_folder.setFixedWidth(260)

		btn_no = QPushButton('Cancel', self)
		btn_no.clicked.connect(self.choosenot)
		btn_no.setFixedWidth(120)

		btn_can = QPushButton('Yes!', self)
		btn_can.clicked.connect(self.cancel)
		btn_can.setFixedWidth(120)

		w0 = QWidget()
		blay0 = QHBoxLayout()
		blay0.setContentsMargins(0, 0, 0, 0)
		blay0.addStretch()
		blay0.addWidget(l0)
		blay0.addStretch()
		w0.setLayout(blay0)

		w1 = QWidget()
		blay1 = QHBoxLayout()
		blay1.setContentsMargins(0, 0, 0, 0)
		blay1.addStretch()
		blay1.addWidget(l1)
		blay1.addStretch()
		w1.setLayout(blay1)

		w2 = QWidget()
		blay2 = QHBoxLayout()
		blay2.setContentsMargins(0, 0, 0, 0)
		blay2.addStretch()
		blay2.addWidget(self.choose_folder)
		blay2.addStretch()
		w2.setLayout(blay2)

		w2_1 = QWidget()
		blay2_1 = QHBoxLayout()
		blay2_1.setContentsMargins(0, 0, 0, 0)
		blay2_1.addStretch()
		blay2_1.addWidget(btn_no)
		blay2_1.addWidget(btn_can)
		blay2_1.addStretch()
		w2_1.setLayout(blay2_1)

		w3 = QWidget()
		blay3 = QVBoxLayout()
		blay3.setContentsMargins(20, 20, 20, 20)
		blay3.addStretch()
		blay3.addWidget(w1)
		blay3.addStretch()
		blay3.addWidget(w0)
		blay3.addStretch()
		blay3.addWidget(w2)
		blay3.addWidget(w2_1)
		w3.setLayout(blay3)
		w3.setObjectName("Main")

		blayend = QHBoxLayout()
		blayend.setContentsMargins(0, 0, 0, 0)
		blayend.addWidget(w3)
		self.setLayout(blayend)

	def center(self):  # 设置窗口居中
		# Get the primary screen's geometry
		screen_geometry = self.screen().availableGeometry()

		# Calculate the centered position
		x_center = int((screen_geometry.width() / 2) - (self.width() / 4))
		y_center = (screen_geometry.height() - self.height()) // 2

		# Move the window to the center position
		self.setGeometry(QRect(x_center, y_center, self.width(), self.height()))

	def choosenot(self):
		with open(BasePath + 'choose.txt', 'w', encoding='utf-8') as f0:
			f0.write('0')
		self.close()

	def cancel(self):  # 设置取消键的功能
		with open(BasePath + 'tarfolder.txt', 'w', encoding='utf-8') as f0:
			f0.write(self.choose_folder.currentText())
		with open(BasePath + 'choose.txt', 'w', encoding='utf-8') as f0:
			f0.write('1')
		self.close()


class CustomDialog_move(QDialog):  # move to folder
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setUpMainWindow()
		self.center()
		self.resize(300, 300)
		self.setFocus()
		self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
		self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

	def setUpMainWindow(self):
		l0 = QLabel('Move to...?', self)
		font = PyQt6.QtGui.QFont()
		font.setFamily("Arial")
		font.setBold(True)
		font.setPointSize(30)
		l0.setFont(font)

		l1 = QLabel(self)
		png = PyQt6.QtGui.QPixmap(BasePath + 'banana.png')  # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
		l1.setPixmap(png)  # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
		l1.setFixedSize(150, 150)
		l1.setScaledContents(True)

		self.choose_folder = QComboBox(self)
		self.choose_folder.setCurrentIndex(0)
		home_dir = str(Path.home())
		tarname1 = "BananaAppPath"
		fulldir1 = os.path.join(home_dir, tarname1)
		tarname2 = "Folder.txt"
		self.fullfolder = os.path.join(fulldir1, tarname2)
		textc = codecs.open(self.fullfolder, 'r', encoding='utf-8').read()
		if textc == '':
			with open(self.fullfolder, 'w', encoding='utf-8') as f0:
				f0.write('Default folder')
			self.choose_folder.addItems(['Default folder'])
		if textc != '':
			listc = textc.split('\n')
			while '' in listc:
				listc.remove('')
			self.choose_folder.addItems(listc)
		self.choose_folder.setFixedWidth(260)

		btn_no = QPushButton('Cancel', self)
		btn_no.clicked.connect(self.choosenot)
		btn_no.setFixedWidth(120)

		btn_can = QPushButton('Yes!', self)
		btn_can.clicked.connect(self.cancel)
		btn_can.setFixedWidth(120)

		w0 = QWidget()
		blay0 = QHBoxLayout()
		blay0.setContentsMargins(0, 0, 0, 0)
		blay0.addStretch()
		blay0.addWidget(l0)
		blay0.addStretch()
		w0.setLayout(blay0)

		w1 = QWidget()
		blay1 = QHBoxLayout()
		blay1.setContentsMargins(0, 0, 0, 0)
		blay1.addStretch()
		blay1.addWidget(l1)
		blay1.addStretch()
		w1.setLayout(blay1)

		w2 = QWidget()
		blay2 = QHBoxLayout()
		blay2.setContentsMargins(0, 0, 0, 0)
		blay2.addStretch()
		blay2.addWidget(self.choose_folder)
		blay2.addStretch()
		w2.setLayout(blay2)

		w2_1 = QWidget()
		blay2_1 = QHBoxLayout()
		blay2_1.setContentsMargins(0, 0, 0, 0)
		blay2_1.addStretch()
		blay2_1.addWidget(btn_no)
		blay2_1.addWidget(btn_can)
		blay2_1.addStretch()
		w2_1.setLayout(blay2_1)

		w3 = QWidget()
		blay3 = QVBoxLayout()
		blay3.setContentsMargins(20, 20, 20, 20)
		blay3.addStretch()
		blay3.addWidget(w1)
		blay3.addStretch()
		blay3.addWidget(w0)
		blay3.addStretch()
		blay3.addWidget(w2)
		blay3.addWidget(w2_1)
		w3.setLayout(blay3)
		w3.setObjectName("Main")

		blayend = QHBoxLayout()
		blayend.setContentsMargins(0, 0, 0, 0)
		blayend.addWidget(w3)
		self.setLayout(blayend)

	def center(self):  # 设置窗口居中
		# Get the primary screen's geometry
		screen_geometry = self.screen().availableGeometry()

		# Calculate the centered position
		x_center = int((screen_geometry.width() / 2) - (self.width() / 4))
		y_center = (screen_geometry.height() - self.height()) // 2

		# Move the window to the center position
		self.setGeometry(QRect(x_center, y_center, self.width(), self.height()))

	def choosenot(self):
		with open(BasePath + 'choose.txt', 'w', encoding='utf-8') as f0:
			f0.write('0')
		self.close()

	def cancel(self):  # 设置取消键的功能
		with open(BasePath + 'tarfolder.txt', 'w', encoding='utf-8') as f0:
			f0.write(self.choose_folder.currentText())
		with open(BasePath + 'choose.txt', 'w', encoding='utf-8') as f0:
			f0.write('1')
		self.close()


class TimeoutException(Exception):
	pass


class window3(QWidget):  # 主程序的代码块（Find a dirty word!）
	def __init__(self):
		super().__init__()
		self.dragPosition = self.pos()
		self.initUI()

	def initUI(self):  # 设置窗口内布局
		self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
		self.setFixedSize(1200, 741)

		home_dir = str(Path.home())
		tarname1 = "BananaAppPath"
		self.fulldir1 = os.path.join(home_dir, tarname1)
		if not os.path.exists(self.fulldir1):
			os.mkdir(self.fulldir1)

		tarname2 = "Folder.txt"
		self.fullfolder = os.path.join(self.fulldir1, tarname2)
		if not os.path.exists(self.fullfolder):
			with open(self.fullfolder, 'w', encoding='utf-8') as f0:
				f0.write('Default folder\nDeleted\n')

		tarname3 = "Records"
		self.fullrecord = os.path.join(self.fulldir1, tarname3)
		if not os.path.exists(self.fullrecord):
			os.mkdir(self.fullrecord)

		tarname4 = "Local"
		self.fulllocal = os.path.join(self.fulldir1, tarname4)
		if not os.path.exists(self.fulllocal):
			os.mkdir(self.fulllocal)

		tarname5 = "Index"
		self.fullIndex = os.path.join(self.fulldir1, tarname5)
		if not os.path.exists(self.fullIndex):
			os.mkdir(self.fullIndex)

		tarname6 = "Midindex"
		self.fullMidindex = os.path.join(self.fulldir1, tarname6)
		if not os.path.exists(self.fullMidindex):
			os.mkdir(self.fullMidindex)

		tarname7 = "Embed"
		self.fullEmbed = os.path.join(self.fulldir1, tarname7)
		if not os.path.exists(self.fullEmbed):
			os.mkdir(self.fullEmbed)

		tarname7_5 = "Midembed"
		self.fullMidembed = os.path.join(self.fulldir1, tarname7_5)
		if not os.path.exists(self.fullMidembed):
			os.mkdir(self.fullMidembed)

		tarname7_6 = "Default folder.txt"
		self.fulldefault = os.path.join(self.fulldir1, tarname7_6)
		if not os.path.exists(self.fulldefault):
			with open(self.fulldefault, 'w', encoding='utf-8') as f0:
				f0.write('')

		tarname7_7 = "Allsearch.txt"
		self.fullse = os.path.join(self.fulldir1, tarname7_7)
		if not os.path.exists(self.fullse):
			with open(self.fullse, 'w', encoding='utf-8') as f0:
				f0.write('')

		tarname7_8 = "Deleted.txt"
		self.fulldel = os.path.join(self.fulldir1, tarname7_8)
		if not os.path.exists(self.fulldel):
			with open(self.fulldel, 'w', encoding='utf-8') as f0:
				f0.write('')

		tarname8 = "Allembed.csv"
		self.fullall1 = os.path.join(self.fulldir1, tarname8)
		if not os.path.exists(self.fullall1):
			with open(self.fullall1, 'w', encoding='utf-8') as f0:
				f0.write('')

		tarname9 = "Allindex.csv"
		self.fullall2 = os.path.join(self.fulldir1, tarname9)
		if not os.path.exists(self.fullall2):
			with open(self.fullall2, 'w', encoding='utf-8') as f0:
				f0.write('')

		tarname10 = "webarchiver.command"
		self.fullcmd = os.path.join(self.fulldir1, tarname10)
		if not os.path.exists(self.fullcmd):
			shutil.copy('webarchiver.command', self.fulldir1)

		self.setUpMainWindow()
		#self.listenshorcut()

		self.center()
		self.setWindowTitle('Webpage Archiver!')

		app.setStyleSheet(style_sheet_ori)

	def setUpMainWindow(self):
		self.le1 = QLineEdit(self)
		self.le1.setPlaceholderText('URL here...')
		self.le1.setFixedHeight(20)

		self.btn1 = QPushButton('Archive this!', self)
		self.btn1.clicked.connect(self.archivethis)
		self.btn1.setFixedSize(120, 20)

		self.le2 = QLineEdit(self)
		self.le2.setPlaceholderText('Search for a title or URL here...')
		self.le2.setFixedHeight(20)
		self.le2.textChanged.connect(self.searchitem)

		self.folder_list = QListWidget(self)
		self.folder_list.itemClicked.connect(self.showcontent)
		folders = codecs.open(self.fullfolder, 'r', encoding='utf-8').read()
		tolist = folders.split('\n')
		while '' in tolist:
			tolist.remove('')
		self.folder_list.addItems(tolist)

		self.le3 = QLineEdit(self)
		self.le3.setPlaceholderText('New folder name here...')
		self.le3.setFixedHeight(20)

		self.btn2 = QPushButton('+', self)
		self.btn2.clicked.connect(self.addfolder)
		self.btn2.setFixedSize(120, 20)

		self.btn3 = QPushButton('-', self)
		self.btn3.clicked.connect(self.deletefolder)
		self.btn3.setFixedSize(120, 20)
		self.btn3.setVisible(False)

		self.item_list = QListWidget(self)

		self.btn4 = QPushButton('Open link', self)
		self.btn4.clicked.connect(self.openlink)

		self.btn5 = QPushButton('Copy link', self)
		self.btn5.clicked.connect(self.copylink)

		self.btn6 = QPushButton('Open archive', self)
		self.btn6.clicked.connect(self.openarchive)

		self.btn7 = QPushButton('Delete link', self)
		self.btn7.clicked.connect(self.deleteitem)
		self.btn7.setVisible(False)

		self.btn8 = QPushButton('Move to', self)
		self.btn8.clicked.connect(self.moveto)

		self.real1 = QTextEdit(self)
		self.real1.setReadOnly(True)
		self.real1.setFixedHeight(200)

		self.text1 = QPlainTextEdit(self)
		self.text1.setReadOnly(False)
		self.text1.setObjectName('edit')
		self.text1.setFixedHeight(100)
		self.text1.setPlaceholderText('Your prompts here...')

		self.widget0 = QComboBox(self)
		self.widget0.setCurrentIndex(0)
		allit_list = os.listdir(self.fullEmbed)
		while '.DS_Store' in allit_list:
			allit_list.remove('.DS_Store')
		while '' in allit_list:
			allit_list.remove('')
		allname_list = ['Context: All']
		if allit_list != []:
			for i in range(len(allit_list)):
				if '.csv' in allit_list[i]:
					allname_list.append(allit_list[i])
		self.widget0.addItems(allname_list)
		self.widget0.currentIndexChanged.connect(self.whichchat)

		self.btn_sub1 = QPushButton('🔺 Send', self)
		self.btn_sub1.clicked.connect(self.searchchat)
		self.btn_sub1.setFixedSize(80, 20)
		self.btn_sub1.setShortcut("Ctrl+Return")

		self.btn_sub2 = QPushButton('🔸 Clear', self)
		self.btn_sub2.clicked.connect(self.clearall)
		self.btn_sub2.setFixedSize(80, 20)

		self.btn_sub3 = QPushButton('🔻 Save', self)
		self.btn_sub3.clicked.connect(self.exportfile)
		self.btn_sub3.setFixedSize(80, 20)

		qw1 = QWidget()
		vbox1 = QHBoxLayout()
		vbox1.setContentsMargins(0, 0, 0, 0)
		vbox1.addWidget(self.le1)
		vbox1.addWidget(self.btn1)
		qw1.setLayout(vbox1)

		qw2 = QWidget()
		vbox2 = QVBoxLayout()
		vbox2.setContentsMargins(0, 0, 0, 0)
		vbox2.addWidget(qw1)
		vbox2.addWidget(self.le2)
		qw2.setLayout(vbox2)

		qw3 = QWidget()
		vbox3 = QHBoxLayout()
		vbox3.setContentsMargins(0, 0, 0, 0)
		vbox3.addWidget(self.le3)
		vbox3.addWidget(self.btn2)
		vbox3.addWidget(self.btn3)
		qw3.setLayout(vbox3)

		qw4 = QWidget()
		vbox4 = QVBoxLayout()
		vbox4.setContentsMargins(0, 0, 0, 0)
		vbox4.addWidget(self.folder_list)
		vbox4.addWidget(qw3)
		qw4.setLayout(vbox4)
		qw4.setFixedWidth(443)

		qw5 = QWidget()
		vbox5 = QHBoxLayout()
		vbox5.setContentsMargins(0, 0, 0, 0)
		vbox5.addWidget(self.btn4)
		vbox5.addWidget(self.btn5)
		vbox5.addWidget(self.btn6)
		vbox5.addWidget(self.btn7)
		vbox5.addWidget(self.btn8)
		qw5.setLayout(vbox5)

		qw6 = QWidget()
		vbox6 = QVBoxLayout()
		vbox6.setContentsMargins(0, 0, 0, 0)
		vbox6.addWidget(self.item_list)
		vbox6.addWidget(qw5)
		qw6.setLayout(vbox6)

		qw6_1 = QWidget()
		vbox6_1 = QHBoxLayout()
		vbox6_1.setContentsMargins(0, 0, 0, 0)
		vbox6_1.addWidget(qw4)
		vbox6_1.addWidget(qw6)
		qw6_1.setLayout(vbox6_1)

		qw7 = QWidget()
		vbox7 = QVBoxLayout()
		vbox7.setContentsMargins(0, 5, 0, 0)
		vbox7.addWidget(self.btn_sub1)
		vbox7.addStretch()
		vbox7.addWidget(self.btn_sub2)
		vbox7.addStretch()
		vbox7.addWidget(self.btn_sub3)
		qw7.setLayout(vbox7)

		qw7_1 = QWidget()
		vbox2 = QVBoxLayout()
		vbox2.setContentsMargins(0, 0, 0, 0)
		vbox2.addWidget(self.widget0)
		vbox2.addStretch()
		vbox2.addWidget(self.text1)
		qw7_1.setLayout(vbox2)

		qw8 = QWidget()
		vbox8 = QHBoxLayout()
		vbox8.setContentsMargins(0, 0, 0, 0)
		vbox8.addWidget(qw7_1)
		vbox8.addWidget(qw7)
		qw8.setLayout(vbox8)

		self.qw9 = QWidget()
		vbox9 = QVBoxLayout()
		vbox9.setContentsMargins(0, 0, 0, 0)
		vbox9.addWidget(self.real1)
		vbox9.addWidget(qw8)
		self.qw9.setLayout(vbox9)
		self.qw9.setVisible(False)

		vbox10 = QVBoxLayout()
		vbox10.setContentsMargins(20, 20, 20, 20)
		vbox10.addWidget(qw2)
		vbox10.addWidget(qw6_1)
		vbox10.addWidget(self.qw9)
		self.setLayout(vbox10)

	def archivethis(self):  # save+index+show
		warn = CustomDialog_warn()
		warn.exec()
		textc = codecs.open(BasePath + 'choose.txt', 'r', encoding='utf-8').read()
		if textc == '1':
			targetURL = self.le1.text()
			signal.signal(signal.SIGALRM, self.timeout_handler)
			signal.alarm(30)
			if targetURL != '':
				try:
					script = """
						set targetURL to "%s"
						tell application "Safari"
							activate
							open location targetURL
							delay 5
							set currentTab to current tab of window 1
							set fileName to name of currentTab as string
							tell front window
								set currentTab to current tab
								tell currentTab
									repeat until (do JavaScript "document.readyState") is "complete"
										delay 0.1
									end repeat
									set scrollHeight to do JavaScript "document.body.scrollHeight" in currentTab
									set scrollTop to do JavaScript "document.body.scrollTop" in currentTab
									set windowHeight to do JavaScript "window.innerHeight" in currentTab
									set scrollPosition to scrollHeight - windowHeight
									repeat while scrollTop < scrollPosition
										set scrollTop to scrollTop + 200
										do JavaScript "window.scrollTo(0, " & scrollTop & ")" in currentTab
										delay 0.01
									end repeat
								end tell
							end tell
						end tell""" % (targetURL)
					subprocess.run(['osascript', '-l', 'AppleScript', '-e', script], capture_output=True, text=True, encoding='utf-8')
					# Fetch the HTML content from the URL
					urllib3.disable_warnings()
					logging.captureWarnings(True)
					s = requests.session()
					s.keep_alive = False  # 关闭多余连接
					response = s.get(targetURL, verify=False)
					response.encoding = 'utf-8'  # Set the encoding based on the response content
					html_content = response.text
					# Parse the HTML using BeautifulSoup
					soup = BeautifulSoup(html_content, "html.parser")
					# Get the title tag
					title_tag = soup.find("title")
					# Return the text content of the title tag
					title_page = title_tag.text if title_tag else None
					self.endText = title_page.replace(' ', '_').replace(':', '').replace('|', '').replace('__', '_').replace('/', '_')
					if self.endText == None or self.endText == '':
						script = """
							set targetURL to "%s"
							tell application "Safari"
								activate
								set currentTab to current tab of window 1
								set fileName to name of currentTab as string
								return fileName
							end tell""" % (targetURL)
						result = subprocess.run(['osascript', '-l', 'AppleScript', '-e', script], capture_output=True,
												text=True, encoding='utf-8')
						self.endText = result.stdout.strip()
						self.endText = self.endText.replace(' ', '_').replace(':', '').replace('|', '').replace('__', '_').replace('/', '_')
						if self.endText == None or self.endText == '':
							self.endText = 'newitem'
					CMD = '''
						on run argv
						  display notification (item 2 of argv) with title (item 1 of argv)
						end run
						'''
					self.notify(CMD, "Banana: Webpage Archiver",
								f"The targeted webpage is being webarchived! \n({self.endText})")
					cmd = """
						set fileName to "%s"
						
						tell application "Safari" to set targetURL to (URL of document 1) as string
						set commandPath to "~/BananaAppPath/webarchiver.command"
						do shell script "" & commandPath & " -url " & targetURL & " -output ~/BananaAppPath/Records/" & fileName & ".webarchive"
						tell application "Safari"
							set currentTab to current tab of window 1
							close currentTab
						end tell
						""" % self.endText
					subprocess.call(['osascript', '-e', cmd])
					file_name = self.endText + '.webarchive'
					record_path = os.path.join(self.fullrecord, file_name)
					if not os.path.exists(record_path):
						cmd = """
							tell application "Safari"
								activate
								set currentTab to current tab of window 1
								tell application "System Events"
									keystroke "s" using {command down, shift down}
									delay 2
									keystroke "d" using {command down} -- save to Desktop
									delay 1
									tell application process "Safari"
										tell pop up button 1 of window 1
											click
											click menu item "Web Archive" of menu 1
										end tell
									end tell
									keystroke return
									delay 1
									keystroke "w" using {command down}
								end tell
							end tell
						"""
						subprocess.call(['osascript', '-e', cmd])
						home_dir = str(Path.home())
						tarname1 = "Desktop"
						fulldir1 = os.path.join(home_dir, tarname1)
						fulldir1 = fulldir1 + '/*.webarchive'
						# List of files
						list_of_files = glob.glob(fulldir1)
						if list_of_files != []:
							# Get most recent created .html file
							latest_file = max(list_of_files, key=os.path.getctime)
							# Rename file
							os.rename(src=latest_file,
									  dst=record_path)
							#os.remove(latest_file)
						if list_of_files == []:
							CMD = '''
								on run argv
								  display notification (item 2 of argv) with title (item 1 of argv)
								end run
								'''
							self.notify(CMD, "Banana: Webpage Archiver",
										f"Failed in storing this web page! \n({self.endText})")
					if os.path.exists(record_path):
						tar_local = self.endText + '.txt'
						output_local = os.path.join(self.fulllocal, tar_local)
						# Fetch the HTML content from the URL
						urllib3.disable_warnings()
						logging.captureWarnings(True)
						s = requests.session()
						s.keep_alive = False  # 关闭多余连接
						response = s.get(targetURL, verify=False)
						response.encoding = 'utf-8'
						html_content = response.text
						# Parse the HTML using BeautifulSoup
						soup = BeautifulSoup(html_content, "html.parser")
						# Remove all images from the parsed HTML
						for img in soup.find_all("img"):
							img.decompose()
						# Convert the parsed HTML to plain text using html2text
						text_maker = html2text.HTML2Text()
						text_maker.ignore_links = True
						text_maker.ignore_images = True
						plain_text = text_maker.handle(str(soup))
						# Convert the plain text to UTF-8
						plain_text_utf8 = plain_text.encode(response.encoding).decode("utf-8")

						for i in range(10):
							plain_text_utf8 = plain_text_utf8.replace('\n\n\n\n', '\n\n')
							plain_text_utf8 = plain_text_utf8.replace('\n\n\n', '\n\n')
							plain_text_utf8 = plain_text_utf8.replace('   ', ' ')
							plain_text_utf8 = plain_text_utf8.replace('  ', ' ')

						plain_list = plain_text_utf8.split('\n\n')
						del_list = []
						for i in range(len(plain_list)):
							aj = jieba.cut(plain_list[i], cut_all=False)
							paj = '/'.join(aj)
							saj = paj.split('/')
							if len(plain_list[i]) < 100:
								del_list.append(plain_list[i])
							if len(saj) > 500:
								ter = saj[0:499]
								tarstr = ' '.join(ter)
								plain_list[i] = tarstr
						end_list = list(set(plain_list) - set(del_list))
						for n in range(len(end_list)):
							end_list[n] = self.default_clean(self.cleanlinebreak(end_list[n])) + '<SOURCE: ' + self.endText + '>'
						end_text = '✡'.join(end_list)
						for i in range(10):
							end_text = end_text.replace('   ', ' ')
							end_text = end_text.replace('  ', ' ')
						end_text = end_text.replace('\n', '')
						end_text = end_text.replace('✡', '\n\n')
						if end_list == [] or end_text == '' or end_text is None:
							end_text = self.endText

						# Save the plain text to a file
						with open(output_local, "w", encoding="utf-8") as f:
							f.write(end_text)

						csv_line = end_text.replace(',', ';').split('\n\n')
						for x in range(len(csv_line)):
							csv_line[x] = "A" + ',' + "B" + ',' + csv_line[x]
						csvtext = '\n'.join(csv_line)
						csvtext = 'title,heading,content\n' + csvtext
						csv_endtar = self.endText + '.csv'
						csv_tarname = os.path.join(self.fullIndex, csv_endtar)
						with open(csv_tarname, 'w', encoding='utf-8') as f0:
							f0.write(csvtext)
						tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
						# 打开 CSV 文件并读取数据
						with open(csv_tarname, mode='r', encoding='utf-8') as csv_file:
							csv_reader = csv.reader(csv_file)
							rows = list(csv_reader)
						# 在数据中添加新列
						header = rows[0]
						header.append('tokens')
						for row in rows[1:]:
							tar = row[-1]
							A = tokenizer.encode(tar, add_special_tokens=True)
							if len(A) <= 1024:
								row.append(str(len(A)))
							else:
								row.append(str(1024))
						# 将更新后的数据写回 CSV 文件
						with open(csv_tarname, mode='w', newline='', encoding='utf-8') as csv_file:
							csv_writer = csv.writer(csv_file)
							csv_writer.writerow(header)
							csv_writer.writerows(rows[1:])

						# delete those which are too long
						cleanlong = codecs.open(csv_tarname, 'r', encoding='utf-8').read()
						cleanlong = cleanlong.replace('\r', '')
						cleanlong_list = cleanlong.split('\n')
						while '' in cleanlong_list:
							cleanlong_list.remove('')
						del cleanlong_list[0]
						lostlist = []
						for f in range(len(cleanlong_list)):
							pattern = re.compile(r',(\d+)$')
							result = pattern.findall(cleanlong_list[f])
							if result != []:
								realnum = int(''.join(result))
								if realnum >= 1024:
									lostlist.append(cleanlong_list[f])
						reallist = list(set(cleanlong_list) - set(lostlist))
						realcsv = '\n'.join(reallist)
						realcsv = 'title,heading,content,tokens\n' + realcsv
						with open(csv_tarname, 'w', encoding='utf-8') as f0:
							f0.write(realcsv)

						shutil.copy(csv_tarname, self.fullMidindex)

						# display
						tarfolder = codecs.open(BasePath + 'tarfolder.txt', 'r', encoding='utf-8').read()
						folderitem = tarfolder + '.txt'
						tarpath = os.path.join(self.fulldir1, folderitem)
						with open(tarpath, 'a', encoding='utf-8') as f0:
							f0.write(targetURL + '✡✡' + self.endText + '\n')
						with open(self.fullse, 'a', encoding='utf-8') as f0:
							f0.write(targetURL + '✡✡' + self.endText + '\n')

						# notify
						CMD = '''
							on run argv
							  display notification (item 2 of argv) with title (item 1 of argv)
							end run
							'''
						self.notify(CMD, "Banana: Webpage Archiver", f"You have successfully stored a webarchive! \n({self.endText})")
				except TimeoutException:
					CMD = '''
						on run argv
						  display notification (item 2 of argv) with title (item 1 of argv)
						end run
						'''
					self.notify(CMD, "Banana: Webpage Archiver", f"Time out, please try again! \n({self.endText})")
				except Exception as e:
					with open(BasePath + 'er.txt', 'w', encoding='utf-8') as f0:
						f0.write(str(e))
			if targetURL == '':
				try:
					script = """
						tell application "Safari"
							activate
							tell front window
								set currentTab to current tab
								tell currentTab
									repeat until (do JavaScript "document.readyState") is "complete"
										delay 0.1
									end repeat
									set scrollHeight to do JavaScript "document.body.scrollHeight" in currentTab
									set scrollTop to do JavaScript "document.body.scrollTop" in currentTab
									set windowHeight to do JavaScript "window.innerHeight" in currentTab
									set scrollPosition to scrollHeight - windowHeight
									repeat while scrollTop < scrollPosition
										set scrollTop to scrollTop + 200
										do JavaScript "window.scrollTo(0, " & scrollTop & ")" in currentTab
										delay 0.01
									end repeat
								end tell
							end tell
							tell application "Safari" to set targetURL to (URL of document 1) as string
							return targetURL
						end tell"""
					result = subprocess.run(['osascript', '-l', 'AppleScript', '-e', script], capture_output=True,
											text=True, encoding='utf-8')
					targetURL = result.stdout.strip()

					# Fetch the HTML content from the URL
					urllib3.disable_warnings()
					logging.captureWarnings(True)
					s = requests.session()
					s.keep_alive = False  # 关闭多余连接
					response = s.get(targetURL, verify=False)
					response.encoding = 'utf-8'  # Set the encoding based on the response content
					html_content = response.text
					# Parse the HTML using BeautifulSoup
					soup = BeautifulSoup(html_content, "html.parser")
					# Get the title tag
					title_tag = soup.find("title")
					# Return the text content of the title tag
					title_page = title_tag.text if title_tag else None
					self.endText = title_page.replace(' ', '_').replace(':', '').replace('|', '').replace('__', '_').replace('/', '_')
					if self.endText == None or self.endText == '':
						script = """
							tell application "Safari"
								activate
								set currentTab to current tab of window 1
								set fileName to name of currentTab as string
								return fileName
							end tell"""
						result = subprocess.run(['osascript', '-l', 'AppleScript', '-e', script], capture_output=True,
												text=True, encoding='utf-8')
						self.endText = result.stdout.strip()
						self.endText = self.endText.replace(' ', '_').replace(':', '').replace('|', '').replace('__', '_').replace('/', '_')
						if self.endText == None or self.endText == '':
							self.endText = 'newitem'
					CMD = '''
						on run argv
						  display notification (item 2 of argv) with title (item 1 of argv)
						end run
						'''
					self.notify(CMD, "Banana: Webpage Archiver",
								f"The targeted webpage is being webarchived! \n({self.endText})")
					cmd = """
						set fileName to "%s"
	
						tell application "Safari" to set targetURL to (URL of document 1) as string
						set commandPath to "~/BananaAppPath/webarchiver.command"
						do shell script "" & commandPath & " -url " & targetURL & " -output ~/BananaAppPath/Records/" & fileName & ".webarchive"
						tell application "Safari"
							set currentTab to current tab of window 1
							close currentTab
						end tell
						""" % self.endText
					subprocess.call(['osascript', '-e', cmd])
					file_name = self.endText + '.webarchive'
					record_path = os.path.join(self.fullrecord, file_name)
					if not os.path.exists(record_path):
						cmd = """
							tell application "Safari"
								activate
								set currentTab to current tab of window 1
								tell application "System Events"
									keystroke "s" using {command down, shift down}
									delay 1
									keystroke "d" using {command down} -- save to Desktop
									delay 1
									tell application process "Safari"
										tell pop up button 1 of window 1
											click
											click menu item "Web Archive" of menu 1
										end tell
									end tell
									keystroke return
									delay 1
									keystroke "w" using {command down}
								end tell
							end tell
						"""
						subprocess.call(['osascript', '-e', cmd])
						home_dir = str(Path.home())
						tarname1 = "Desktop"
						fulldir1 = os.path.join(home_dir, tarname1)
						fulldir1 = fulldir1 + '/*.webarchive'
						# List of files
						list_of_files = glob.glob(fulldir1)
						if list_of_files != []:
							# Get most recent created .html file
							latest_file = max(list_of_files, key=os.path.getctime)
							# Rename file
							os.rename(src=latest_file,
									  dst=record_path)
							#os.remove(latest_file)
						if list_of_files == []:
							CMD = '''
								on run argv
								  display notification (item 2 of argv) with title (item 1 of argv)
								end run
								'''
							self.notify(CMD, "Banana: Webpage Archiver",
										f"Failed in storing this web page! \n({self.endText})")
					if os.path.exists(record_path):
						tar_local = self.endText + '.txt'
						output_local = os.path.join(self.fulllocal, tar_local)
						# Fetch the HTML content from the URL
						urllib3.disable_warnings()
						logging.captureWarnings(True)
						s = requests.session()
						s.keep_alive = False  # 关闭多余连接
						response = s.get(targetURL, verify=False)
						response.encoding = 'utf-8'  # Set the encoding based on the response content
						html_content = response.text
						# Parse the HTML using BeautifulSoup
						soup = BeautifulSoup(html_content, "html.parser")
						# Remove all images from the parsed HTML
						for img in soup.find_all("img"):
							img.decompose()
						# Convert the parsed HTML to plain text using html2text
						text_maker = html2text.HTML2Text()
						text_maker.ignore_links = True
						text_maker.ignore_images = True
						plain_text = text_maker.handle(str(soup))
						# Convert the plain text to UTF-8
						plain_text_utf8 = plain_text.encode(response.encoding).decode("utf-8")

						for i in range(10):
							plain_text_utf8 = plain_text_utf8.replace('\n\n\n\n', '\n\n')
							plain_text_utf8 = plain_text_utf8.replace('\n\n\n', '\n\n')
							plain_text_utf8 = plain_text_utf8.replace('   ', ' ')
							plain_text_utf8 = plain_text_utf8.replace('  ', ' ')

						plain_list = plain_text_utf8.split('\n\n')
						del_list = []
						for i in range(len(plain_list)):
							aj = jieba.cut(plain_list[i], cut_all=False)
							paj = '/'.join(aj)
							saj = paj.split('/')
							if len(plain_list[i]) < 100:
								del_list.append(plain_list[i])
							if len(saj) > 500:
								ter = saj[0:499]
								tarstr = ' '.join(ter)
								plain_list[i] = tarstr
						end_list = list(set(plain_list) - set(del_list))
						for n in range(len(end_list)):
							end_list[n] = self.default_clean(self.cleanlinebreak(end_list[n])) + '<SOURCE: ' + self.endText + '>'
						end_text = '✡'.join(end_list)
						for i in range(10):
							end_text = end_text.replace('   ', ' ')
							end_text = end_text.replace('  ', ' ')
						end_text = end_text.replace('\n', '')
						end_text = end_text.replace('✡', '\n\n')
						if end_list == [] or end_text == '' or end_text is None:
							end_text = self.endText

						# Save the plain text to a file
						with open(output_local, "w", encoding="utf-8") as f:
							f.write(end_text)

						csv_line = end_text.replace(',', ';').split('\n\n')
						for x in range(len(csv_line)):
							csv_line[x] = "A" + ',' + "B" + ',' + csv_line[x]
						csvtext = '\n'.join(csv_line)
						csvtext = 'title,heading,content\n' + csvtext
						csv_endtar = self.endText + '.csv'
						csv_tarname = os.path.join(self.fullIndex, csv_endtar)
						with open(csv_tarname, 'w', encoding='utf-8') as f0:
							f0.write(csvtext)
						tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
						# 打开 CSV 文件并读取数据
						with open(csv_tarname, mode='r', encoding='utf-8') as csv_file:
							csv_reader = csv.reader(csv_file)
							rows = list(csv_reader)
						# 在数据中添加新列
						header = rows[0]
						header.append('tokens')
						for row in rows[1:]:
							tar = row[-1]
							A = tokenizer.encode(tar, add_special_tokens=True)
							if len(A) <= 1024:
								row.append(str(len(A)))
							else:
								row.append(str(1024))
						# 将更新后的数据写回 CSV 文件
						with open(csv_tarname, mode='w', newline='', encoding='utf-8') as csv_file:
							csv_writer = csv.writer(csv_file)
							csv_writer.writerow(header)
							csv_writer.writerows(rows[1:])

						# delete those which are too long
						cleanlong = codecs.open(csv_tarname, 'r', encoding='utf-8').read()
						cleanlong = cleanlong.replace('\r', '')
						cleanlong_list = cleanlong.split('\n')
						while '' in cleanlong_list:
							cleanlong_list.remove('')
						del cleanlong_list[0]
						lostlist = []
						for f in range(len(cleanlong_list)):
							pattern = re.compile(r',(\d+)$')
							result = pattern.findall(cleanlong_list[f])
							if result != []:
								realnum = int(''.join(result))
								if realnum >= 1024:
									lostlist.append(cleanlong_list[f])
						reallist = list(set(cleanlong_list) - set(lostlist))
						realcsv = '\n'.join(reallist)
						realcsv = 'title,heading,content,tokens\n' + realcsv
						with open(csv_tarname, 'w', encoding='utf-8') as f0:
							f0.write(realcsv)

						shutil.copy(csv_tarname, self.fullMidindex)

						# display
						tarfolder = codecs.open(BasePath + 'tarfolder.txt', 'r', encoding='utf-8').read()
						folderitem = tarfolder + '.txt'
						tarpath = os.path.join(self.fulldir1, folderitem)
						with open(tarpath, 'a', encoding='utf-8') as f0:
							f0.write(targetURL + '✡✡' + self.endText + '\n')
						with open(self.fullse, 'a', encoding='utf-8') as f0:
							f0.write(targetURL + '✡✡' + self.endText + '\n')

						# notify
						CMD = '''
							on run argv
							  display notification (item 2 of argv) with title (item 1 of argv)
							end run
							'''
						self.notify(CMD, "Banana: Webpage Archiver", f"You have successfully stored a webarchive! \n({self.endText})")
				except TimeoutException:
					CMD = '''
						on run argv
						  display notification (item 2 of argv) with title (item 1 of argv)
						end run
						'''
					self.notify(CMD, "Banana: Webpage Archiver", f"Time out, please try again! \n({self.endText})")
				except Exception as e:
					with open(BasePath + 'er.txt', 'w', encoding='utf-8') as f0:
						f0.write(str(e))
			signal.alarm(0)  # reset timer
			self.le1.clear()
			with open(BasePath + 'choose.txt', 'w', encoding='utf-8') as f0:
				f0.write('0')

			# refresh display
			if self.folder_list.currentItem() != None:
				endname = self.folder_list.currentItem().text() + '.txt'
				tarfile = os.path.join(self.fulldir1, endname)
				items = codecs.open(tarfile, 'r', encoding='utf-8').read()
				item_list = items.split('\n')
				while '' in item_list:
					item_list.remove('')
				showlist = []
				for i in range(len(item_list)):
					showlist.append(item_list[i].split('✡✡')[1])
				self.item_list.clear()
				self.item_list.addItems(showlist)

	def showcontent(self, item):
		endname = item.text() + '.txt'
		tarfile = os.path.join(self.fulldir1, endname)
		items = codecs.open(tarfile, 'r', encoding='utf-8').read()
		item_list = items.split('\n')
		while '' in item_list:
			item_list.remove('')
		showlist = []
		for i in range(len(item_list)):
			showlist.append(item_list[i].split('✡✡')[1])
		self.item_list.clear()
		self.item_list.addItems(showlist)

	def embeditem(self):
		AccountGPT = codecs.open(BasePath + 'api.txt', 'r', encoding='utf-8').read()
		if AccountGPT != '':
			SUCC = 0
			icon = QIcon(BasePath + "embeding.icns")
			tray.setIcon(icon)
			tray.setVisible(True)
			list_dir = os.listdir(self.fullMidindex)
			while '.DS_Store' in list_dir:
				list_dir.remove('.DS_Store')
			oricon = codecs.open(self.fullall2, 'r', encoding='utf-8').read()
			if oricon == '':
				with open(self.fullall2, 'a', encoding='utf-8') as f0:
					f0.write('title,heading,content,tokens\n')
			if list_dir != []:
				for i in range(len(list_dir)):
					if 0 <= i <= 4:
						QApplication.processEvents()
						QApplication.restoreOverrideCursor()
						tarnamecsv = os.path.join(self.fullMidindex, list_dir[i])
						embedcsv = os.path.join(self.fullEmbed, list_dir[i])
						try:
							QApplication.processEvents()
							QApplication.restoreOverrideCursor()
							CMD = '''
								on run argv
								  display notification (item 2 of argv) with title (item 1 of argv)
								end run
								'''
							xnum = len(list_dir)
							if len(list_dir) > 5:
								xnum = 5
							xper = str(int(((i + 1) / xnum) * 100)) + '%'
							self.notify(CMD, "Banana: Webpage Archiver", f"{str(i + 1)}/{xnum} is being embedded! ({xper})")

							# midindex to embed
							EMBEDDING_MODEL = "text-embedding-ada-002"
							openai.api_key = AccountGPT
							df = pd.read_csv(tarnamecsv)
							df = df.set_index(["title", "heading"])
							df.sample(1)
							def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
								result = openai.Embedding.create(
									model=model,
									input=text
								)
								time.sleep(0.5)
								return result["data"][0]["embedding"]
							df["embedding"] = df.content.apply(lambda x: get_embedding(x, EMBEDDING_MODEL))
							df.to_csv(BasePath + 'with_embeddings.csv')
							with open(BasePath + 'with_embeddings.csv', 'r', encoding='utf-8') as input_file:
								reader = csv.reader(input_file)
								# 获取 CSV 文件的标题行
								header = next(reader)
								# 获取要删除的列的索引
								column_to_delete_index = header.index('tokens')
								# 创建一个新的 CSV 文件，并写入标题行
								with open(BasePath + 'with_embeddings2.csv', 'w', newline='', encoding='utf-8') as output_file:
									writer = csv.writer(output_file)
									writer.writerow([h for h in header if h != 'tokens'])
									# 遍历 CSV 文件的每一行，并删除要删除的列
									for row in reader:
										del row[column_to_delete_index]
										writer.writerow(row)
							cf = codecs.open(BasePath + 'with_embeddings2.csv', 'r', encoding='utf-8').read()
							cf = cf.replace('[', '')
							cf = cf.replace(']', '')
							cf = cf.replace('"', '')
							cfline = cf.split('\n')
							lenline = []
							for i in range(len(cfline)):
								lenline.append(len(cfline[i].split(',')) - 3)
							lenline.sort()
							num = lenline[-1]
							listnum = []
							for r in range(num):
								listnum.append(r)
							for m in range(len(listnum)):
								listnum[m] = str(listnum[m])
							liststr = ','.join(listnum)
							del cfline[0]
							cfstr = '\n'.join(cfline)
							cfstr = 'title,heading,content,' + liststr + '\n' + cfstr
							with open(BasePath + 'with_embeddings3.csv', 'w', encoding='utf-8') as f0:
								f0.write(cfstr)
							# 读取 CSV 文件
							with open(BasePath + 'with_embeddings3.csv', 'r', encoding='utf-8') as input_file:
								reader = csv.reader(input_file)
								# 获取 CSV 文件的标题行
								header = next(reader)
								# 获取要删除的列的索引
								column_to_delete_index = header.index('content')
								# 创建一个新的 CSV 文件，并写入标题行
								with open(embedcsv, 'w', newline='', encoding='utf-8') as output_file:
									writer = csv.writer(output_file)
									writer.writerow([h for h in header if h != 'content'])
									# 遍历 CSV 文件的每一行，并删除要删除的列
									for row in reader:
										del row[column_to_delete_index]
										writer.writerow(row)

							# midindex to midembed
							shutil.copy(tarnamecsv, self.fullMidembed)
							with open(BasePath + 'todeletemidindex.txt', 'a', encoding='utf-8') as f0:
								f0.write(tarnamecsv + '\n')
							SUCC = 1
						except Exception as e:
							print(str(e) + 'part 1')
							SUCC = 0
				if SUCC == 1:
					# midembed to allindex.csv
					list_dir = os.listdir(self.fullMidembed)
					list_dir.sort()
					while '.DS_Store' in list_dir:
						list_dir.remove('.DS_Store')
					with open(self.fullall2, 'w', encoding='utf-8') as f0:
						f0.write('title,heading,content,tokens\n')
					if list_dir != []:
						for i in range(len(list_dir)):
							tarnamecsv = os.path.join(self.fullMidembed, list_dir[i])
							midembedtext = codecs.open(tarnamecsv, 'r', encoding='utf-8').read()
							midembedtext = midembedtext.replace('title,heading,content,tokens', '')
							midembedtext_list = midembedtext.split('\n')
							while '' in midembedtext_list:
								midembedtext_list.remove('')
							midembedtext = '\n'.join(midembedtext_list)
							with open(self.fullall2, 'a', encoding='utf-8') as f0:
								f0.write(midembedtext + '\n')

					# embed to allembed.csv
					list_dir = os.listdir(self.fullEmbed)
					list_dir.sort()
					while '.DS_Store' in list_dir:
						list_dir.remove('.DS_Store')
					parta = ''
					for d in range(0, 1536):
						parta = parta + str(d) + ','
					parta = parta.rstrip(',')
					with open(self.fullall1, 'w', encoding='utf-8') as f0:
						f0.write('title,heading,' + parta + '\n')
					if list_dir != []:
						for i in range(len(list_dir)):
							tarnamecsv = os.path.join(self.fullEmbed, list_dir[i])
							midembedtext = codecs.open(tarnamecsv, 'r', encoding='utf-8').read()
							midembedtext_list = midembedtext.split('\n')
							while '' in midembedtext_list:
								midembedtext_list.remove('')
							del midembedtext_list[0]
							midembedtext = '\n'.join(midembedtext_list)
							with open(self.fullall1, 'a', encoding='utf-8') as f0:
								f0.write(midembedtext + '\n')

					# modify two all.csv
					text = codecs.open(self.fullall1, 'r', encoding='utf-8').read()
					text = text.replace('\r', '').replace('\n\n', '\n')
					with open(self.fullall1, 'w', encoding='utf-8') as f0:
						f0.write(text)
					text = codecs.open(self.fullall2, 'r', encoding='utf-8').read()
					text = text.replace('\r', '').replace('\n\n', '\n')
					with open(self.fullall2, 'w', encoding='utf-8') as f0:
						f0.write(text)

					# remove
					midindexde = codecs.open(BasePath + 'todeletemidindex.txt', 'r', encoding='utf-8').read()
					midindexde_list = midindexde.split('\n')
					while '' in midindexde_list:
						midindexde_list.remove('')
					if midindexde_list != []:
						for i in range(len(midindexde_list)):
							try:
								os.remove(midindexde_list[i])
							except:
								pass
					with open(BasePath + 'todeletemidindex.txt', 'w', encoding='utf-8') as f0:
						f0.write('')

					# notify
					CMD = '''
						on run argv
						  display notification (item 2 of argv) with title (item 1 of argv)
						end run
						'''
					self.notify(CMD, "Banana: Webpage Archiver",
								f"Your embeddings are completed!\n Now you can chat with them!")
				if SUCC == 0:
					CMD = '''
						on run argv
						  display notification (item 2 of argv) with title (item 1 of argv)
						end run
						'''
					self.notify(CMD, "Banana: Webpage Archiver",
								f"Embedding failed! Please try again!")
			icon = QIcon(BasePath + "banana.icns")
			tray.setIcon(icon)
			tray.setVisible(True)

			allit_list = os.listdir(self.fullEmbed)
			while '.DS_Store' in allit_list:
				allit_list.remove('.DS_Store')
			while '' in allit_list:
				allit_list.remove('')
			allname_list = ['Context: All']
			if allit_list != []:
				for i in range(len(allit_list)):
					if '.csv' in allit_list[i]:
						allname_list.append(allit_list[i])
			self.widget0.clear()
			self.widget0.addItems(allname_list)
		if AccountGPT == '':
			CMD = '''
				on run argv
				  display notification (item 2 of argv) with title (item 1 of argv)
				end run
				'''
			self.notify(CMD, "Banana: Webpage Archiver",
						f"Your openai API is empty!\n Please enter your key in Settings!")

	def showchat(self):
		if action6.isChecked():
			self.qw9.setVisible(True)
		if not action6.isChecked():
			self.qw9.setVisible(False)

	def whichchat(self):
		tarname = self.widget0.currentText()
		with open(BasePath + "chatwith.txt", 'w', encoding='utf-8') as f0:
			f0.write(tarname)

	def searchchat(self):
		COMPLETIONS_MODEL = "gpt-3.5-turbo"
		EMBEDDING_MODEL = "text-embedding-ada-002"
		AccountGPT = codecs.open(BasePath + 'api.txt', 'r', encoding='utf-8').read()
		openai.api_key = AccountGPT
		TEMP = int(codecs.open(BasePath + 'temperature.txt', 'r', encoding='utf-8').read())
		MAXT = int(codecs.open(BasePath + 'maxtokens.txt', 'r', encoding='utf-8').read())

		if self.text1.toPlainText() != '':
			df = pd.read_csv(self.fullall2)
			chatwith = codecs.open(BasePath + "chatwith.txt", 'r', encoding='utf-8').read()
			if chatwith != 'Context: All':
				chatpath1 = os.path.join(self.fullIndex, chatwith)
				df = pd.read_csv(chatpath1)
			df = df.set_index(["title", "heading"])
			#print(f"{len(df)} rows in the data.")
			df.sample(5)

			def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
				result = openai.Embedding.create(
					model=model,
					input=text
				)
				time.sleep(0.5)
				return result["data"][0]["embedding"]

			def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
				df = pd.read_csv(fname, header=0)
				max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
				return {
					(r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
				}

			document_embeddings = load_embeddings(self.fullall1)
			if chatwith != 'Context: All':
				chatpath2 = os.path.join(self.fullEmbed, chatwith)
				document_embeddings = load_embeddings(chatpath2)

			def vector_similarity(x: list[float], y: list[float]) -> float:
				return np.dot(np.array(x), np.array(y))

			def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
				(float, (str, str))]:
				query_embedding = get_embedding(query)

				document_similarities = sorted([
					(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in
					contexts.items()
				], reverse=True)

				return document_similarities

			MAX_SECTION_LEN = 1024
			SEPARATOR = "\n* "

			tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
			separator_len = len(tokenizer.encode(SEPARATOR))

			def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
				most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

				chosen_sections = []
				chosen_sections_len = 0
				chosen_sections_indexes = []

				for _, section_index in most_relevant_document_sections:
					# Add contexts until we run out of space.
					document_section = df.loc[section_index]

					chosen_sections_len += document_section.tokens + separator_len
					if (chosen_sections_len > MAX_SECTION_LEN).any():
						break

					chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
					chosen_sections_indexes.append(str(section_index))

				header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

				return header + "".join(str(chosen_sections)) + "\n\n Q: " + question + "\n A:"

			def answer_query_with_context(
					query: str,
					df: pd.DataFrame,
					document_embeddings: dict[(str, str), np.array],
					show_prompt: bool = False
			) -> str:
				prompt = construct_prompt(
					query,
					document_embeddings,
					df
				)

				if show_prompt:
					print(prompt)

				response = openai.ChatCompletion.create(
					model=COMPLETIONS_MODEL,
					messages=[{"role": "user", "content": prompt}],
					temperature=TEMP,
					max_tokens=MAXT,
				)

				return response.choices[0].message["content"].strip('\n')

			self.LastQ = str(self.text1.toPlainText())
			if AccountGPT != '':
				QApplication.processEvents()
				QApplication.restoreOverrideCursor()
				self.text1.setReadOnly(True)
				md = '- Q: ' + self.text1.toPlainText() + '\n\n'
				with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
					f1.write(md)
				PromText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
				newhtml = self.md2html(PromText)
				self.real1.setHtml(newhtml)
				self.real1.ensureCursorVisible()  # 游标可用
				cursor = self.real1.textCursor()  # 设置游标
				pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
				cursor.setPosition(pos)  # 游标位置设置为尾部
				self.real1.setTextCursor(cursor)  # 滚动到游标位置
				signal.signal(signal.SIGALRM, self.timeout_handler)
				signal.alarm(60)
				Which = codecs.open(BasePath + 'which.txt', 'r', encoding='utf-8').read()
				if Which == '0':
					try:
						query = self.text1.toPlainText()
						message = answer_query_with_context(query, df, document_embeddings)
						message = message.lstrip('\n')
						message = message.replace('\n', '\n\n\t')
						message = message.replace('\n\n\t\n\n\t', '\n\n\t')
						message = '\n\t' + message
						EndMess = '- A: ' + message + '\n\n---\n\n'
						with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
							f1.write(EndMess)
						AllText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
						endhtml = self.md2html(AllText)
						self.real1.setHtml(endhtml)
						self.real1.ensureCursorVisible()  # 游标可用
						cursor = self.real1.textCursor()  # 设置游标
						pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
						cursor.setPosition(pos)  # 游标位置设置为尾部
						self.real1.setTextCursor(cursor)  # 滚动到游标位置
						QApplication.processEvents()
						QApplication.restoreOverrideCursor()

						self.text1.clear()
					except TimeoutException:
						with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
							f1.write('- A: Timed out, please try again!' + '\n\n---\n\n')
						AllText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
						endhtml = self.md2html(AllText)
						self.real1.setHtml(endhtml)
						self.real1.ensureCursorVisible()  # 游标可用
						cursor = self.real1.textCursor()  # 设置游标
						pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
						cursor.setPosition(pos)  # 游标位置设置为尾部
						self.real1.setTextCursor(cursor)  # 滚动到游标位置
						self.text1.setPlainText(self.LastQ)
					except Exception as e:
						with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
							f1.write('- A: Error, please try again!' + str(e) + '\n\n---\n\n')
						AllText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
						endhtml = self.md2html(AllText)
						self.real1.setHtml(endhtml)
						self.real1.ensureCursorVisible()  # 游标可用
						cursor = self.real1.textCursor()  # 设置游标
						pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
						cursor.setPosition(pos)  # 游标位置设置为尾部
						self.real1.setTextCursor(cursor)  # 滚动到游标位置
						self.text1.setPlainText(self.LastQ)
				if Which == '1':
					ENDPOINT = 'https://api.openai.com/v1/chat/completions'
					HEADERS = {"Authorization": f"Bearer {AccountGPT}"}
					async def answer_query(
							query: str,
							df: pd.DataFrame,
							document_embeddings: dict[(str, str), np.array],
							show_prompt: bool = False
					) -> str:
						prompt = construct_prompt(
							query,
							document_embeddings,
							df
						)

						if show_prompt:
							print(prompt)

						ori_history = [{"role": "user", "content": "Hey."},
									   {"role": "assistant", "content": "Hello! I'm happy to help you."}]
						conversation_history = ori_history
						try:
							response = await chat_gpt(prompt, conversation_history)
							message = response.lstrip('assistant:').strip()
							return message
						except Exception as e:
							pass

					async def chat_gpt(message, conversation_history=None, tokens_limit=4096):
						if conversation_history is None:
							conversation_history = []

						conversation_history.append({"role": "user", "content": message})

						input_text = "".join([f"{msg['role']}:{msg['content']}\n" for msg in conversation_history])

						# Truncate or shorten the input text if it exceeds the token limit
						encoded_input_text = input_text.encode("utf-8")
						while len(encoded_input_text) > tokens_limit:
							conversation_history.pop(0)
							input_text = "".join([f"{msg['role']}:{msg['content']}\n" for msg in conversation_history])
							encoded_input_text = input_text.encode("utf-8")

						# Set up the API call data
						data = {
							"model": "gpt-3.5-turbo",
							"messages": [{"role": "user", "content": input_text}],
							"max_tokens": MAXT,
							"temperature": TEMP,
							"n": 1,
							"stop": None,
						}

						# Make the API call asynchronously
						async with httpx.AsyncClient() as client:
							response = await client.post(ENDPOINT, json=data, headers=HEADERS, timeout=60.0)

						# Process the API response
						if response.status_code == 200:
							response_data = response.json()
							chat_output = response_data["choices"][0]["message"]["content"].strip()
							return chat_output
						else:
							raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
					try:
						query = self.text1.toPlainText()
						message = asyncio.run(answer_query(query, df, document_embeddings))
						message = message.lstrip('\n')
						message = message.replace('\n', '\n\n\t')
						message = message.replace('\n\n\t\n\n\t', '\n\n\t')
						message = '\n\t' + message
						EndMess = '- A: ' + message + '\n\n---\n\n'
						with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
							f1.write(EndMess)
						AllText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
						endhtml = self.md2html(AllText)
						self.real1.setHtml(endhtml)
						self.real1.ensureCursorVisible()  # 游标可用
						cursor = self.real1.textCursor()  # 设置游标
						pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
						cursor.setPosition(pos)  # 游标位置设置为尾部
						self.real1.setTextCursor(cursor)  # 滚动到游标位置
						QApplication.processEvents()
						QApplication.restoreOverrideCursor()

						self.text1.clear()
					except TimeoutException:
						with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
							f1.write('- A: Timed out, please try again!' + '\n\n---\n\n')
						AllText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
						endhtml = self.md2html(AllText)
						self.real1.setHtml(endhtml)
						self.real1.ensureCursorVisible()  # 游标可用
						cursor = self.real1.textCursor()  # 设置游标
						pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
						cursor.setPosition(pos)  # 游标位置设置为尾部
						self.real1.setTextCursor(cursor)  # 滚动到游标位置
						self.text1.setPlainText(self.LastQ)
					except Exception as e:
						with open(BasePath + 'output.txt', 'a', encoding='utf-8') as f1:
							f1.write('- A: Error, please try again!' + str(e) + '\n\n---\n\n')
						AllText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
						endhtml = self.md2html(AllText)
						self.real1.setHtml(endhtml)
						self.real1.ensureCursorVisible()  # 游标可用
						cursor = self.real1.textCursor()  # 设置游标
						pos = len(self.real1.toPlainText())  # 获取文本尾部的位置
						cursor.setPosition(pos)  # 游标位置设置为尾部
						self.real1.setTextCursor(cursor)  # 滚动到游标位置
						self.text1.setPlainText(self.LastQ)
				signal.alarm(0)  # reset timer
				self.text1.setReadOnly(False)
			if AccountGPT == '':
				CMD = '''
					on run argv
					  display notification (item 2 of argv) with title (item 1 of argv)
					end run
					'''
				self.notify(CMD, "Banana: Webpage Archiver",
							f"Your openai API is empty!\n Please enter your key in Settings!")

	def clearall(self):
		self.text1.clear()
		self.text1.setReadOnly(False)
		self.real1.clear()
		with open(BasePath + 'output.txt', 'w', encoding='utf-8') as f1:
			f1.write('')

	def exportfile(self):
		home_dir = str(Path.home())
		fj = QFileDialog.getExistingDirectory(self, 'Open', home_dir)
		if fj != '':
			ConText = codecs.open(BasePath + 'output.txt', 'r', encoding='utf-8').read()
			ISOTIMEFORMAT = '%Y%m%d %H-%M-%S-%f'
			theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
			tarname = theTime + " GPToutput.md"
			fulldir = os.path.join(fj, tarname)
			with open(fulldir, 'w', encoding='utf-8') as f1:
				f1.write(ConText)

	def searchitem(self):
		itemtext = codecs.open(self.fullse, 'r', encoding='utf-8').read()
		itemlist = itemtext.split('\n')
		while '' in itemlist:
			itemlist.remove('')
		if itemlist != [] and self.le2.text() != '':
			tarlist = []
			for i in range(len(itemlist)):
				if self.le2.text() in itemlist[i]:
					tarlist.append(itemlist[i].split('✡✡')[1])
			self.item_list.clear()
			self.item_list.addItems(tarlist)

	def addfolder(self):
		if self.le3.text() != '':
			with open(self.fullfolder, 'a', encoding='utf-8') as f0:
				f0.write(self.le3.text() + '\n')
			folders = codecs.open(self.fullfolder, 'r', encoding='utf-8').read()
			tolist = folders.split('\n')
			while '' in tolist:
				tolist.remove('')
			self.folder_list.clear()
			self.folder_list.addItems(tolist)
			tarname7_6 = self.le3.text() + '.txt'
			newfolder = os.path.join(self.fulldir1, tarname7_6)
			if not os.path.exists(newfolder):
				with open(newfolder, 'w', encoding='utf-8') as f0:
					f0.write('')
			self.le3.clear()

	def deletefolder(self):
		if self.folder_list.currentItem() != None:
			todel = self.folder_list.currentItem().text()
			if todel != 'Deleted' and todel != '':
				folders = codecs.open(self.fullfolder, 'r', encoding='utf-8').read()
				tolist = folders.split('\n')
				if tolist != []:
					# del foldername
					while '' in tolist:
						tolist.remove('')
					tolist.remove(todel)
					outlist = '\n'.join(tolist) + '\n'
					with open(self.fullfolder, 'w', encoding='utf-8') as f0:
						f0.write(outlist)
					self.folder_list.clear()
					self.folder_list.addItems(tolist)
					# move to Deleted
					tarname7_6 = todel + '.txt'
					delfolder = os.path.join(self.fulldir1, tarname7_6)
					deltext = codecs.open(delfolder, 'r', encoding='utf-8').read()
					with open(self.fulldel, 'a', encoding='utf-8') as f0:
						f0.write(deltext + '\n')
					# del folder
					os.remove(delfolder)
			if todel == 'Deleted':
				allitem = codecs.open(self.fulldel, 'r', encoding='utf-8').read()
				alllist = allitem.split('\n')
				while '' in alllist:
					alllist.remove('')
				for i in range(len(alllist)):
					# del records
					tar1 = alllist[i].split('✡✡')[1] + '.webarchive'
					del1 = os.path.join(self.fullrecord, tar1)
					try:
						os.remove(del1)
					except:
						pass
					# del local
					tar2 = alllist[i].split('✡✡')[1] + '.txt'
					del2 = os.path.join(self.fulllocal, tar2)
					try:
						os.remove(del2)
					except:
						pass
					# del index
					tar3 = alllist[i].split('✡✡')[1] + '.csv'
					del3 = os.path.join(self.fullIndex, tar3)
					try:
						os.remove(del3)
					except:
						pass
					# del embed
					tar4 = alllist[i].split('✡✡')[1] + '.csv'
					del4 = os.path.join(self.fullEmbed, tar4)
					try:
						os.remove(del4)
					except:
						pass
					# del midindex
					tar5 = alllist[i].split('✡✡')[1] + '.csv'
					del5 = os.path.join(self.fullMidindex, tar5)
					try:
						os.remove(del5)
					except:
						pass
					# del midembed
					tar6 = alllist[i].split('✡✡')[1] + '.csv'
					del6 = os.path.join(self.fullMidembed, tar6)
					try:
						os.remove(del6)
					except:
						pass
					# del in allsearch.txt
					searit = codecs.open(self.fullse, 'r', encoding='utf-8').read()
					searit_list = searit.split('\n')
					while '' in searit_list:
						searit_list.remove('')
					if searit_list != []:
						emptylist = []
						for m in range(len(searit_list)):
							if alllist[i] in searit_list[m]:
								emptylist.append(searit_list[m])
						putlist = list(set(searit_list) - set(emptylist))
						putstr = '\n'.join(putlist) + '\n'
						with open(self.fullse, 'w', encoding='utf-8') as f0:
							f0.write(putstr)
				# redo allindex
				list_dir = os.listdir(self.fullMidembed)
				list_dir.sort()
				while '.DS_Store' in list_dir:
					list_dir.remove('.DS_Store')
				with open(self.fullall2, 'w', encoding='utf-8') as f0:
					f0.write('title,heading,content,tokens\n')
				if list_dir != []:
					for i in range(len(list_dir)):
						tarnamecsv = os.path.join(self.fullMidembed, list_dir[i])
						midembedtext = codecs.open(tarnamecsv, 'r', encoding='utf-8').read()
						midembedtext = midembedtext.replace('title,heading,content,tokens', '')
						midembedtext_list = midembedtext.split('\n')
						while '' in midembedtext_list:
							midembedtext_list.remove('')
						midembedtext = '\n'.join(midembedtext_list)
						with open(self.fullall2, 'a', encoding='utf-8') as f0:
							f0.write(midembedtext + '\n')
				# redo allembed
				list_dir = os.listdir(self.fullEmbed)
				list_dir.sort()
				while '.DS_Store' in list_dir:
					list_dir.remove('.DS_Store')
				parta = ''
				for d in range(0, 1536):
					parta = parta + str(d) + ','
				parta = parta.rstrip(',')
				with open(self.fullall1, 'w', encoding='utf-8') as f0:
					f0.write('title,heading,' + parta + '\n')
				if list_dir != []:
					for i in range(len(list_dir)):
						tarnamecsv = os.path.join(self.fullEmbed, list_dir[i])
						midembedtext = codecs.open(tarnamecsv, 'r', encoding='utf-8').read()
						midembedtext_list = midembedtext.split('\n')
						while '' in midembedtext_list:
							midembedtext_list.remove('')
						del midembedtext_list[0]
						midembedtext = '\n'.join(midembedtext_list)
						with open(self.fullall1, 'a', encoding='utf-8') as f0:
							f0.write(midembedtext + '\n')
				# clear deleted.txt item
				with open(self.fulldel, 'w', encoding='utf-8') as f0:
					f0.write('')
		# refresh display
		if self.folder_list.currentItem() != None:
			endname = self.folder_list.currentItem().text() + '.txt'
			tarfile = os.path.join(self.fulldir1, endname)
			items = codecs.open(tarfile, 'r', encoding='utf-8').read()
			item_list = items.split('\n')
			while '' in item_list:
				item_list.remove('')
			showlist = []
			for i in range(len(item_list)):
				showlist.append(item_list[i].split('✡✡')[1])
			self.item_list.clear()
			self.item_list.addItems(showlist)

	def showdelbutton(self):
		if action8.isChecked():
			self.btn3.setVisible(True)
			self.btn7.setVisible(True)
		if not action8.isChecked():
			self.btn3.setVisible(False)
			self.btn7.setVisible(False)

	def openlink(self):
		if self.item_list.currentItem() != None:
			tarit = self.item_list.currentItem().text()
			allitems = codecs.open(self.fullse, 'r', encoding='utf-8').read()
			allitems_list = allitems.split('\n')
			while '' in allitems_list:
				allitems_list.remove('')
			if allitems_list != []:
				find_item = []
				for i in range(len(allitems_list)):
					if tarit in allitems_list[i]:
						find_item.append(allitems_list[i])
				find_url = str(find_item[0].split('✡✡')[0])
				webbrowser.open(find_url)

	def copylink(self):
		if self.item_list.currentItem() != None:
			tarit = self.item_list.currentItem().text()
			allitems = codecs.open(self.fullse, 'r', encoding='utf-8').read()
			allitems_list = allitems.split('\n')
			while '' in allitems_list:
				allitems_list.remove('')
			if allitems_list != []:
				find_item = []
				for i in range(len(allitems_list)):
					if tarit in allitems_list[i]:
						find_item.append(allitems_list[i])
				find_url = str(find_item[0].split('✡✡')[0])
				pyperclip.copy(find_url)

	def openarchive(self):
		if self.item_list.currentItem() != None:
			tarit = self.item_list.currentItem().text() + '.webarchive'
			topath = os.path.join(self.fullrecord, tarit)
			subprocess.run(["open", "-a", "Safari", topath])

	def deleteitem(self):
		if self.folder_list.currentItem() != None and self.item_list.currentItem() != None:
			todel = self.folder_list.currentItem().text()
			tarit = self.item_list.currentItem().text()
			if todel != 'Deleted' and todel != '' and tarit != '':
				tarname7_6 = todel + '.txt'
				delfolder = os.path.join(self.fulldir1, tarname7_6)
				deltext = codecs.open(delfolder, 'r', encoding='utf-8').read()
				deltext_list = deltext.split('\n')
				while '' in deltext_list:
					deltext_list.remove('')
				if deltext_list != []:
					newlist = []
					for i in range(len(deltext_list)):
						if tarit in deltext_list[i]:
							newlist.append(deltext_list[i])
					endlist = list(set(deltext_list) - set(newlist))
					endtext = '\n'.join(endlist)
					with open(delfolder, 'w', encoding='utf-8') as f0:
						f0.write(endtext + '\n')
					todel = ''.join(newlist)
					with open(self.fulldel, 'a', encoding='utf-8') as f0:
						f0.write(todel + '\n')
		# refresh display
		if self.folder_list.currentItem() != None:
			endname = self.folder_list.currentItem().text() + '.txt'
			tarfile = os.path.join(self.fulldir1, endname)
			items = codecs.open(tarfile, 'r', encoding='utf-8').read()
			item_list = items.split('\n')
			while '' in item_list:
				item_list.remove('')
			showlist = []
			for i in range(len(item_list)):
				showlist.append(item_list[i].split('✡✡')[1])
			self.item_list.clear()
			self.item_list.addItems(showlist)

	def moveto(self):
		move = CustomDialog_move()
		move.exec()
		textc = codecs.open(BasePath + 'choose.txt', 'r', encoding='utf-8').read()
		if textc == '1':
			if self.folder_list.currentItem() != None and self.item_list.currentItem() != None:
				todel = self.folder_list.currentItem().text()
				tarit = self.item_list.currentItem().text()
				if todel != '' and tarit != '':
					tarname7_6 = todel + '.txt'
					delfolder = os.path.join(self.fulldir1, tarname7_6)
					deltext = codecs.open(delfolder, 'r', encoding='utf-8').read()
					deltext_list = deltext.split('\n')
					while '' in deltext_list:
						deltext_list.remove('')
					if deltext_list != []:
						newlist = []
						for i in range(len(deltext_list)):
							if tarit in deltext_list[i]:
								newlist.append(deltext_list[i])
						endlist = list(set(deltext_list) - set(newlist))
						endtext = '\n'.join(endlist)
						with open(delfolder, 'w', encoding='utf-8') as f0:
							f0.write(endtext + '\n')
						tofolder = codecs.open(BasePath + 'tarfolder.txt', 'r', encoding='utf-8').read()
						tarnamenew = tofolder + '.txt'
						topath = os.path.join(self.fulldir1, tarnamenew)
						todel = ''.join(newlist)
						with open(topath, 'a', encoding='utf-8') as f0:
							f0.write(todel + '\n')
		# refresh display
		if self.folder_list.currentItem() != None:
			endname = self.folder_list.currentItem().text() + '.txt'
			tarfile = os.path.join(self.fulldir1, endname)
			items = codecs.open(tarfile, 'r', encoding='utf-8').read()
			item_list = items.split('\n')
			while '' in item_list:
				item_list.remove('')
			showlist = []
			for i in range(len(item_list)):
				showlist.append(item_list[i].split('✡✡')[1])
			self.item_list.clear()
			self.item_list.addItems(showlist)

	def notify(self, CMD, title, text):
		subprocess.call(['osascript', '-e', CMD, title, text])

	def timeout_handler(self, signum, frame):
		raise TimeoutException("Timeout")

	def md2html(self, mdstr):
		extras = ['code-friendly', 'fenced-code-blocks', 'footnotes', 'tables', 'code-color', 'pyshell', 'nofollow',
				  'cuddled-lists', 'header ids', 'nofollow']

		html = """
		<html>
		<head>
		<meta content="text/html; charset=utf-8" http-equiv="content-type" />
		<style>
			.hll { background-color: #ffffcc }
			.c { color: #0099FF; font-style: italic } /* Comment */
			.err { color: #AA0000; background-color: #FFAAAA } /* Error */
			.k { color: #006699; font-weight: bold } /* Keyword */
			.o { color: #555555 } /* Operator */
			.ch { color: #0099FF; font-style: italic } /* Comment.Hashbang */
			.cm { color: #0099FF; font-style: italic } /* Comment.Multiline */
			.cp { color: #009999 } /* Comment.Preproc */
			.cpf { color: #0099FF; font-style: italic } /* Comment.PreprocFile */
			.c1 { color: #0099FF; font-style: italic } /* Comment.Single */
			.cs { color: #0099FF; font-weight: bold; font-style: italic } /* Comment.Special */
			.gd { background-color: #FFCCCC; border: 1px solid #CC0000 } /* Generic.Deleted */
			.ge { font-style: italic } /* Generic.Emph */
			.gr { color: #FF0000 } /* Generic.Error */
			.gh { color: #003300; font-weight: bold } /* Generic.Heading */
			.gi { background-color: #CCFFCC; border: 1px solid #00CC00 } /* Generic.Inserted */
			.go { color: #AAAAAA } /* Generic.Output */
			.gp { color: #000099; font-weight: bold } /* Generic.Prompt */
			.gs { font-weight: bold } /* Generic.Strong */
			.gu { color: #003300; font-weight: bold } /* Generic.Subheading */
			.gt { color: #99CC66 } /* Generic.Traceback */
			.kc { color: #006699; font-weight: bold } /* Keyword.Constant */
			.kd { color: #006699; font-weight: bold } /* Keyword.Declaration */
			.kn { color: #006699; font-weight: bold } /* Keyword.Namespace */
			.kp { color: #006699 } /* Keyword.Pseudo */
			.kr { color: #006699; font-weight: bold } /* Keyword.Reserved */
			.kt { color: #007788; font-weight: bold } /* Keyword.Type */
			.m { color: #FF6600 } /* Literal.Number */
			.s { color: #CC3300 } /* Literal.String */
			.na { color: #330099 } /* Name.Attribute */
			.nb { color: #336666 } /* Name.Builtin */
			.nc { color: #00AA88; font-weight: bold } /* Name.Class */
			.no { color: #336600 } /* Name.Constant */
			.nd { color: #9999FF } /* Name.Decorator */
			.ni { color: #999999; font-weight: bold } /* Name.Entity */
			.ne { color: #CC0000; font-weight: bold } /* Name.Exception */
			.nf { color: #CC00FF } /* Name.Function */
			.nl { color: #9999FF } /* Name.Label */
			.nn { color: #00CCFF; font-weight: bold } /* Name.Namespace */
			.nt { color: #330099; font-weight: bold } /* Name.Tag */
			.nv { color: #003333 } /* Name.Variable */
			.ow { color: #000000; font-weight: bold } /* Operator.Word */
			.w { color: #bbbbbb } /* Text.Whitespace */
			.mb { color: #FF6600 } /* Literal.Number.Bin */
			.mf { color: #FF6600 } /* Literal.Number.Float */
			.mh { color: #FF6600 } /* Literal.Number.Hex */
			.mi { color: #FF6600 } /* Literal.Number.Integer */
			.mo { color: #FF6600 } /* Literal.Number.Oct */
			.sa { color: #CC3300 } /* Literal.String.Affix */
			.sb { color: #CC3300 } /* Literal.String.Backtick */
			.sc { color: #CC3300 } /* Literal.String.Char */
			.dl { color: #CC3300 } /* Literal.String.Delimiter */
			.sd { color: #CC3300; font-style: italic } /* Literal.String.Doc */
			.s2 { color: #CC3300 } /* Literal.String.Double */
			.se { color: #CC3300; font-weight: bold } /* Literal.String.Escape */
			.sh { color: #CC3300 } /* Literal.String.Heredoc */
			.si { color: #AA0000 } /* Literal.String.Interpol */
			.sx { color: #CC3300 } /* Literal.String.Other */
			.sr { color: #33AAAA } /* Literal.String.Regex */
			.s1 { color: #CC3300 } /* Literal.String.Single */
			.ss { color: #FFCC33 } /* Literal.String.Symbol */
			.bp { color: #336666 } /* Name.Builtin.Pseudo */
			.fm { color: #CC00FF } /* Name.Function.Magic */
			.vc { color: #003333 } /* Name.Variable.Class */
			.vg { color: #003333 } /* Name.Variable.Global */
			.vi { color: #003333 } /* Name.Variable.Instance */
			.vm { color: #003333 } /* Name.Variable.Magic */
			.il { color: #FF6600 } /* Literal.Number.Integer.Long */
			table {
					font-family: verdana,arial,sans-serif;
					font-size:11px;
					color:#333333;
					border-width: 1px;
					border-color: #999999;
					border-collapse: collapse;
					}
			th {
				background:#b5cfd2 url('cell-blue.jpg');
				border-width: 1px;
				padding: 8px;
				border-style: solid;
				border-color: #999999;
				}
			td {
				background:#dcddc0 url('cell-grey.jpg');
				border-width: 1px;
				padding: 8px;
				border-style: solid;
				border-color: #999999;
				}
		</style>
		</head>
		<body>
			%s
		</body>
		</html>
		"""
		ret = markdown2.markdown(mdstr, extras=extras)
		return html % ret

	def cleanlinebreak(self, a):  # 设置清除断行的基本代码块
		for i in range(10):
			a = a.replace('\r', ' ')
			a = a.replace('\n', ' ')
		a = a.replace('   ', ' ')
		a = a.replace('  ', ' ')
		return a

	def default_clean(self, a):  # 最基本功能块
		# 【共同块】不管是全中文/全英文/中英混排，都需要清除的不规范的符号与排版
		# 清除文档排版符号
		a = a.replace('\t', '')

		# 清除连续空格（如连续两个和三个空格）
		for i in range(10):
			a = a.replace('   ', ' ')
			a = a.replace('  ', ' ')
			a = a.replace('，，，', '，')
			a = a.replace('，，', '，')
			a = a.replace(',,,', ',')
			a = a.replace(',,', ',')

		# 清除那些引用标记（括号内为纯数字），如圈圈数字和方括号引用，同时由于方括号和六角括号混用较多，清理前后不搭的情况中的引用符号
		a = re.sub(r"\{(\s)*(\d+\s)*(\d)*?\}|\[(\s)*(\d+\s)*(\d)*?\]|〔(\s)*(\d+\s)*(\d)*?〕|﹝(\s)*(\d+\s)*(\d)*?﹞", "", a)
		a = re.sub(r"\[(\s)*(\d+\s)*(\d)*?〕|\[(\s)*(\d+\s)*(\d)*?﹞|〔(\s)*(\d+\s)*(\d)*?\]|〔(\s)*(\d+\s)*(\d)*?﹞|﹝(\s)*(\d+\s)*(\d)*?\]|﹝(\s)*(\d+\s)*(\d)*?〕", "", a)
		a = re.sub(r"（(\s)*(\d+\s)*(\d)*?）|\[(\s)*(\d+\s)*(\d)*?）|（(\s)*(\d+\s)*(\d)*?\]|（(\s)*(\d+\s)*(\d)*?】|【(\s)*(\d+\s)*(\d)*?）", "", a)
		a = re.sub(r"\((\s)*(\d+\s)*(\d)*?〕|\((\s)*(\d+\s)*(\d)*?﹞|〔(\s)*(\d+\s)*(\d)*?\)|﹝(\s)*(\d+\s)*(\d)*?\)|\((\s)*(\d+\s)*(\d)*?\)|\[(\s)*(\d+\s)*(\d)*?\)|\((\s)*(\d+\s)*(\d)*?\]", "", a)
		a = re.sub(u'\u24EA|[\u2460-\u2473]|[\u3251-\u325F]|[\u32B1-\u32BF]|[\u2776-\u277F]|\u24FF|[\u24EB-\u24F4]',
				   "", a)
		a = re.sub(r"\<(\s)*(\d+\s)*(\d)*?\>|\《(\s)*(\d+\s)*(\d)*?\》|\〈(\s)*(\d+\s)*(\d)*?\〉|\＜(\s)*(\d+\s)*(\d)*?\＞", "", a)
		a = re.sub(r"\<(\s)*(\d+\s)*(\d)*?\》|\<(\s)*(\d+\s)*(\d)*?\〉|\<(\s)*(\d+\s)*(\d)*?\＞",
				   "", a)
		a = re.sub(r"\《(\s)*(\d+\s)*(\d)*?\>|\《(\s)*(\d+\s)*(\d)*?\〉|\《(\s)*(\d+\s)*(\d)*?\＞",
				   "", a)
		a = re.sub(r"\〈(\s)*(\d+\s)*(\d)*?\>|\〈(\s)*(\d+\s)*(\d)*?\》|\〈(\s)*(\d+\s)*(\d)*?\＞",
				   "", a)
		a = re.sub(r"\＜(\s)*(\d+\s)*(\d)*?\>|\＜(\s)*(\d+\s)*(\d)*?\》|\＜(\s)*(\d+\s)*(\d)*?\〉",
				   "", a)
		a = a.replace('◎', '')
		a = a.replace('®', '')
		a = a.replace('*', '')

		# 错误标点纠正：将奇怪的弯引号换为正常的弯引号，为下面执行弯引号与直引号的清除提供条件
		a = a.replace('〞', '”')
		a = a.replace('〝', '“')

		# 错误标点纠正：将角分符号（′）替换为弯引号（若需要使用角分符号则不运行此条）
		a = a.replace('′', "’")
		# 错误标点纠正：将角秒符号（″）替换为弯引号（若需要使用角秒符号则不运行此条）
		a = a.replace('″', '”')

		# 错误标点纠正1（两个同向单引号变成一个双引号<前>，改为前后弯双引号）
		pattern = re.compile(r'‘‘(.*?)”')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('‘‘{}”'.format(i), '“{}”'.format(i))

		# 错误标点纠正2（两个同向单引号变成一个双引号<后>，改为前后弯双引号）
		p1 = r"(?<=“).+?(?=’’)"
		pattern1 = re.compile(p1)
		result = pattern1.findall(a)
		for i in result:
			a = a.replace('“{}’’'.format(i), '“{}”'.format(i))

		# 错误标点纠正3（前后两个单引号变成一组双引号）
		pattern = re.compile(r'‘‘(.*?)’’')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('‘‘{}’’'.format(i), '“{}”'.format(i))

		# 错误标点纠正4（两个同向双引号去掉一个<前>）
		pattern = re.compile(r'““(.*?)”')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('““{}”'.format(i), '“{}”'.format(i))

		# 错误标点纠正5（两个同向双引号去掉一个<后>）
		p1 = r"(?<=“).+?(?=””)"
		pattern1 = re.compile(p1)
		result = pattern1.findall(a)
		for i in result:
			a = a.replace('“{}””'.format(i), '“{}”'.format(i))

		# 错误标点纠正6（两组双引号变成一组双引号）
		pattern = re.compile(r'““(.*?)””')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('““{}””'.format(i), '“{}”'.format(i))

		# 错误标点纠正7（前直单引号<前>，后弯双引号<后>，改为前后弯双引号）
		pattern = re.compile(r"'(.*?)”")
		result = pattern.findall(a)
		for i in result:
			a = a.replace("'{}”".format(i), '“{}”'.format(i))

		# 错误标点纠正8（前直双引号<前>，后弯双引号<后>，改为前后弯双引号）
		pattern = re.compile(r'"(.*?)”')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('"{}”'.format(i), '“{}”'.format(i))

		# 错误标点纠正9（前弯双引号<前>，后直单引号<后>，改为前后弯双引号）
		p1 = r"(?<=“).+?(?=')"
		pattern1 = re.compile(p1)
		result = pattern1.findall(a)
		for i in result:
			a = a.replace("“{}'".format(i), '“{}”'.format(i))

		# 错误标点纠正10（前弯双引号<前>，后直双引号<后>，改为前后弯双引号）
		p1 = r'(?<=“).+?(?=")'
		pattern1 = re.compile(p1)
		result = pattern1.findall(a)
		for i in result:
			a = a.replace('“{}"'.format(i), '“{}”'.format(i))

		# 将成对的直双引号改为成对的弯双引号
		pattern = re.compile(r'"(.*?)"')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('"{}"'.format(i), '“{}”'.format(i))

		# 将成对的直单引号改为成对的弯单引号
		pattern = re.compile(r"'(.*?)'")
		result = pattern.findall(a)
		for i in result:
			a = a.replace("'{}'".format(i), "‘{}’".format(i))

		# 对文段进行再次多余部分的清洗
		# 错误标点纠正1（两个同向单引号变成一个双引号<前>，改为前后弯双引号）
		pattern = re.compile(r'‘‘(.*?)”')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('‘‘{}”'.format(i), '“{}”'.format(i))

		# 错误标点纠正2（两个同向单引号变成一个双引号<后>，改为前后弯双引号）
		p1 = r"(?<=“).+?(?=’’)"
		pattern1 = re.compile(p1)
		result = pattern1.findall(a)
		for i in result:
			a = a.replace('“{}’’'.format(i), '“{}”'.format(i))

		# 错误标点纠正3（前后两个单引号变成一组双引号）
		pattern = re.compile(r'‘‘(.*?)’’')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('‘‘{}’’'.format(i), '“{}”'.format(i))

		# 错误标点纠正4（两个同向双引号去掉一个<前>）
		pattern = re.compile(r'““(.*?)”')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('““{}”'.format(i), '“{}”'.format(i))

		# 错误标点纠正5（两个同向双引号去掉一个<后>）
		p1 = r"(?<=“).+?(?=””)"
		pattern1 = re.compile(p1)
		result = pattern1.findall(a)
		for i in result:
			a = a.replace('“{}””'.format(i), '“{}”'.format(i))

		# 错误标点纠正6（两组双引号变成一组双引号）
		pattern = re.compile(r'““(.*?)””')
		result = pattern.findall(a)
		for i in result:
			a = a.replace('““{}””'.format(i), '“{}”'.format(i))

		# 将单独的单双直引号替换为空(清除剩余的直引号)
		a = a.replace("'", '')
		a = a.replace('"', '')

		# 【判断块】判断文段是全中文、全英文还是中英混排。
		def containenglish(str0):  # 判断是否包含英文字母
			import re
			return bool(re.search('[a-zA-Zａ-ｚＡ-Ｚ]', str0))

		def is_contain_chinese(check_str):  # 判断是否包含中文字
			for ch in check_str:
				if u'\u4e00' <= ch <= u'\u9fff':
					return True
			return False

		def is_contain_num(str0):  # 判断是否包含数字
			import re
			return bool(re.search('[0-9０-９]', str0))

		def is_contain_symbol(keyword):
			if re.search(r"\W", keyword):
				return True
			else:
				return False

		if is_contain_num(str(a)) and not containenglish(str(a)) and not is_contain_chinese(str(a)):
			# 【全数块】清除数字中的空格，将全角数字转为半角数字
			a = a.replace(' ', '')

			def is_Qnumber(uchar):
				"""判断一个unicode是否是全角数字"""
				if uchar >= u'\uff10' and uchar <= u'\uff19':
					return True
				else:
					return False

			def Q2B(uchar):
				"""单个字符 全角转半角"""
				inside_code = ord(uchar)
				if inside_code == 0x3000:
					inside_code = 0x0020
				else:
					inside_code -= 0xfee0
				if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
					return uchar
				return chr(inside_code)

			def stringpartQ2B(ustring):
				"""把字符串中数字全角转半角"""
				return "".join(
					[Q2B(uchar) if is_Qnumber(uchar) else uchar for uchar in ustring])

			a = stringpartQ2B(a)

			# 对全数字文段的货币符号、百分号和度数这三个符号进行专门处理
			i = 0
			while i <= len(a) - 1:
				if a[i] == '¥' and not is_contain_symbol(str(a[i - 1])):
					a = list(a)
					a.insert(i, ' ')
					a = ''.join(a)
					i = i + 2
					continue
				if a[i] == '$' and not is_contain_symbol(str(a[i - 1])):
					a = list(a)
					a.insert(i, ' ')
					a = ''.join(a)
					i = i + 2
					continue
				if a[i] == "%":
					if a[i - 1] == ' ':
						a = list(a)
						del a[i - 1]
						a = ''.join(a)
						i = i - 1
						continue
					else:
						a = list(a)
						a.insert(i + 1, ' ')
						a = ''.join(a)
						i = i + 2
						continue
				if a[i] == "°":
					if a[i - 1] == ' ':
						a = list(a)
						del a[i - 1]
						a = ''.join(a)
						i = i - 1
						continue
					else:
						a = list(a)
						a.insert(i + 1, ' ')
						a = ''.join(a)
						i = i + 2
						continue
				else:
					i = i + 1
					continue

			a = a.replace('  ', ' ')
			return a

		elif not containenglish(str(a)) and is_contain_chinese(str(a)):
			# 【中（数）块】
			# 去除不必要的中英文符号及空格
			a = a.replace('*', '')
			a = a.replace(' ', '')
			a = a.replace('#', '')
			a = a.replace('^', '')
			a = a.replace('~', '')
			a = a.replace('～', '')

			# 修改一些排版中常见的符号错误
			a = a.replace('。。', '。')
			a = a.replace('。。。', '……')
			a = a.replace('—', "——")
			a = a.replace('一一', "——")
			# Black Circle, Katakana Middle Dot, Bullet, Bullet Operator 替换为标准中间点（U+00B7 MIDDLE DOT）
			a = a.replace('●', "·")
			a = a.replace('・', "·")
			a = a.replace('•', "·")
			a = a.replace('∙', "·")
			# U+2027 HYPHENATION POINT 替换为中间点（U+00B7 MIDDLE DOT）
			a = a.replace('‧', "·")
			# 加重符号、乘号、点号替换为中间点（U+00B7 MIDDLE DOT）【如果使用乘号，应使用叉号乘，慎用点乘】
			a = a.replace('•', "·")
			a = a.replace('·', "·")
			a = a.replace('▪', "·")
			# Phoenician Word Separator (U+1091F) to middle dot
			a = a.replace('𐤟', "·")
			for i in range(10):
				a = a.replace('————————', "——")
				a = a.replace('——————', "——")
				a = a.replace('————', "——")

			# 将中文和数字混排中的全角数字转为半角数字，不改变标点的全半角情况
			def is_Qnumber(uchar):
				"""判断一个unicode是否是全角数字"""
				if uchar >= u'\uff10' and uchar <= u'\uff19':
					return True
				else:
					return False

			def Q2B(uchar):
				"""单个字符 全角转半角"""
				inside_code = ord(uchar)
				if inside_code == 0x3000:
					inside_code = 0x0020
				else:
					inside_code -= 0xfee0
				if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
					return uchar
				return chr(inside_code)

			def stringpartQ2B(ustring):
				"""把字符串中数字全角转半角"""
				return "".join(
					[Q2B(uchar) if is_Qnumber(uchar) else uchar for uchar in ustring])

			a = stringpartQ2B(a)

			# 给中文和数字的混排增加空格
			def find_this(q, i):
				result = q[i]
				return result

			def find_next(q, i):
				result = q[i + 1]
				return result

			i = 0
			while i >= 0 and i < len(a) - 1:
				if is_contain_chinese(str(find_this(a, i))) and is_contain_num(str(find_next(a, i))):  # 从中文转数字
					a = list(a)
					a.insert(i + 1, ' ')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(a, i))) and is_contain_num(str(find_this(a, i))):  # 从数字转中文
					a = list(a)
					a.insert(i + 1, ' ')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 将常用英文标点转换为中文标点
			def E_trans_to_C(string):
				E_pun = u',.;:!?[]()<>'
				C_pun = u'，。；：！？【】（）《》'
				table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
				return string.translate(table)

			a = E_trans_to_C(str(a))

			# 对特殊数字符号进行处理
			i = 0
			while i <= len(a) - 1:
				if a[i] == '¥' and not is_contain_symbol(str(a[i - 1])):
					a = list(a)
					a.insert(i, ' ')
					a = ''.join(a)
					i = i + 2
					continue
				if a[i] == '$' and not is_contain_symbol(str(a[i - 1])):
					a = list(a)
					a.insert(i, ' ')
					a = ''.join(a)
					i = i + 2
					continue
				if a[i] == "%":
					if a[i - 1] == ' ':
						a = list(a)
						del a[i - 1]
						a = ''.join(a)
						i = i - 1
						continue
					else:
						a = list(a)
						a.insert(i + 1, ' ')
						a = ''.join(a)
						i = i + 2
						continue
				if a[i] == "°":
					if a[i - 1] == ' ':
						a = list(a)
						del a[i - 1]
						a = ''.join(a)
						i = i - 1
						continue
					else:
						a = list(a)
						a.insert(i + 1, ' ')
						a = ''.join(a)
						i = i + 2
						continue
				else:
					i = i + 1
					continue

			a = a.replace('  ', ' ')
			return a

		elif containenglish(str(a)) and not is_contain_chinese(str(a)):
			# 【英（数）块】给英文和数字混排的情况增加空格
			def find_this(q, i):
				result = q[i]
				return result

			def find_next(q, i):
				result = q[i + 1]
				return result

			i = 0
			while i >= 0 and i < len(a) - 1:
				if is_contain_num(str(find_this(a, i))) and containenglish(str(find_next(a, i))):  # 从数字转英文
					a = list(a)
					a.insert(i + 1, ' ')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_next(a, i))) and containenglish(str(find_this(a, i))):  # 从英文转数字
					a = list(a)
					a.insert(i + 1, ' ')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 将全角英文字符和数字转为半角英文和半角数字
			def is_Qnumber(uchar):
				"""判断一个unicode是否是全角数字"""
				if uchar >= u'\uff10' and uchar <= u'\uff19':
					return True
				else:
					return False

			def is_Qalphabet(uchar):
				"""判断一个unicode是否是全角英文字母"""
				if (uchar >= u'\uff21' and uchar <= u'\uff3a') or (uchar >= u'\uff41' and uchar <= u'\uff5a'):
					return True
				else:
					return False

			def Q2B(uchar):
				"""单个字符 全角转半角"""
				inside_code = ord(uchar)
				if inside_code == 0x3000:
					inside_code = 0x0020
				else:
					inside_code -= 0xfee0
				if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
					return uchar
				return chr(inside_code)

			def stringpartQ2B(ustring):
				"""把字符串中字母和数字全角转半角"""
				return "".join(
					[Q2B(uchar) if is_Qnumber(uchar) or is_Qalphabet(uchar) else uchar for uchar in ustring])

			a = stringpartQ2B(a)

			# 将文段中的中文符号转换为英文符号
			def C_trans_to_E(string):
				E_pun = u',.;:!?[]()<>'
				C_pun = u'，。；：！？【】（）《》'
				table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
				return string.translate(table)

			a = C_trans_to_E(str(a))

			# One Dot Leader (U+2024) to full stop (U+002E) （句号）
			a = a.replace('․', ".")

			# 清除英文标点符号前面的空格（,.;:?!）
			a = list(a)
			i = 0
			while i >= 0 and i < len(a) - 1:
				if a[i] == ',':
					if a[i - 1] == ' ':
						del a[i - 1]
						continue
					else:
						i = i + 1
						continue
				if a[i] == '.':
					if a[i - 1] == ' ':
						del a[i - 1]
						continue
					else:
						i = i + 1
						continue
				if a[i] == ';':
					if a[i - 1] == ' ':
						del a[i - 1]
						continue
					else:
						i = i + 1
						continue
				if a[i] == ':':
					if a[i - 1] == ' ':
						del a[i - 1]
						continue
					else:
						i = i + 1
						continue
				if a[i] == '?':
					if a[i - 1] == ' ':
						del a[i - 1]
						continue
					else:
						i = i + 1
						continue
				if a[i] == '!':
					if a[i - 1] == ' ':
						del a[i - 1]
						continue
					else:
						i = i + 1
						continue
				else:
					i = i + 1
					continue
			a = ''.join(a)

			# 对全数字文段的货币符号、百分号和度数这三个符号进行专门处理
			i = 0
			while i <= len(a) - 1:
				if a[i] == '¥' and not is_contain_symbol(str(a[i - 1])):
					a = list(a)
					a.insert(i, ' ')
					a = ''.join(a)
					i = i + 2
					continue
				if a[i] == '$' and not is_contain_symbol(str(a[i - 1])):
					a = list(a)
					a.insert(i, ' ')
					a = ''.join(a)
					i = i + 2
					continue
				if a[i] == "%":
					if a[i - 1] == ' ':
						a = list(a)
						del a[i - 1]
						a = ''.join(a)
						i = i - 1
						continue
					else:
						a = list(a)
						a.insert(i + 1, ' ')
						a = ''.join(a)
						i = i + 2
						continue
				if a[i] == "°":
					if a[i - 1] == ' ':
						a = list(a)
						del a[i - 1]
						a = ''.join(a)
						i = i - 1
						continue
					else:
						a = list(a)
						a.insert(i + 1, ' ')
						a = ''.join(a)
						i = i + 2
						continue
				else:
					i = i + 1
					continue

			a = a.replace('  ', ' ')
			return a

		elif containenglish(str(a)) and is_contain_chinese(str(a)) or \
				containenglish(str(a)) and is_contain_chinese(str(a)) and is_contain_num(str(a)):
			# 【中英（数）混排块】识别中英文字符，对英文字符保留空格，对中文字符去掉空格。标点默认使用原文标点，以中文为主（默认使用情况为在中文中引用英文）。
			def find_this(q, i):
				result = q[i]
				return result

			def find_pre(q, i):
				result = q[i - 1]
				return result

			def find_next(q, i):
				result = q[i + 1]
				return result

			def find_pre2(q, i):
				result = q[i - 2]
				return result

			def find_next2(q, i):
				result = q[i + 2]
				return result

			def find_next3(q, i):
				result = q[i + 3]
				return result

			# 首先来一遍此一后一的精准筛查
			i = 0
			while i >= 0 and i < len(a) - 1:
				if is_contain_chinese(str(find_this(a, i))) and containenglish(str(find_next(a, i))):  # 从中文转英文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_this(a, i))) and is_contain_num(str(find_next(a, i))):  # 从中文转数字
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(a, i))) and is_contain_num(str(find_this(a, i))):  # 从数字转中文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_this(a, i))) and containenglish(str(find_next(a, i))):  # 从数字转英文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_next(a, i))) and containenglish(str(find_this(a, i))):  # 从英文转数字
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(a, i))) and containenglish(str(find_this(a, i))):  # 从英文转中文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 再进行前一后一的插入
			i = 1
			while i > 0 and i < len(a) - 1:
				if is_contain_chinese(str(find_pre(a, i))) and containenglish(str(find_next(a, i))):  # 从中文转英文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_pre(a, i))) and is_contain_num(str(find_next(a, i))):  # 从中文转数字
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(a, i))) and is_contain_num(str(find_pre(a, i))):  # 从数字转中文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_pre(a, i))) and containenglish(str(find_next(a, i))):  # 从数字转英文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_next(a, i))) and containenglish(str(find_pre(a, i))):  # 从英文转数字
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(a, i))) and containenglish(str(find_pre(a, i))):  # 从英文转中文
					a = list(a)
					a.insert(i + 1, '*')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 进行前一后二的筛查
			i = 1
			while i > 0 and i < len(a) - 2:
				if is_contain_chinese(str(find_pre(a, i))) and containenglish(str(find_next2(a, i))):  # 从中文转英文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_pre(a, i))) and is_contain_num(str(find_next2(a, i))):  # 从中文转数字
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next2(a, i))) and is_contain_num(str(find_pre(a, i))):  # 从数字转中文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_pre(a, i))) and containenglish(str(find_next2(a, i))):  # 从数字转英文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_next2(a, i))) and containenglish(str(find_pre(a, i))):  # 从英文转数字
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next2(a, i))) and containenglish(str(find_pre(a, i))):  # 从英文转中文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 再进行前二后二的筛查
			i = 1
			while i > 0 and i < len(a) - 2:
				if is_contain_chinese(str(find_pre2(a, i))) and containenglish(str(find_next2(a, i))):  # 从中文转英文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_pre2(a, i))) and is_contain_num(str(find_next2(a, i))):  # 从中文转数字
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next2(a, i))) and is_contain_num(str(find_pre2(a, i))):  # 从数字转中文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_pre2(a, i))) and containenglish(str(find_next2(a, i))):  # 从数字转英文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_next2(a, i))) and containenglish(str(find_pre2(a, i))):  # 从英文转数字
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next2(a, i))) and containenglish(str(find_pre2(a, i))):  # 从英文转中文
					a = list(a)
					a.insert(i + 2, '*')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 最后进行一次前二后三的检查，这个比较少见，只在「武力⋯⋯”(1974」这个情况中存在
			i = 1
			while i > 0 and i < len(a) - 3:
				if is_contain_chinese(str(find_pre2(a, i))) and containenglish(str(find_next3(a, i))):  # 从中文转英文
					a = list(a)
					a.insert(i + 3, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_pre2(a, i))) and is_contain_num(str(find_next3(a, i))):  # 从中文转数字
					a = list(a)
					a.insert(i + 3, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next3(a, i))) and is_contain_num(str(find_pre2(a, i))):  # 从数字转中文
					a = list(a)
					a.insert(i + 3, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_pre2(a, i))) and containenglish(str(find_next3(a, i))):  # 从数字转英文
					a = list(a)
					a.insert(i + 3, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_num(str(find_next3(a, i))) and containenglish(str(find_pre2(a, i))):  # 从英文转数字
					a = list(a)
					a.insert(i + 3, '*')
					a = ''.join(a)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next3(a, i))) and containenglish(str(find_pre2(a, i))):  # 从英文转中文
					a = list(a)
					a.insert(i + 3, '*')
					a = ''.join(a)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 将多个*号替换成一个*。
			a = a.replace('*****', "*")
			a = a.replace('****', "*")
			a = a.replace('***', "*")
			a = a.replace("**", "*")

			# 转换为三个列表（考虑在每个星号之后打上顺序，这样成为了列表后每个元素有一个代码i☆
			b = a.split('*')
			i = 0
			while i >= 0 and i <= len(b) - 1:
				b[i] = str(i + 1), '☆', b[i], '*'
				b[i] = ''.join(b[i])
				i = i + 1
				continue

			b_ch = []  # 中文（待清理）
			for i in range(len(b)):
				b_ch.append(b[i])
			c_en = []  # 英文（待清理）
			for i in range(len(b)):
				c_en.append(b[i])
			d_nu = []  # 数字（待清理）
			for i in range(len(b)):
				d_nu.append(b[i])

			# 读取列表元素中☆之后的元素，定义一个函数
			def qingli(k, i):
				x = k[i]
				z = x.index("☆") + 1
				y = x[z: len(x)]
				return y

			# 执行清理
			n = 0
			while n <= len(b_ch) - 1:
				if containenglish(str(qingli(b_ch, n))) or is_contain_num(str(qingli(b_ch, n))):
					del b_ch[n]  # 中文，除掉英文和数字
					n = n
					continue
				else:
					n = n + 1
					continue

			n = 0
			while n <= len(c_en) - 1:
				if is_contain_chinese(str(qingli(c_en, n))) or is_contain_num(str(qingli(c_en, n))):
					del c_en[n]  # 英文，除掉中文和数字
					n = n
					continue
				else:
					n = n + 1
					continue

			n = 0
			while n <= len(d_nu) - 1:
				if is_contain_chinese(str(qingli(d_nu, n))) or containenglish(str(qingli(d_nu, n))):
					del d_nu[n]  # 数字，除掉中文和英文
					n = n
					continue
				else:
					n = n + 1
					continue

			# 【对中文处理】
			zh = ''.join(b_ch)
			# 去除不必要的中英文符号及空格
			zh = zh.replace(' ', '')
			zh = zh.replace('#', '')
			zh = zh.replace('^', '')
			zh = zh.replace('~', '')
			zh = zh.replace('～', '')

			# 修改一些排版中常见的符号错误
			zh = zh.replace('。。', '。')
			zh = zh.replace('。。。', '……')
			zh = zh.replace('—', "——")
			zh = zh.replace('一一', "——")
			# Black Circle, Katakana Middle Dot, Bullet, Bullet Operator 替换为标准中间点（U+00B7 MIDDLE DOT）
			zh = zh.replace('●', "·")
			zh = zh.replace('・', "·")
			zh = zh.replace('•', "·")
			zh = zh.replace('∙', "·")
			# U+2027 HYPHENATION POINT 替换为中间点（U+00B7 MIDDLE DOT）
			zh = zh.replace('‧', "·")
			# 加重符号、乘号、点号替换为中间点（U+00B7 MIDDLE DOT）
			zh = zh.replace('•', "·")
			zh = zh.replace('·', "·")
			zh = zh.replace('▪', "·")
			# Phoenician Word Separator (U+1091F) to middle dot
			zh = zh.replace('𐤟', "·")
			for i in range(10):
				zh = zh.replace('————————', "——")
				zh = zh.replace('——————', "——")
				zh = zh.replace('————', "——")

			# 将常用英文标点转换为中文标点
			def E_trans_to_C(string):
				E_pun = u',.;:!?[]()<>'
				C_pun = u'，。；：！？【】（）《》'
				table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
				return string.translate(table)

			zh = E_trans_to_C(str(zh))

			# 合成待整合的中文列表
			zh_he = zh.split('*')

			def Q2B(uchar):
				"""单个字符 全角转半角"""
				inside_code = ord(uchar)
				if inside_code == 0x3000:
					inside_code = 0x0020
				else:
					inside_code -= 0xfee0
				if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
					return uchar
				return chr(inside_code)

			# 【对英文处理】将全角英文字母转为半角英文字母，不改变符号的全半角，标点符号（,.;:?!）前面去空格。
			en = ''.join(c_en)

			def is_Qalphabet(uchar):
				"""判断一个unicode是否是全角英文字母"""
				if (uchar >= u'\uff21' and uchar <= u'\uff3a') or (uchar >= u'\uff41' and uchar <= u'\uff5a'):
					return True
				else:
					return False

			def stringpartQ2B(ustring):
				"""把字符串中字母全角转半角"""
				return "".join([Q2B(uchar) if is_Qalphabet(uchar) else uchar for uchar in ustring])

			en = stringpartQ2B(en)

			# One Dot Leader (U+2024) to full stop (U+002E) （句号）
			en = en.replace('․', ".")

			# 去除标点符号前面的空格
			en = list(en)
			i = 0
			while i >= 0 and i < len(en) - 1:
				if en[i] == ',':
					if en[i - 1] == ' ':
						del en[i - 1]
						continue
					else:
						i = i + 1
						continue
				if en[i] == '.':
					if en[i - 1] == ' ':
						del en[i - 1]
						continue
					else:
						i = i + 1
						continue
				if en[i] == ';':
					if en[i - 1] == ' ':
						del en[i - 1]
						continue
					else:
						i = i + 1
						continue
				if en[i] == ':':
					if en[i - 1] == ' ':
						del en[i - 1]
						continue
					else:
						i = i + 1
						continue
				if en[i] == '?':
					if en[i - 1] == ' ':
						del en[i - 1]
						continue
					else:
						i = i + 1
						continue
				if en[i] == '!':
					if en[i - 1] == ' ':
						del en[i - 1]
						continue
					else:
						i = i + 1
						continue
				else:
					i = i + 1
					continue
			en = ''.join(en)

			en_he = en.split('*')

			# 【对数字处理】将全角数字转为半角数字，不改变符号的全半角
			shu = ''.join(d_nu)

			def is_Qnumber(uchar):
				"""判断一个unicode是否是全角数字"""
				if uchar >= u'\uff10' and uchar <= u'\uff19':
					return True
				else:
					return False

			def stringpartQ2B(ustring):
				"""把字符串中数字全角转半角"""
				return "".join(
					[Q2B(uchar) if is_Qnumber(uchar) else uchar for uchar in ustring])

			shu = stringpartQ2B(shu)

			shu_he = shu.split('*')

			# 合在一起（存在大于10的数变成小于2的问题，后面解决）
			he = zh_he + en_he + shu_he

			# 清掉空以及前面的顺序符号
			n = 0
			while n >= 0 and n <= len(he) - 1:
				if he[n] == '':
					he.remove('')
					continue
				else:
					n = n + 1
					continue

			he.sort(key=lambda x: int(x.split('☆')[0]))

			m = 0
			while m >= 0 and m <= len(he) - 1:
				f = he[m]
				g = f.index('☆') + 1
				h = f[g: len(f)]
				he[m] = h
				m = m + 1

			# 将列表转化为字符串相连，这里本可以转化成空格，但是这样会因为分割点问题产生问题，故先整体以"空"合并
			zhong = ''.join(he)

			# 解决因为分块不当带来的括号问题（当括号分到英文块的时候没有被处理到），此处默认全部换成中文括号
			zhong = zhong.replace('(', '（')
			zhong = zhong.replace(')', '）')
			zhong = zhong.replace('[', '【')
			zhong = zhong.replace(']', '】')
			zhong = zhong.replace('<', '《')
			zhong = zhong.replace('>', '》')

			# 清除因为分块不当带来的括号、引号、顿号前后的空格
			zhong = list(zhong)
			i = 0
			while i >= 0 and i < len(zhong) - 1:
				if zhong[i] == '（':
					if zhong[i - 1] == ' ':
						del zhong[i - 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '）':
					if zhong[i - 1] == ' ':
						del zhong[i - 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '、':
					if zhong[i - 1] == ' ':
						del zhong[i - 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '“':
					if zhong[i - 1] == ' ':
						del zhong[i - 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '”':
					if zhong[i - 1] == ' ':
						del zhong[i - 1]
						continue
					else:
						i = i + 1
						continue
				else:
					i = i + 1
					continue

			i = 0
			while i >= 0 and i < len(zhong) - 1:
				if zhong[i] == '（':
					if zhong[i + 1] == ' ':
						del zhong[i + 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '）':
					if zhong[i + 1] == ' ':
						del zhong[i + 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '、':
					if zhong[i + 1] == ' ':
						del zhong[i + 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '“':
					if zhong[i + 1] == ' ':
						del zhong[i + 1]
						continue
					else:
						i = i + 1
						continue
				if zhong[i] == '”':
					if zhong[i + 1] == ' ':
						del zhong[i + 1]
						continue
					else:
						i = i + 1
						continue
				else:
					i = i + 1
					continue

			zhong = ''.join(zhong)

			# 给中英数三者相邻的文本插入空格，给特定的单位符号前后增减空格（注意，如果是探索，不能等号，如果是全局修改，必须<=）
			i = 0
			while i <= len(zhong) - 1:
				if zhong[i] == '¥' and not is_contain_symbol(str(zhong[i - 1])):
					zhong = list(zhong)
					zhong.insert(i, ' ')
					zhong = ''.join(zhong)
					i = i + 2
					continue
				if zhong[i] == '$' and not is_contain_symbol(str(zhong[i - 1])):
					zhong = list(zhong)
					zhong.insert(i, ' ')
					zhong = ''.join(zhong)
					i = i + 2
					continue
				if zhong[i] == "%":
					if zhong[i - 1] == ' ':
						zhong = list(zhong)
						del zhong[i - 1]
						zhong = ''.join(zhong)
						i = i - 1
						continue
					else:
						zhong = list(zhong)
						zhong.insert(i + 1, ' ')
						zhong = ''.join(zhong)
						i = i + 2
						continue
				if zhong[i] == "°":
					if zhong[i - 1] == ' ':
						zhong = list(zhong)
						del zhong[i - 1]
						zhong = ''.join(zhong)
						i = i - 1
						continue
					else:
						zhong = list(zhong)
						zhong.insert(i + 1, ' ')
						zhong = ''.join(zhong)
						i = i + 2
						continue
				else:
					i = i + 1
					continue

			i = 0
			while i >= 0 and i < len(zhong) - 1:
				if is_contain_chinese(str(find_this(zhong, i))) and containenglish(str(find_next(zhong, i))):  # 从中文转英文
					zhong = list(zhong)
					zhong.insert(i + 1, ' ')
					zhong = ''.join(zhong)
					i = i + 1
					continue
				if is_contain_chinese(str(find_this(zhong, i))) and is_contain_num(str(find_next(zhong, i))):  # 从中文转数字
					zhong = list(zhong)
					zhong.insert(i + 1, ' ')
					zhong = ''.join(zhong)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(zhong, i))) and is_contain_num(str(find_this(zhong, i))):  # 从数字转中文
					zhong = list(zhong)
					zhong.insert(i + 1, ' ')
					zhong = ''.join(zhong)
					i = i + 1
					continue
				if is_contain_num(str(find_this(zhong, i))) and containenglish(str(find_next(zhong, i))):  # 从数字转英文
					zhong = list(zhong)
					zhong.insert(i + 1, ' ')
					zhong = ''.join(zhong)
					i = i + 1
					continue
				if is_contain_num(str(find_next(zhong, i))) and containenglish(str(find_this(zhong, i))):  # 从英文转数字
					zhong = list(zhong)
					zhong.insert(i + 1, ' ')
					zhong = ''.join(zhong)
					i = i + 1
					continue
				if is_contain_chinese(str(find_next(zhong, i))) and containenglish(str(find_this(zhong, i))):  # 从英文转中文
					zhong = list(zhong)
					zhong.insert(i + 1, ' ')
					zhong = ''.join(zhong)
					i = i + 1
					continue
				else:
					i = i + 1
					continue

			# 清除连续空格
			zhong = zhong.replace('  ', ' ')
			return zhong

	def activate(self):  # 设置窗口显示
		self.show()
		self.setFocus()
		self.raise_()
		self.activateWindow()

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def keyPressEvent(self, e):  # 当页面显示的时候，按下esc键可关闭窗口
		if e.key() == Qt.Key.Key_Escape.value:
			self.close()

	def cancel(self):  # 设置取消键的功能
		self.close()


class window4(QWidget):  # Customization settings
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):  # 设置窗口内布局
		self.setUpMainWindow()
		self.resize(350, 150)
		self.center()
		self.setWindowTitle('Customization settings')
		self.setFocus()

	def setUpMainWindow(self):
		self.widgeten = QComboBox(self)
		self.widgeten.setEditable(False)
		defalist = ['ChatGPT (Official module)', 'ChatGPT (httpx)']
		self.widgeten.addItems(defalist)
		Which = codecs.open(BasePath + 'which.txt', 'r', encoding='utf-8').read()
		if Which == '0':
			self.widgeten.setCurrentIndex(0)
		if Which == '1':
			self.widgeten.setCurrentIndex(1)
		self.widgeten.currentIndexChanged.connect(self.IndexChange)

		self.leapi = QLineEdit(self)
		self.leapi.setPlaceholderText('API here...')
		Apis = codecs.open(BasePath + 'api.txt', 'r', encoding='utf-8').read()
		if Apis != '':
			self.leapi.setText(Apis)

		self.lemaxtokens = QLineEdit(self)
		self.lemaxtokens.setPlaceholderText('Maxtokens here...(0~1024)')
		maxto = codecs.open(BasePath + 'maxtokens.txt', 'r', encoding='utf-8').read()
		if maxto != '':
			self.lemaxtokens.setText(maxto)

		self.letemp = QLineEdit(self)
		self.letemp.setPlaceholderText('Temperature here...(0~1)')
		temp = codecs.open(BasePath + 'temperature.txt', 'r', encoding='utf-8').read()
		if temp != '':
			self.letemp.setText(temp)

		btn_1 = QPushButton('Save', self)
		btn_1.clicked.connect(self.SaveAPI)
		btn_1.setFixedSize(80, 20)

		qw2 = QWidget()
		vbox2 = QHBoxLayout()
		vbox2.setContentsMargins(0, 0, 0, 0)
		vbox2.addStretch()
		vbox2.addWidget(btn_1)
		vbox2.addStretch()
		qw2.setLayout(vbox2)

		vbox1 = QVBoxLayout()
		vbox1.setContentsMargins(20, 20, 20, 20)
		vbox1.addWidget(self.widgeten)
		vbox1.addWidget(self.leapi)
		vbox1.addWidget(self.lemaxtokens)
		vbox1.addWidget(self.letemp)
		vbox1.addWidget(qw2)
		self.setLayout(vbox1)

	def IndexChange(self, i):
		if i == 0:
			with open(BasePath + 'which.txt', 'w', encoding='utf-8') as f0:
				f0.write('0')
		if i == 1:
			with open(BasePath + 'which.txt', 'w', encoding='utf-8') as f0:
				f0.write('1')

	def SaveAPI(self):
		with open(BasePath + 'api.txt', 'w', encoding='utf-8') as f1:
			f1.write(self.leapi.text())
		with open(BasePath + 'maxtokens.txt', 'w', encoding='utf-8') as f1:
			f1.write(self.lemaxtokens.text())
		with open(BasePath + 'temperature.txt', 'w', encoding='utf-8') as f1:
			f1.write(self.letemp.text())
		self.close()

	def center(self):  # 设置窗口居中
		qr = self.frameGeometry()
		cp = self.screen().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def keyPressEvent(self, e):  # 当页面显示的时候，按下esc键可关闭窗口
		if e.key() == Qt.Key.Key_Escape.value:
			self.close()

	def activate(self):  # 设置窗口显示
		self.show()
		self.setFocus()
		self.raise_()
		self.activateWindow()

	def cancel(self):  # 设置取消键的功能
		self.close()


class window5(QWidget):  # Customization settings
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):  # 设置窗口内布局
		self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
		self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
		SCREEN_WEIGHT = int(self.screen().availableGeometry().width())
		SCREEN_HEIGHT = int(self.screen().availableGeometry().height())
		self.setFixedSize(SCREEN_WEIGHT, SCREEN_HEIGHT)
		self.cleanup_handler = QObjectCleanupHandler()

		home_dir = str(Path.home())
		tarname1 = "Library"
		fulldir1 = os.path.join(home_dir, tarname1)
		tarname2 = "Safari"
		fulldir2 = os.path.join(fulldir1, tarname2)
		tarname3 = 'Bookmarks.plist'
		fulldir3 = os.path.join(fulldir2, tarname3)
		plist = readPlist(fulldir3)

		try:
			FoldAllList = plist['Children']
			self.FoldList = []
			self.URLList = []
			for i in range(len(FoldAllList)):
				try:
					ItemAllList = plist['Children'][i]['Children']
					ItemList = []
					ItemURLList = []
					for t in range(len(ItemAllList)):
						ItemURL = ItemAllList[t]['URLString']
						ItemTLT = ItemAllList[t]['URIDictionary']['title']
						if len(ItemTLT) > 15:
							ItemTLT = ItemTLT[:15] + '...'
						ItemList.append(ItemTLT)
						ItemURLList.append(ItemURL)
					self.FoldList.append(ItemList)
					self.URLList.append(ItemURLList)
				except:
					self.FoldList.append('XXX')
					self.URLList.append('XXX')
			while 'XXX' in self.FoldList:
				self.FoldList.remove('XXX')
			while 'XXX' in self.URLList:
				self.URLList.remove('XXX')

			self.topFiller = QWidget()
			self.list_widgets = []
			for f in range(len(self.FoldList)):
				self.MapButton = QListWidget(self.topFiller)
				self.MapButton.setMaximumWidth(250)
				self.MapButton.setMinimumHeight(SCREEN_HEIGHT - 20)
				self.MapButton.addItems(self.FoldList[f])
				self.MapButton.move(f * 260, 10)
				self.MapButton.itemClicked.connect(self.list_widget_action)
				self.list_widgets.append(self.MapButton)
			self.topFiller.setMinimumSize(len(self.FoldList)*260, SCREEN_HEIGHT-10)  #######设置滚动条的尺寸
			##创建一个滚动条
			self.scroll = QScrollArea()
			self.scroll.setWidget(self.topFiller)
			self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

			self.w = QWidget()
			self.vbox = QVBoxLayout()
			self.vbox.setContentsMargins(10, 0, 10, 0)
			self.vbox.addWidget(self.scroll)
			self.w.setLayout(self.vbox)
			self.w.setObjectName("Main")

			self.hbox = QHBoxLayout()
			self.hbox.setContentsMargins(0, 0, 0, 0)
			self.hbox.addWidget(self.w)
			self.setLayout(self.hbox)
		except:
			CMD = '''
				on run argv
				  display notification (item 2 of argv) with title (item 1 of argv)
				end run
				'''
			self.notify(CMD, "Banana: Webpage Archiver",
						f'You have to grant Banana "Full Disk Access" to enable this function!')

	def notify(self, CMD, title, text):
		subprocess.call(['osascript', '-e', CMD, title, text])

	def list_widget_action(self, item):
		clicked_widget = self.sender()  # 获取发出信号的 QListWidget
		list_index = self.list_widgets.index(clicked_widget)  # 获取在 list_widgets 列表中的位置
		target_num = self.FoldList[list_index].index(item.text())
		target_url = self.URLList[list_index][target_num]
		webbrowser.open(target_url)
		# 遍历所有QListWidget
		for list_widget in self.list_widgets:
			# 如果是触发操作的QListWidget，则跳过
			if list_widget == clicked_widget:
				continue
			# 清除其他QListWidgets的选中记录
			list_widget.clearSelection()
		# 在处理完其他QListWidgets的选中记录后，清除触发操作的QListWidget的选中记录
		clicked_widget.clearSelection()
		self.close()
		action9.setChecked(False)
		btna6.setChecked(False)

	def activate(self):
		if action9.isChecked():
			try:
				self.cleanup_handler.add(self.scroll)
				self.cleanup_handler.add(self.vbox)
				self.cleanup_handler.add(self.w)
				self.cleanup_handler.add(self.hbox)
				self.cleanup_handler.clear()

				SCREEN_WEIGHT = int(self.screen().availableGeometry().width())
				SCREEN_HEIGHT = int(self.screen().availableGeometry().height())
				self.setFixedSize(SCREEN_WEIGHT, SCREEN_HEIGHT)

				home_dir = str(Path.home())
				tarname1 = "Library"
				fulldir1 = os.path.join(home_dir, tarname1)
				tarname2 = "Safari"
				fulldir2 = os.path.join(fulldir1, tarname2)
				tarname3 = 'Bookmarks.plist'
				fulldir3 = os.path.join(fulldir2, tarname3)
				plist = readPlist(fulldir3)

				FoldAllList = plist['Children']
				self.FoldList = []
				self.URLList = []
				for i in range(len(FoldAllList)):
					try:
						ItemAllList = plist['Children'][i]['Children']
						ItemList = []
						ItemURLList = []
						for t in range(len(ItemAllList)):
							ItemURL = ItemAllList[t]['URLString']
							ItemTLT = ItemAllList[t]['URIDictionary']['title']
							if len(ItemTLT) > 15:
								ItemTLT = ItemTLT[:15] + '...'
							ItemList.append(ItemTLT)
							ItemURLList.append(ItemURL)
						self.FoldList.append(ItemList)
						self.URLList.append(ItemURLList)
					except:
						self.FoldList.append('XXX')
						self.URLList.append('XXX')
				while 'XXX' in self.FoldList:
					self.FoldList.remove('XXX')
				while 'XXX' in self.URLList:
					self.URLList.remove('XXX')

				self.topFiller = QWidget()
				self.list_widgets = []
				for f in range(len(self.FoldList)):
					self.MapButton = QListWidget(self.topFiller)
					self.MapButton.setMaximumWidth(250)
					self.MapButton.setMinimumHeight(SCREEN_HEIGHT - 20)
					self.MapButton.addItems(self.FoldList[f])
					self.MapButton.move(f * 260, 10)
					self.MapButton.itemClicked.connect(self.list_widget_action)
					self.list_widgets.append(self.MapButton)
				self.topFiller.setMinimumSize(len(self.FoldList) * 260, SCREEN_HEIGHT - 10)  #######设置滚动条的尺寸
				##创建一个滚动条
				self.scroll = QScrollArea()
				self.scroll.setWidget(self.topFiller)
				self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

				self.w = QWidget()
				self.vbox = QVBoxLayout()
				self.vbox.setContentsMargins(10, 0, 10, 0)
				self.vbox.addWidget(self.scroll)
				self.w.setLayout(self.vbox)
				self.w.setObjectName("Main")

				self.hbox = QHBoxLayout()
				self.hbox.setContentsMargins(0, 0, 0, 0)
				self.hbox.addWidget(self.w)
				self.setLayout(self.hbox)

				self.setStyleSheet(style_sheet_ori)
				self.show()
				self.setFocus()
				self.raise_()
				btna6.setChecked(True)
			except:
				CMD = '''
					on run argv
					  display notification (item 2 of argv) with title (item 1 of argv)
					end run
					'''
				self.notify(CMD, "Banana: Webpage Archiver",
							f'You have to grant Banana "Full Disk Access" to enable this function!')
		if not action9.isChecked():
			self.close()
			btna6.setChecked(False)

	def activate2(self):
		if btna6.isChecked():
			try:
				self.cleanup_handler.add(self.scroll)
				self.cleanup_handler.add(self.vbox)
				self.cleanup_handler.add(self.w)
				self.cleanup_handler.add(self.hbox)
				self.cleanup_handler.clear()

				SCREEN_WEIGHT = int(self.screen().availableGeometry().width())
				SCREEN_HEIGHT = int(self.screen().availableGeometry().height())
				self.setFixedSize(SCREEN_WEIGHT, SCREEN_HEIGHT)

				home_dir = str(Path.home())
				tarname1 = "Library"
				fulldir1 = os.path.join(home_dir, tarname1)
				tarname2 = "Safari"
				fulldir2 = os.path.join(fulldir1, tarname2)
				tarname3 = 'Bookmarks.plist'
				fulldir3 = os.path.join(fulldir2, tarname3)
				plist = readPlist(fulldir3)

				FoldAllList = plist['Children']
				self.FoldList = []
				self.URLList = []
				for i in range(len(FoldAllList)):
					try:
						ItemAllList = plist['Children'][i]['Children']
						ItemList = []
						ItemURLList = []
						for t in range(len(ItemAllList)):
							ItemURL = ItemAllList[t]['URLString']
							ItemTLT = ItemAllList[t]['URIDictionary']['title']
							if len(ItemTLT) > 15:
								ItemTLT = ItemTLT[:15] + '...'
							ItemList.append(ItemTLT)
							ItemURLList.append(ItemURL)
						self.FoldList.append(ItemList)
						self.URLList.append(ItemURLList)
					except:
						self.FoldList.append('XXX')
						self.URLList.append('XXX')
				while 'XXX' in self.FoldList:
					self.FoldList.remove('XXX')
				while 'XXX' in self.URLList:
					self.URLList.remove('XXX')

				self.topFiller = QWidget()
				self.list_widgets = []
				for f in range(len(self.FoldList)):
					self.MapButton = QListWidget(self.topFiller)
					self.MapButton.setMaximumWidth(250)
					self.MapButton.setMinimumHeight(SCREEN_HEIGHT - 20)
					self.MapButton.addItems(self.FoldList[f])
					self.MapButton.move(f * 260, 10)
					self.MapButton.itemClicked.connect(self.list_widget_action)
					self.list_widgets.append(self.MapButton)
				self.topFiller.setMinimumSize(len(self.FoldList) * 260, SCREEN_HEIGHT - 10)  #######设置滚动条的尺寸
				##创建一个滚动条
				self.scroll = QScrollArea()
				self.scroll.setWidget(self.topFiller)
				self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

				self.w = QWidget()
				self.vbox = QVBoxLayout()
				self.vbox.setContentsMargins(10, 0, 10, 0)
				self.vbox.addWidget(self.scroll)
				self.w.setLayout(self.vbox)
				self.w.setObjectName("Main")

				self.hbox = QHBoxLayout()
				self.hbox.setContentsMargins(0, 0, 0, 0)
				self.hbox.addWidget(self.w)
				self.setLayout(self.hbox)

				self.setStyleSheet(style_sheet_ori)
				self.show()
				self.setFocus()
				self.raise_()
				action9.setChecked(True)
			except:
				CMD = '''
					on run argv
					  display notification (item 2 of argv) with title (item 1 of argv)
					end run
					'''
				self.notify(CMD, "Banana: Webpage Archiver",
							f'You have to grant Banana "Full Disk Access" to enable this function!')
		if not btna6.isChecked():
			self.close()
			action9.setChecked(False)


style_sheet_ori = '''
	QTabWidget::pane {
		border: 1px solid #ECECEC;
		background: #ECECEC;
		border-radius: 9px;
}
	QTableWidget{
		border: 1px solid grey;  
		border-radius:4px;
		background-clip: border;
		background-color: #FFFFFF;
		color: #000000;
		font: 14pt Helvetica;
}
	QWidget#Main {
		border: 1px solid #ECECEC;
		background: #ECECEC;
		border-radius: 9px;
}
	QPushButton{
		border: 1px outset grey;
		background-color: #FFFFFF;
		border-radius: 4px;
		padding: 1px;
		color: #000000
}
	QPushButton:pressed{
		border: 1px outset grey;
		background-color: #0085FF;
		border-radius: 4px;
		padding: 1px;
		color: #FFFFFF
}
	QPlainTextEdit{
		border: 1px solid grey;  
		border-radius:4px;
		padding: 1px 5px 1px 3px; 
		background-clip: border;
		background-color: #F3F2EE;
		color: #000000;
		font: 14pt Times New Roman;
}
	QPlainTextEdit#edit{
		border: 1px solid grey;  
		border-radius:4px;
		padding: 1px 5px 1px 3px; 
		background-clip: border;
		background-color: #FFFFFF;
		color: rgb(113, 113, 113);
		font: 14pt Helvetica;
}
	QTableWidget#small{
		border: 1px solid grey;  
		border-radius:4px;
		background-clip: border;
		background-color: #F3F2EE;
		color: #000000;
		font: 14pt Times New Roman;
}
	QLineEdit{
		border-radius:4px;
		border: 1px solid gray;
		background-color: #FFFFFF;
}
	QTextEdit{
		border: 1px solid grey;  
		border-radius:4px;
		padding: 1px 5px 1px 3px; 
		background-clip: border;
		background-color: #F3F2EE;
		color: #000000;
		font: 14pt Times New Roman;
}
	QListWidget{
		border: 1px solid grey;  
		border-radius:4px;
		padding: 1px 5px 1px 3px; 
		background-clip: border;
		background-color: #F3F2EE;
		color: #000000;
		font: 14pt Times New Roman;
}
	QListWidget::item{
		padding-top: 10px;
		padding-bottom: 10px;
}
	QListWidget::item:hover{
		border: 4px outset black;
		background-color: transparent;
		border-radius: 12px;
		padding: 1px;
		color: #000000
}
'''

if __name__ == '__main__':
	w1 = window_about()  # about
	w2 = window_update()  # update
	w3 = window3()  # main1
	w3.setAutoFillBackground(True)
	p = w3.palette()
	p.setColor(w3.backgroundRole(), QColor('#ECECEC'))
	w3.setPalette(p)
	w4 = window4()  # CUSTOMIZING
	w5 = window5()  # Safari
	action1.triggered.connect(w1.activate)
	action2.triggered.connect(w2.activate)
	action3.triggered.connect(w3.activate)
	action4.triggered.connect(w3.archivethis)
	action5.triggered.connect(w3.embeditem)
	action6.triggered.connect(w3.showchat)
	action7.triggered.connect(w4.activate)
	action8.triggered.connect(w3.showdelbutton)
	action9.triggered.connect(w5.activate)
	btna4.triggered.connect(w3.activate)
	btna5.triggered.connect(w3.archivethis)
	btna6.triggered.connect(w5.activate2)
	app.setStyleSheet(style_sheet_ori)
	app.exec()
