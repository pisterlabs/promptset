# Imports
from wave import Error
import speech_recognition as sr
from datetime import datetime
import random
import sys
import openai
import subprocess
import webbrowser
import os
import time
import pyodbc
import threading
import pyautogui
import sys
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import keyboard




def init():
    app = QApplication(sys.argv)

    initt = QMainWindow()
    initt.setWindowTitle("Inicialização")
    initt.setWindowState(initt.windowState() | QtCore.Qt.WindowFullScreen)

    # Define a cor de fundo como preto
    palette = initt.palette()
    palette.setColor(initt.backgroundRole(), QColor(0, 0, 0))
    initt.setPalette(palette)

    # Mostra o GIF do centro da tela
    gif = QLabel(initt)
    movie = QMovie(r"Interface\Graficos\GEEKtyper_com - Aperture Science.gif")
    gif.setMovie(movie)
    window_frame = initt.frameGeometry()
    center_point = QDesktopWidget().availableGeometry().center()
    window_frame.moveCenter(center_point)
    initt.move(window_frame.topLeft())
    gif.setAlignment(Qt.AlignCenter)
    gif.setGeometry(0, 0, window_frame.width(), window_frame.height())

    # Função para verificar se o último quadro foi alcançado e imprimir a mensagem
    def check_last_frame(frame):
        if frame == movie.frameCount() - 1:
            print("Reproduzido")
            return True

    # Conecta o sinal 'frameChanged' do QMovie à função check_last_frame
    movie.frameChanged.connect(check_last_frame)

    initt.show()
    movie.start()
    sys.exit(app.exec())
