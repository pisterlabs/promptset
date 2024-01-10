#!/usr/bin/python

# Importamos la libreria de PySerial
import serial
import pyttsx3 as voz
import speech_recognition as sr
import subprocess as sub
from datetime import datetime
import openai
import time



abc="abcdefghijklmnopqrstuvwxyz"
ABC="ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Abrimos el puerto de la placa a 9600
PuertoSerie = serial.Serial('COM8', 9600)

while True:
    print("Mover")
    sArduino = PuertoSerie.write(b'123;')
    time.sleep(1)
    print("Parar")
    sArduino = PuertoSerie.write(b'1218;')
    time.sleep(1)
    print("Mover")
    sArduino = PuertoSerie.write(b'1')
    time.sleep(1)
    print("Parar")
    sArduino = PuertoSerie.write(b'\n')
    print(str(PuertoSerie.readline()))











