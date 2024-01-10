import re
import threading
import socket
import heapq
import time
#import serial

from gpiozero import Servo

#These are the 5th and 6th pins from the top left with usb facing downwards
#Use servo.value = val to set the servos position and delays to make things smooth.
#Range is -1 max speed in one direction and 1 is max speed in other direction


Right = Servo(27)
Left = Servo(17)


class Receiver:
  def __init__(self, host, port):
    self.commands = []
    self.executions = []
    self.transmit = []
    self.timer = 0
    self.HOST = host
    self.PORT = port

    #connecting using scokets
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    #managing error exception
    try:
      self.s.connect((self.HOST, self.PORT))
    except socket.error:
        print ('Bind failed ')

    print('Connected')
    print('Socket awaiting messages')
    
  
  """Method receives commands from client and add to queue"""
  def telemetryReceive(self):
    # Commands must be in form "PRIORITY|{COMMAND}|TIMESTAMP|CHECKSUM"
    # awaiting for message
    while True:
      action = self.s.recv(1024).decode('UTF-8')
      if len(action) > 0:
        heapq.heappush(self.commands,action)
        print("Received|"+action)
        heapq.heappush(self.transmit,"Received|"+action)
          
  """Method checks commands from queue and adds to execution queue"""
  def checkCommand(self):
    while True:
      if len(self.commands) > 0:
        popped = heapq.heappop(self.commands) #gets smallest value command
        heapq.heappush(self.transmit, "Correct|"+popped)
        heapq.heappush(self.executions, popped)
        print(self.transmit)
        print(self.executions)
        
  def telemetryTransmit(self):
    print("Nothing")
    while True:
      if len(self.transmit) > 0:
        print("Transmit queue", self.transmit)
        self.s.send(bytes(heapq.heappop(self.transmit),'utf-8'))    
  
  def moveRobot(self, command):
    arr = [int(i) for i in command.split(" ")]
    Left.value = arr[0]
    Right.value = arr[1]
    time.sleep(0.1)

  def execute(self):
    while True:
      if len(self.executions) > 0:
        command = heapq.heappop(self.executions)
        print(command)
        heapq.heappush(self.transmit, "Executed|"+command)
        if "move" in command:
            self.moveRobot(command[5:])
        # if command == "Password A_on_LED":
          # if onAndOfffLed():
          #   print("Executed: ", command)
          # else:
          #   print("Did not execute correctly ", command)
        print("Inside execute",self.executions)
        time.sleep(5)

  def chatBot(self):
    import os
import openai

openai.organization = "org-suTf2NBidIX4WHsJYzbuwbME"
file = open("/home/pi/Desktop/openai.txt")
openai.api_key = file.readline().strip()
openai.Model.list()

# pip install SpeechRecognition
# brew install portaudio
# pip install pyaudio
# pip install pyttsx3


# Python program to translate
# speech to text and text to speech

import os
os.environ['PA_ALSA_PLUGHW']='1'

import speech_recognition as sr
from gtts import gTTS
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()
engine = pyttsx3.init("espeak")

# Function to convert text to
# speech

engine.setProperty('rate', 200)
engine.setProperty('volume', 1)
engine.setProperty('voice', "en-us")

def SpeakText(command):
	# Initialize the engine
    engine.say(command)
    engine.runAndWait()


# Loop infinitely for user to
# speak

while (1):

	# Exception handling to handle
	# exceptions at the runtime
    try:

		# use the microphone as source for input.
        #device_index=2
        with sr.Microphone() as source2:

			# wait for a second to let the recognizer
			# adjust the energy threshold based on
			# the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)

			# listens for the user's input
            audio2 = r.listen(source2, phrase_time_limit=5)
			# Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()


            #Query the API
            response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=MyText,
            temperature=0,
            max_tokens=150,

            )
            print(f"you said: {MyText}")
            print(f"response:")
            query = response.choices[0].text
            query = query.replace("John", "Baymax")
            print(query)
            SpeakText(query)

    except sr.RequestError as e:
        print()
        #print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        #print("unknown error occurred")
        print()

  def gaitGen(self):
    print("Nothing")

  def sensorData(self):
    var = ""
    # Test
  #  print("Inside")
  #  ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
  #  ser.reset_input_buffer()
  #  while True:
  #    print("Inside data")
  ##    # Read from Arduinos to know what motors and sensors there are
    #  ser.write("Send ****s plz\n".encode('utf-8'))
    #  line = ser.readline().decode('utf-8').rstrip()
    #  print(line)

  def runSimul(self):
    threading.Thread(target=self.telemetryReceive).start()
    threading.Thread(target=self.checkCommand).start()
    threading.Thread(target=self.telemetryTransmit).start()
    threading.Thread(target=self.execute).start()

    # threading.Thread(target=self.sensorData).start()

    # threading.Thread(target=self.balance).start()
    # threading.Thread(target=self.gaitGen).start()
    # threading.Thread(target=self.comuterVision).start()

def startBoot():
  simulation = Receiver('172.20.10.2',12345)
  simulation.runSimul()
