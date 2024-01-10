import os
import openai

import time
import numpy as np

import torch

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

import paho.mqtt.client as mqtt

from googletrans import Translator
google_translator = Translator()

my_global_mssg = ''
broker = "158.42.170.142" 
port = 1883

openai.api_key = ""

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("topic/person_to_bot")

def on_message(client, userdata, msg):
    
    # t = datetime.now()
    global my_global_mssg

    message = msg.payload.decode()
    my_global_mssg = message

    client.disconnect()
    client.loop_stop()

start_sequence = "Yolo:"
restart_sequence = "Human: "

# message = "The following is a conversation with Yolo. Yolo is helpful, creative, clever, and very friendly." \
# 		  "Human: Hello, who are you?" \
# 		  "Yolo: I am an AI created by OpenAI. My name is Yolo. How can I help you today? "
          
message = "La siguiente conversación es con un hombre de 35 años que se llama Antonio y es muy amable, creativo e inteligente.\n \nYo: Hola, encantado de conocerte. \nAntonio: Hola, soy Antonio. Es un placer conocerte. \nYo: ¿Cómo estás? \nAntonio: Muy bien, muy contento. ¿Y tú cómo estás? \nYo: ¡Muy bien! \nAntonio: Me alegro. ¿Te gustaría preguntarme algo?"

initial_tokens = 50
while True:
    
    # ############################
    # ### RECIEVE MQTT MESSAGE ###
    # ############################

    client = mqtt.Client()
    client.connect("158.42.170.142", 1883)

    client.on_connect = on_connect
    client.on_message = on_message

    print("Loop forever")

    client.loop_forever()  
    
    # ###################
    # ### TRANSLATION ###
    # ###################
    # Here the message is translated from spanish to english.

    person_message_sp = my_global_mssg
    # x = google_translator.translate(person_message_sp, dest='es')
    
    # x = google_translator.translate(person_message_sp) # COMENTADO POR LUCIA
    # person_message = x.text # COMENTADO POR LUCIA
    person_message = person_message_sp # AÑADIDO POR LUCIA

    # message = message + "Human: " + person_message # COMENTADO POR LUCIA
    message = message + "\nYo: " + person_message + "\nAntonio:" # MODIFICADO POR LUCIA

    print("***** Message *****")
    print(message)

    t0 = time.time()
    
    response = openai.Completion.create(
      engine="davinci",
      prompt=message,
	  temperature=0.9,
      max_tokens=initial_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=["Yo:"] # stop=["Human:"] # MODIFICADO POR LUCIA
    )
    bot_answer = response["choices"][0]["text"]

    # print("Time of the answer", np.round(time.time()-t0, 5), "s") # COMENTADO POR LUCIA
    
    # ###################
    # ### TRANSLATION ###
    # ###################
    # Here the message is translated from english to spanish.
    
    message = message + bot_answer
    print("bot_answer", message)
    

    # bot_answer = bot_answer.replace("Yolo:", "") if "Yolo:" in bot_answer else bot_answer # COMENTADO POR LUCIA
    bot_answer = bot_answer.replace("Antonio:", "") if "Antonio:" in bot_answer else bot_answer # AÑADIDO POR LUCIA

    # x = google_translator.translate(bot_answer, dest='es') # COMENTADO POR LUCIA
    # bot_message_sp = x.text # COMENTADO POR LUCIA
    bot_message_sp =  bot_answer # AÑADIDO POR LUCIA

    print("Message of the bot: ", bot_message_sp)
    
    client = mqtt.Client()
    client.connect("158.42.170.142", 1883)
    client.publish("topic/bot_to_person", bot_message_sp)
    time.sleep(0.5)
    
    my_global_mssg = ""