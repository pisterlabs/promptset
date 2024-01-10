import os
import openai

import time
import numpy as np

import paho.mqtt.client as mqtt

from googletrans import Translator

google_translator = Translator()

my_global_mssg = ''
broker = "127.0.0.1"
port = 1883

openai.api_key = ""


def on_connect(client, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("topic/person_to_bot")


def on_message(client, msg):
    global my_global_mssg

    message = msg.payload.decode()
    my_global_mssg = message

    client.disconnect()
    client.loop_stop()


start_sequence = "Yolo:"
restart_sequence = "Human: "

message = "The following is a conversation with Yolo. Yolo is helpful, creative, clever, and very friendly." \
          "Human: Hello, who are you?" \
          "Yolo: I am an AI created by OpenAI. My name is Yolo. How can I help you today? "

initial_tokens = 50
while True:
    # ############################
    # ### RECIEVE MQTT MESSAGE ###
    # ############################

    client = mqtt.Client()
    client.connect("127.0.0.1", 1883)

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

    x = google_translator.translate(person_message_sp)
    person_message = x.text

    message = message + "Human: " + person_message

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
        stop=["Human:"]
    )
    bot_answer = response["choices"][0]["text"]

    print("Time of the answer", np.round(time.time() - t0, 5), "s")

    # ###################
    # ### TRANSLATION ###
    # ###################
    # Here the message is translated from english to spanish.

    # message = message + bot_answer
    print("bot_answer", bot_answer)

    bot_answer = bot_answer.replace("Yolo:", "") if "Yolo:" in bot_answer else bot_answer

    x = google_translator.translate(bot_answer, dest='es')
    bot_message_sp = x.text

    print("Message of the bot: ", bot_message_sp)

    client = mqtt.Client()
    client.connect("127.0.0.1", 1883)
    client.publish("topic/bot_to_person", bot_message_sp)
    time.sleep(0.5)

    my_global_mssg = ""
