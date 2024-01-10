import os

import openai
import paho.mqtt.client as mqtt

from dotenv.main import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# prompt
PROMPT = os.getenv("PROMPT")

# MQTT broker details
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_SUB_TOPIC = os.getenv("MQTT_SUB_TOPIC")
MQTT_PUB_TOPIC = os.getenv("MQTT_PUB_TOPIC")

# MQTT client initialization
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)


def generate_prompt(content):
    return PROMPT.capitalize().format(
        content.capitalize()
    )


# Define on_connect callback
def on_connect(client, userdata, flags, rc):
    mqtt_client.subscribe(MQTT_SUB_TOPIC)


# Define on_message callback
def on_message(client, userdata, msg):
    response = openai.Completion.create(
        model=OPENAI_MODEL,
        prompt=generate_prompt(msg.payload),
        temperature=0.5,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Extract response text from JSON response
    response_text = response.choices[0].text
    # Publish response to MQTT topic
    mqtt_client.publish(MQTT_PUB_TOPIC, response_text)


# Set on_connect and on_message callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
mqtt_client.loop_forever()
