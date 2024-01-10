#!/usr/bin/env python
import os
import openai
from jinja2 import Environment, FileSystemLoader, select_autoescape
from transformers import GPT2TokenizerFast
import paho.mqtt.client as mqtt
import datetime
import re
import threading

openai.api_key = os.getenv("OPENAI_API_KEY")

env = Environment(
  loader=FileSystemLoader("."),
  autoescape=select_autoescape()
)
template = env.get_template("prompt.txt")


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

MAX_TOKENS = 256

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("edi/cmd/assistant")


timers = []

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global timers
    # for timer in timers:
    #   timer.cancel()
    # timers = []
    prompt = template.render(
      time=f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}]",
      topic=msg.topic,
      payload=msg.payload.decode("utf-8")
    )
    cost = f"{(len(tokenizer(prompt, max_length=2048, truncation=True)['input_ids']) + MAX_TOKENS) * 0.06 / 1000:.2f}"
    prompt = f"{ prompt }\n// This'll cost around ${ cost }"
    print(f"This will cost about { cost } dollars")
    print("Prompt:", prompt)
    completion = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      max_tokens=MAX_TOKENS,
      temperature=0.7,
      stop="edi/cmd/assistant"
    )
    for line in completion.choices[0].text.splitlines():
      print("Line:", line)
      components = re.search(r"\[(.+?)\] (.+?) ({.+})", line)
      if components is not None:
        components = components.groups()
        time = datetime.datetime.fromisoformat(components[0])
        topic = components[1]
        message = components[2]
        def send_message(topic, message):
          def callback():
            client.publish(topic, message)
          return callback
        timer = threading.Timer((datetime.datetime.now(datetime.timezone.utc) - time).total_seconds(), send_message(topic, message))
        timer.start()
        timers.append(timer)
        print("Scheduled message for", time)
        print("OwO:", components)
    print("Completion:", completion.choices[0].text )

def main():
  client = mqtt.Client()
  client.on_connect = on_connect
  client.on_message = on_message

  client.connect("mqtt", 1883, 60)

  # Blocking call that processes network traffic, dispatches callbacks and
  # handles reconnecting.
  # Other loop*() functions are available that give a threaded interface and a
  # manual interface.
  client.loop_forever()



if __name__ == '__main__':
  main()
