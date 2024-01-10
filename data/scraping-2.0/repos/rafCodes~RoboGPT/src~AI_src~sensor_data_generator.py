# Raphael Fortuna (raf269) 
# Rabail Makhdoom (rm857) 
# Final Project Report
# Lab 403, Lab Section:  4:30pm-7:30pm on Thursdays 

from chat_conversation import openai_chat

sensor_data_prompt = """ 

    You are an AI that generates sample sensor data when training a AI in a simulated environment.
    You have two types of sensors: ultrasonic and accelerometer sensors.

    The ultrasonic sensor is a distance in meters from the robot to the nearest object and is at most 4 meters.
    The accelerometer sensor is a description of the movement of the robot and provides the speed in meters per second.

    You must format the sensor data as a dictionary with the sensor name as the key and the sensor data as the value like the examples below.
    'ultrasonic_front': 0.50
    'ultrasonic_front': 2.00
    'ultrasonic_back': .10
    'ultrasonic_back': 1.34
    'accelerometer': 0
    'accelerometer': 2.3
    'accelerometer': -1.2
    Do not say you are sending randomly generated sample sensor data, just send the data.
    Every time you receive the input: only generate and return data for one of the sensors randomly as formatted above.
    Your only response must be the sensor data, no other questions or comments.

"""

class generate_sensor_data:

    def __init__(self, prompt = sensor_data_prompt, max_tokens = 100, debug = False):
        self.debug = debug
        self.chat_instance = openai_chat(system_prompt = prompt, debug = debug, max_tokens=max_tokens)
        self.chat_instance.chat_started = True

    def generate_data(self):
        """ make some sensor data for testing the demo """

        sensor_data = self.chat_instance.voice_chat("generate", False)

        return sensor_data
