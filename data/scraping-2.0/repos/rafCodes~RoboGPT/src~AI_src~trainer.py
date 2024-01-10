# Raphael Fortuna (raf269) 
# Rabail Makhdoom (rm857) 
# Final Project Report
# Lab 403, Lab Section:  4:30pm-7:30pm on Thursdays 

from robot_core import full_cycle_demo
from ai_config import robot_prompt, trainer_prompt
from chat_conversation import openai_chat
from sensor_data_generator import generate_sensor_data
from utils import *

# class to use gpt to train the prompt for robot_demo.py
class trainer_gpt:

    def __init__(self, prompt = trainer_prompt, max_tokens = 2000, debug = False):

        # two instances of openai_chat
        self.chat_instance = openai_chat(system_prompt = prompt, max_tokens = max_tokens, debug = debug)
        self.robot_chat_instance = full_cycle_demo(voice_on = False, generate_sensor_data=generate_sensor_data)

    def train_prompt(self):
        """ train the prompt """

        # start the chat
        self.chat_instance.init_chat()

        # get the robot's response
        robot_response_to_trainer = self.robot_chat_instance.intialize_core()

        i = 0

        while i < 8:
            # send the robot response to the bot

            trainer_response_to_robot = self.chat_instance.voice_chat(robot_response_to_trainer, False)
            robot_response_to_trainer = self.robot_chat_instance._run_one_demo_cycle(trainer_response_to_robot)
            i+= 1

        # now that the training is done, get feedback from the trainer

        evaluation_text_prompt = "The robot evaluation has been completed, please provide feedback on the robot's performance and how the prompt can be improved with specific examples of where to change the prompt."

        trainer_chat = self.chat_instance.voice_chat(evaluation_text_prompt, False)

        with ColorText("yellow") as colorPrinter:
            colorPrinter.print('\n\n\n\n\n\n\n')
            colorPrinter.print("########################################################")
            colorPrinter.print("###################### FEEDBACK ########################")
            colorPrinter.print("########################################################")
            colorPrinter.print(trainer_chat)

if __name__ == "__main__":
    train = trainer_gpt()
    train.train_prompt()